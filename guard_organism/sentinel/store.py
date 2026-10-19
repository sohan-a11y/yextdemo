"""Durable memory of the Sentinel: alerts, the windows it saw, requests for a human, and the neurons the Forge produced.

One SQLite file inside the organism's own home directory. It keeps old rows only for `retention_days`. The Forge and the service share it,
so a neuron approved on the command line appears in the running service without a restart.
"""
import json
import time

from lkm.live.store import Store, _j

SCHEMA = """
CREATE TABLE IF NOT EXISTS windows (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, raw TEXT, score REAL, flagged INTEGER DEFAULT 0, label TEXT);
CREATE INDEX IF NOT EXISTS w_ts ON windows(ts);
CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, rule TEXT, source TEXT, severity TEXT, title TEXT, why TEXT,
  evidence TEXT, key TEXT, target TEXT, status TEXT DEFAULT 'open', window_id INTEGER, judged REAL);
CREATE INDEX IF NOT EXISTS a_status ON alerts(status);
CREATE INDEX IF NOT EXISTS a_ts ON alerts(ts);
CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, alert_id INTEGER, kind TEXT, target TEXT, suggestion TEXT, status TEXT DEFAULT 'open');
CREATE TABLE IF NOT EXISTS neurons (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, version INTEGER, state TEXT, purpose TEXT, spec TEXT, weights BLOB,
  metrics TEXT, created REAL, decided_by TEXT, decided REAL, teacher TEXT, baseline TEXT);
CREATE INDEX IF NOT EXISTS n_state ON neurons(state);
CREATE TABLE IF NOT EXISTS event_vectors (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, sentence TEXT, vec BLOB);
CREATE INDEX IF NOT EXISTS ev_ts ON event_vectors(ts);
CREATE TABLE IF NOT EXISTS concepts (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, state TEXT, data TEXT, severity TEXT,
  created REAL, decided_by TEXT, decided REAL, stats TEXT);
CREATE TABLE IF NOT EXISTS blobs (key TEXT PRIMARY KEY, ts REAL, blob BLOB);
"""


class SentinelStore(Store):
    def __init__(self, home):
        super().__init__(home)
        self.db.executescript(SCHEMA)

    # ------------------------------------------------------------------ windows
    def add_window(self, ts, raw, score, flagged):
        with self.lock:
            cur = self.db.execute("INSERT INTO windows(ts,raw,score,flagged) VALUES(?,?,?,?)", (ts, json.dumps(raw, default=_j), score, int(flagged)))
            return cur.lastrowid

    def window(self, wid):
        with self.lock:
            r = self.db.execute("SELECT id,ts,raw,score,flagged,label FROM windows WHERE id=?", (wid,)).fetchone()
        return None if r is None else {"id": r[0], "ts": r[1], "raw": json.loads(r[2]), "score": r[3], "flagged": r[4], "label": r[5]}

    def set_window_label(self, wid, label):
        with self.lock:
            self.db.execute("UPDATE windows SET label=? WHERE id=?", (label, wid))

    def baseline_windows(self, limit=20000):
        """Windows nobody objected to: not flagged, not marked real. These are 'normal for this machine'."""
        with self.lock:
            rows = self.db.execute("SELECT raw FROM windows WHERE flagged=0 AND (label IS NULL OR label!='real') ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows][::-1]

    def n_windows(self):
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM windows").fetchone()[0]

    # ------------------------------------------------------------------ alerts
    def add_alert(self, a, ts, window_id=None):
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO alerts(ts,rule,source,severity,title,why,evidence,key,target,window_id) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (ts, a["rule"], a.get("source", "instinct"), a["severity"], a["title"], a["why"], json.dumps(a.get("evidence", []), default=_j),
                 a["key"], json.dumps(a.get("target", {}), default=_j), window_id))
            return cur.lastrowid

    def alert(self, aid):
        with self.lock:
            r = self.db.execute("SELECT id,ts,rule,source,severity,title,why,evidence,key,target,status,window_id FROM alerts WHERE id=?", (aid,)).fetchone()
        return None if r is None else self._alert(r)

    @staticmethod
    def _alert(r):
        return {"id": r[0], "ts": r[1], "rule": r[2], "source": r[3], "severity": r[4], "title": r[5], "why": r[6], "evidence": json.loads(r[7]),
                "key": r[8], "target": json.loads(r[9]), "status": r[10], "window_id": r[11]}

    def alerts(self, status=None, limit=50):
        q = "SELECT id,ts,rule,source,severity,title,why,evidence,key,target,status,window_id FROM alerts"
        with self.lock:
            rows = self.db.execute(q + (" WHERE status=?" if status else "") + " ORDER BY id DESC LIMIT ?", ((status, limit) if status else (limit,))).fetchall()
        return [self._alert(r) for r in rows]

    def judge(self, aid, verdict, ts=None):
        with self.lock:
            self.db.execute("UPDATE alerts SET status=?, judged=? WHERE id=?", (verdict, ts or time.time(), aid))

    def alert_counts(self):
        with self.lock:
            rows = self.db.execute("SELECT severity,status,COUNT(*) FROM alerts GROUP BY severity,status").fetchall()
        return [{"severity": a, "status": b, "n": c} for a, b, c in rows]

    # ------------------------------------------------------------------ requests to a human
    def add_request(self, alert_id, kind, target, suggestion, ts):
        with self.lock:
            tj = json.dumps(target, default=_j)
            if self.db.execute("SELECT 1 FROM requests WHERE status='open' AND kind=? AND target=?", (kind, tj)).fetchone():
                return
            self.db.execute("INSERT INTO requests(ts,alert_id,kind,target,suggestion) VALUES(?,?,?,?,?)", (ts, alert_id, kind, json.dumps(target, default=_j), suggestion))

    def requests(self, status="open", limit=30):
        with self.lock:
            rows = self.db.execute("SELECT id,ts,alert_id,kind,target,suggestion,status FROM requests WHERE status=? ORDER BY id DESC LIMIT ?", (status, limit)).fetchall()
        return [{"id": r[0], "ts": r[1], "alert_id": r[2], "kind": r[3], "target": json.loads(r[4]), "suggestion": r[5], "status": r[6]} for r in rows]

    def close_request(self, rid, status="done"):
        with self.lock:
            self.db.execute("UPDATE requests SET status=? WHERE id=?", (status, rid))

    # ------------------------------------------------------------------ neurons (the Forge's output). Nothing here ever deletes one.
    def add_neuron(self, name, purpose, spec, weights, metrics, teacher, baseline, state="proposed", ts=None):
        with self.lock:
            v = (self.db.execute("SELECT COALESCE(MAX(version),0) FROM neurons WHERE name=?", (name,)).fetchone()[0]) + 1
            cur = self.db.execute("INSERT INTO neurons(name,version,state,purpose,spec,weights,metrics,created,teacher,baseline) VALUES(?,?,?,?,?,?,?,?,?,?)",
                                  (name, v, state, purpose, json.dumps(spec), weights, json.dumps(metrics, default=_j), ts or time.time(), teacher, json.dumps(baseline, default=_j)))
            return cur.lastrowid, v

    def neurons(self, state=None):
        q = "SELECT id,name,version,state,purpose,spec,metrics,created,decided_by,decided,teacher,baseline FROM neurons"
        with self.lock:
            rows = self.db.execute(q + (" WHERE state=?" if state else "") + " ORDER BY id", ((state,) if state else ())).fetchall()
        return [{"id": r[0], "name": r[1], "version": r[2], "state": r[3], "purpose": r[4], "spec": json.loads(r[5]), "metrics": json.loads(r[6]),
                 "created": r[7], "decided_by": r[8], "decided": r[9], "teacher": r[10], "baseline": json.loads(r[11] or "{}")} for r in rows]

    def neuron_weights(self, nid):
        with self.lock:
            r = self.db.execute("SELECT weights FROM neurons WHERE id=?", (nid,)).fetchone()
        return None if r is None else r[0]

    def set_neuron_state(self, nid, state, by, ts=None):
        with self.lock:
            self.db.execute("UPDATE neurons SET state=?, decided_by=?, decided=? WHERE id=?", (state, by, ts or time.time(), nid))

    def neuron_stamp(self):
        """Changes whenever a neuron is approved, disabled or added, so the running service knows to reload."""
        with self.lock:
            r = self.db.execute("SELECT COUNT(*), COALESCE(MAX(decided),0), COALESCE(SUM(CASE WHEN state='active' THEN id END),0) FROM neurons").fetchone()
        return tuple(r)

    # ------------------------------------------------------------------ what the cortex read (meaning vectors of recent events)
    def add_event_vectors(self, ts, items):
        """items: [(kind, sentence, vector)] -> ids"""
        import numpy as np
        ids = []
        with self.lock:
            for kind, sent, v in items:
                cur = self.db.execute("INSERT INTO event_vectors(ts,kind,sentence,vec) VALUES(?,?,?,?)", (ts, kind, sent[:400], np.asarray(v, np.float16).tobytes()))
                ids.append(cur.lastrowid)
        return ids

    def recent_vectors(self, n=20000):
        import numpy as np
        with self.lock:
            rows = self.db.execute("SELECT id,ts,kind,sentence,vec FROM event_vectors ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        if not rows:
            return [], np.zeros((0, 384), np.float32), [], []
        V = np.array([np.frombuffer(r[4], np.float16) for r in rows], np.float32)
        return [r[0] for r in rows], V, [r[3] for r in rows], [r[1] for r in rows]

    def vector(self, vid):
        import numpy as np
        with self.lock:
            r = self.db.execute("SELECT vec FROM event_vectors WHERE id=?", (vid,)).fetchone()
        return None if r is None else np.frombuffer(r[0], np.float16).astype(np.float32)

    # ------------------------------------------------------------------ concept neurons (made from words in milliseconds)
    def add_concept(self, name, description, data, severity, by, ts=None, stats=None):
        ts = ts or time.time()
        with self.lock:
            cur = self.db.execute("INSERT INTO concepts(name,description,state,data,severity,created,decided_by,decided,stats) VALUES(?,?,?,?,?,?,?,?,?)",
                                  (name, description, "active", json.dumps(data), severity, ts, by, ts, json.dumps(stats or {})))
            return cur.lastrowid

    def concepts(self, state=None):
        q = "SELECT id,name,description,state,data,severity,created,decided_by,decided,stats FROM concepts"
        with self.lock:
            rows = self.db.execute(q + (" WHERE state=?" if state else "") + " ORDER BY id", ((state,) if state else ())).fetchall()
        return [{"id": r[0], "name": r[1], "description": r[2], "state": r[3], "data": json.loads(r[4]), "severity": r[5], "created": r[6],
                 "decided_by": r[7], "decided": r[8], "stats": json.loads(r[9] or "{}")} for r in rows]

    def update_concept(self, cid, data):
        with self.lock:
            self.db.execute("UPDATE concepts SET data=? WHERE id=?", (json.dumps(data), cid))

    def set_concept_state(self, cid, state, by, ts=None):
        with self.lock:
            self.db.execute("UPDATE concepts SET state=?, decided_by=?, decided=? WHERE id=?", (state, by, ts or time.time(), cid))

    def concept_stamp(self):
        with self.lock:
            return tuple(self.db.execute("SELECT COUNT(*), COALESCE(MAX(decided),0), COALESCE(SUM(CASE WHEN state='active' THEN id END),0) FROM concepts").fetchone())

    # ------------------------------------------------------------------ named blobs (the growing memory)
    def put_blob(self, key, blob, ts=None):
        with self.lock:
            self.db.execute("INSERT INTO blobs(key,ts,blob) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET ts=excluded.ts, blob=excluded.blob", (key, ts or time.time(), blob))

    def get_blob(self, key):
        with self.lock:
            r = self.db.execute("SELECT blob FROM blobs WHERE key=?", (key,)).fetchone()
        return None if r is None else r[0]

    # ------------------------------------------------------------------ housekeeping inside our own database only
    def prune(self, now, days):
        cut = now - days * 86400
        with self.tx() as db:
            db.execute("DELETE FROM windows WHERE ts<? AND flagged=0", (cut,))
            db.execute("DELETE FROM alerts WHERE ts<? AND status!='open'", (cut,))
            db.execute("DELETE FROM events WHERE ts<?", (cut,))
            db.execute("DELETE FROM event_vectors WHERE id <= (SELECT COALESCE(MAX(id),0) FROM event_vectors) - 50000")
