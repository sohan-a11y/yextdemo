"""Durable memory of the organism: one SQLite file (WAL) holding its work queue, answers, events and the latest body snapshots.

The body snapshot and the queue rows of a cycle are written in ONE transaction, so after a crash (even kill -9) the
organism wakes up in a state that is consistent with its queue.
"""
import io, json, os, sqlite3, threading, time
from contextlib import contextmanager
import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS queue (
  qid INTEGER PRIMARY KEY AUTOINCREMENT, entity_id TEXT NOT NULL, cycle INTEGER, created REAL, status TEXT NOT NULL,
  label INTEGER, answered REAL, source TEXT, worker INTEGER, uid INTEGER, risk REAL, why TEXT, ctx TEXT, data_version TEXT);
CREATE INDEX IF NOT EXISTS q_status ON queue(status);
CREATE INDEX IF NOT EXISTS q_entity ON queue(entity_id);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS snapshots (cycle INTEGER PRIMARY KEY, ts REAL, blob BLOB);
"""


def pack(arrays: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def unpack_blob(blob: bytes) -> dict:
    z = np.load(io.BytesIO(blob), allow_pickle=False)
    return {k: z[k] for k in z.files}


class Store:
    def __init__(self, home):
        os.makedirs(home, exist_ok=True)
        self.path = os.path.join(home, "organism.db")
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA)

    @contextmanager
    def tx(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------ meta / events
    def meta_get(self, key, default=None):
        with self.lock:
            r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if r is None else json.loads(r[0])

    def meta_set(self, key, value):
        with self.lock:
            self.db.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                            (key, json.dumps(value)))

    def event(self, kind, data=None, ts=None):
        with self.lock:
            self.db.execute("INSERT INTO events(ts,kind,data) VALUES(?,?,?)", (ts or time.time(), kind, json.dumps(data or {}, default=_j)))

    def events(self, n=50, kind=None):
        q = "SELECT ts,kind,data FROM events" + (" WHERE kind=?" if kind else "") + " ORDER BY id DESC LIMIT ?"
        with self.lock:
            rows = self.db.execute(q, ((kind, n) if kind else (n,))).fetchall()
        return [{"ts": r[0], "kind": r[1], "data": json.loads(r[2])} for r in rows]

    # ------------------------------------------------------------------ queue
    def pending(self, limit=None):
        with self.lock:
            q = "SELECT qid,entity_id,cycle,created,risk,why,worker,uid FROM queue WHERE status='pending' ORDER BY qid"
            rows = self.db.execute(q + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
        return [dict(zip(("qid", "entity_id", "cycle", "created", "risk", "why", "worker", "uid"), r)) for r in rows]

    def pending_ids(self):
        with self.lock:
            return {r[0] for r in self.db.execute("SELECT entity_id FROM queue WHERE status='pending'")}

    def answer(self, entity_id, label, source="api", ts=None):
        """Record an answer. Idempotent: a second answer for an already answered item is ignored and reported."""
        ts = ts or time.time()
        with self.tx() as db:
            r = db.execute("SELECT qid FROM queue WHERE entity_id=? AND status='pending' ORDER BY qid LIMIT 1", (entity_id,)).fetchone()
            if r is not None:
                db.execute("UPDATE queue SET status='answered', label=?, answered=?, source=? WHERE qid=?", (int(label), ts, source, r[0]))
                return "answered"
            dup = db.execute("SELECT qid FROM queue WHERE entity_id=? AND answered=? AND label=?", (entity_id, ts, int(label))).fetchone()
            if dup is not None:
                return "duplicate"
            db.execute("INSERT INTO queue(entity_id,cycle,created,status,label,answered,source) VALUES(?,?,?,?,?,?,?)",
                       (entity_id, None, ts, "answered", int(label), ts, source))
            return "recorded_unsolicited"

    def answered_unintegrated(self):
        with self.lock:
            rows = self.db.execute("SELECT qid,entity_id,label,answered,worker,uid,ctx FROM queue WHERE status='answered' ORDER BY qid").fetchall()
        return [dict(zip(("qid", "entity_id", "label", "answered", "worker", "uid", "ctx"), r)) for r in rows]

    def labels_since(self, since_ts):
        """Latest known label per entity answered after `since_ts` (integrated or not)."""
        with self.lock:
            rows = self.db.execute("SELECT entity_id,label,answered FROM queue WHERE label IS NOT NULL AND answered>=? ORDER BY answered", (since_ts,)).fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}

    def counts(self):
        with self.lock:
            rows = self.db.execute("SELECT status,COUNT(*) FROM queue GROUP BY status").fetchall()
        return dict(rows)

    def all_answered(self):
        with self.lock:
            return self.db.execute("SELECT entity_id,label,created,answered,data_version,risk FROM queue WHERE label IS NOT NULL AND worker IS NOT NULL").fetchall()

    # ------------------------------------------------------------------ snapshots
    def save_snapshot(self, cycle, ts, blob, keep=3):
        self.db.execute("INSERT OR REPLACE INTO snapshots(cycle,ts,blob) VALUES(?,?,?)", (cycle, ts, blob))
        self.db.execute("DELETE FROM snapshots WHERE cycle <= ?", (cycle - keep,))

    def load_snapshot(self):
        with self.lock:
            r = self.db.execute("SELECT cycle,ts,blob FROM snapshots ORDER BY cycle DESC LIMIT 1").fetchone()
        return None if r is None else {"cycle": r[0], "ts": r[1], "blob": r[2]}

    def close(self):
        with self.lock:
            self.db.close()


def _j(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
