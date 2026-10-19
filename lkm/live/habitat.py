"""The habitat: the instance in which the organism lives.

Life cycle
    boot      take the single-instance lock, open memory, load models, restore the body if one exists
    cycle     one waking period (default once a day): hear back from the world, sense the world as it is NOW, let time pass,
              learn from answers, evolve, choose what to check next, write the decisions and the body to memory atomically
    sleep     wait for the next cycle (or for an answer to arrive; answers are stored immediately and used in the next cycle)
    shutdown  on SIGTERM/SIGINT finish the current cycle, save, release the lock
    crash     nothing is lost: body and queue are committed together once per cycle

It only ever *proposes* checks (queue, outbox file, optional webhook). It never changes the data it watches.
"""
import csv, fcntl, glob, json, os, pickle, shutil, signal, threading, time
import numpy as np
from .store import Store, pack, unpack_blob
from .body import LiveOrganism
from .. import nca as N

DEFAULTS = dict(
    budget_per_cycle=None, budget_frac_per_cycle=0.00375,       # 3 percent over 8 cycles, as in the study
    interval_seconds=86400.0, label_ttl_days=30.0, pending_ttl_days=7.0, max_pending_factor=3,
    rounds=8, workers_per_slot=4, evolve=True, adapt_l2=40.0, dry_run=False, seed=0,
    host="127.0.0.1", port=8765, token_env="ORGANISM_TOKEN", webhook_url=None,
    source={"type": "csv", "path": "world.csv"}, verifier=None)
DAY = 86400.0


class LiveEnv:
    """What the organism perceives of the world in this cycle (same surface as the simulated game's Env)."""

    def __init__(self, ds, ids, rounds, budget, round_no, seed):
        self.ds, self.G, self.n = ds, ds.graph(), ds.n
        self.ids = ids
        self.rounds, self.round = rounds, round_no
        self.B = np.full(rounds, budget, int)
        self.rng = np.random.default_rng(seed)
        self.verified = np.zeros(self.n, np.int8)
        self.blocked = np.zeros(self.n, bool)

    @property
    def rev_pos(self):
        return (self.verified == 2).astype(np.float32)

    @property
    def rev_neg(self):
        return (self.verified == 1).astype(np.float32)


class VirtualClock:
    """For demonstrations and tests: one cycle = one simulated day, without waiting."""

    def __init__(self, start=1.8e9):
        self.t = start

    def now(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class Habitat:
    def __init__(self, home, source, cfg=None, clock=time.time, sleep=time.sleep, verifier=None, log=print):
        self.home = os.path.abspath(home)
        os.makedirs(self.home, exist_ok=True)
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.source, self.clock, self.sleep, self.verifier, self.log = source, clock, sleep, verifier, log
        self.store = None
        self.org = None
        self.models = None
        self.stop = threading.Event()
        self.paused = False
        self._lockf = None
        self._ver = None
        self._world = None
        self.cycle_no = 0
        self.last = {}
        self.boot_ts = None

    # ------------------------------------------------------------------ waking up
    def boot(self):
        self._lockf = open(os.path.join(self.home, "lock"), "w")
        try:
            fcntl.flock(self._lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lockf.close()
            raise RuntimeError(f"another organism is already living in {self.home}")
        self._lockf.write(str(os.getpid())); self._lockf.flush()
        self.store = Store(self.home)
        with open(os.path.join(self.home, "models", "models.pkl"), "rb") as f:
            self.models = pickle.load(f)                           # our own file, written by bootstrap
        self.boot_ts = self.clock()
        born = self.store.meta_get("born_ts")
        if born is None:
            self.store.meta_set("born_ts", self.boot_ts)
        self.store.meta_set("boots", (self.store.meta_get("boots", 0) or 0) + 1)
        snap = self.store.load_snapshot()
        self.cycle_no = int(self.store.meta_get("cycle_no", 0))
        self._restore_blob = unpack_blob(snap["blob"]) if snap else None
        self.paused = os.path.exists(os.path.join(self.home, "PAUSE"))
        self.store.event("boot", {"restored": snap is not None, "cycle": self.cycle_no, "pid": os.getpid()}, ts=self.boot_ts)
        self.log(f"[habitat] awake in {self.home}; {'restored a body at cycle ' + str(self.cycle_no) if snap else 'no body yet: it will be born at the first cycle'}")

    def shutdown(self):
        if self.store:
            self.store.event("shutdown", {"cycle": self.cycle_no}, ts=self.clock())
            self.store.close()
        if self._lockf:
            fcntl.flock(self._lockf, fcntl.LOCK_UN); self._lockf.close()

    # ------------------------------------------------------------------ senses
    def _sense(self):
        v = self.source.version()
        if self._world is None or v != self._ver:
            ds, ids = self.source.snapshot()
            base = np.asarray(self.models.graph.score(ds), np.float32)
            self._world = (ds, np.asarray(ids), base)
            self._ver = v
            self.store.event("sensed", {"version": v, "entities": int(ds.n)}, ts=self.clock())
        return self._world

    def budget(self, n):
        b = self.cfg["budget_per_cycle"]
        return int(b) if b else max(1, int(round(self.cfg["budget_frac_per_cycle"] * n)))

    # ------------------------------------------------------------------ hearing back
    def answer(self, items, source="api"):
        """items: iterable of (entity_id, label). Returns per-item status."""
        res = []
        for e, lab in items:
            res.append(self.store.answer(str(e), int(lab), source, ts=self.clock()))
        return res

    def _ingest_inbox(self):
        inbox = os.path.join(self.home, "inbox")
        os.makedirs(os.path.join(inbox, "done"), exist_ok=True)
        n = 0
        for path in sorted(glob.glob(os.path.join(inbox, "*.csv"))):
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    self.store.answer(str(row["entity_id"]), int(row["label"]), "inbox", ts=self.clock()); n += 1
            shutil.move(path, os.path.join(inbox, "done", os.path.basename(path) + f".{int(self.clock())}"))
        return n

    # ------------------------------------------------------------------ one waking period
    def cycle(self):
        cfg, st, now = self.cfg, self.store, self.clock()
        if self.paused or os.path.exists(os.path.join(self.home, "PAUSE")):
            self.paused = True
            st.event("paused", {}, ts=now)
            return {"paused": True}
        t0 = time.time()
        # 1. the outside world answers
        if self.verifier is not None:
            for e, lab, src in self.verifier.poll(self, now):
                st.answer(e, lab, src, ts=now)
        inbox_n = self._ingest_inbox()
        # 2. sense the world as it is now
        ds, ids, base = self._sense()
        index = {e: i for i, e in enumerate(ids.tolist())}
        B = self.budget(ds.n)
        env = LiveEnv(ds, ids, cfg["rounds"], B, self.cycle_no % cfg["rounds"], cfg["seed"] + self.cycle_no)
        # 3. body: restore, or be born
        M = self.models
        if self.org is None:
            self.org = LiveOrganism(M.thetas["full"], M.P, vocab=getattr(M, "vocab", 0), seed=cfg["seed"], workers_per_slot=cfg["workers_per_slot"],
                                    adapt_l2=cfg["adapt_l2"], evolve=cfg["evolve"])
            self.org.theta = np.asarray(M.thetas["full"], np.float64)
            if self._restore_blob is not None:
                self.org.restore(self._restore_blob)
                self.org.attach(env, ids, base)
                self._restore_blob = None
            else:
                self.org.birth(env, ids, base)
                st.event("birth", {"workers": int(self.org.W)}, ts=now)
        else:
            self.org.attach(env, ids, base)
        org = self.org
        # 4. time passes; answers are taken in
        org.tick_world()
        answered = st.answered_unintegrated()
        items = []
        for a in answered:
            ctx = json.loads(a["ctx"]) if a["ctx"] else {}
            items.append({"worker": a["worker"], "uid": a["uid"], "entity_idx": index.get(a["entity_id"]), "label": a["label"],
                          "h_final": ctx.get("h_final"), "x": ctx.get("x")})
        credited = org.credit(items)
        # 5. what is known and what is pending
        ttl = cfg["label_ttl_days"] * DAY
        for e, (lab, ts) in st.labels_since(now - ttl).items():
            i = index.get(e)
            if i is not None:
                env.verified[i] = 1 + int(lab)
        expired_cut = now - cfg["pending_ttl_days"] * DAY
        pend = st.pending()
        expired = [p["qid"] for p in pend if p["created"] < expired_cut]
        live_pending = [p for p in pend if p["created"] >= expired_cut]
        for p in live_pending:
            i = index.get(p["entity_id"])
            if i is not None:
                env.blocked[i] = True
        # 6. learn and evolve
        org.refresh_cells(env)
        born = org.evolve_live()
        # 7. choose what to check next
        allowance = max(0, cfg["max_pending_factor"] * B - len(live_pending))
        k = min(B, allowance)
        picks = org.choose(env, k) if k > 0 else []
        # 8. explain the decisions
        from ..narrate import Narrator
        nar = Narrator(ds, base, M.P)
        known = env.verified > 0
        nar.set_labels((env.verified == 2).astype(float), known)
        rows = []
        for p in picks:
            i = p["entity_idx"]
            p["entity_id"] = ids[i]
            p["why"] = nar.explain(i, risk=p["risk"], env=env)
            rows.append(p)
        # 9. commit: queue rows, integrations, expiries and the body in ONE transaction
        sayings = [{"entity_id": r["entity_id"], "risk": r["risk"], "why": r["why"]} for r in rows]
        stats = {"cycle": self.cycle_no, "entities": int(ds.n), "budget": B, "picked": len(rows), "answers_integrated": len(answered),
                 "credited": credited, "expired": len(expired), "births": born, "mean_energy": float(org.energy.mean()),
                 "pending": len(live_pending) + len(rows), "inbox": inbox_n, "seconds": round(time.time() - t0, 2),
                 "data_version": self._ver, "words_used": int(len(np.unique(N.words(M.P, org._cache)))) if (org._cache is not None and "Wt" in M.P) else 0,
                 "workers": int(org.W), "total_births": int(org.generation_births)}
        with st.tx() as db:
            for r in rows:
                db.execute("INSERT INTO queue(entity_id,cycle,created,status,worker,uid,risk,why,ctx,data_version) VALUES(?,?,?,?,?,?,?,?,?,?)",
                           (r["entity_id"], self.cycle_no, now, "pending", r["worker"], r["uid"], r["risk"], r["why"], r["ctx"], self._ver))
            if answered:
                db.executemany("UPDATE queue SET status='integrated' WHERE qid=?", [(a["qid"],) for a in answered])
            if expired:
                db.executemany("UPDATE queue SET status='expired' WHERE qid=?", [(q,) for q in expired])
            st.save_snapshot(self.cycle_no, now, pack(org.export()))
            db.execute("INSERT INTO meta(key,value) VALUES('cycle_no',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (json.dumps(self.cycle_no + 1),))
            db.execute("INSERT INTO meta(key,value) VALUES('last_cycle_ts',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (json.dumps(now),))
            db.execute("INSERT INTO meta(key,value) VALUES('last_stats',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (json.dumps(stats),))
            db.execute("INSERT INTO events(ts,kind,data) VALUES(?,?,?)", (now, "cycle", json.dumps({**stats, "sayings": sayings[:20]})))
        self.cycle_no += 1
        # 10. tell the world (best effort; the queue itself is the source of truth)
        if rows and not cfg["dry_run"]:
            self._emit(rows, now)
        self.last = stats
        self.log(f"[habitat] cycle {stats['cycle']}: picked {stats['picked']} (budget {B}), integrated {stats['answers_integrated']} answers, "
                 f"{stats['births']} births, pending {stats['pending']}, {stats['seconds']}s")
        return stats

    def _emit(self, rows, now):
        out = [{"entity_id": r["entity_id"], "risk": round(r["risk"], 4), "why": r["why"], "created": now, "data_version": self._ver} for r in rows]
        with open(os.path.join(self.home, "outbox.jsonl"), "a") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
        url = self.cfg.get("webhook_url")
        if url:
            try:
                import urllib.request
                req = urllib.request.Request(url, data=json.dumps({"checks": out}).encode(), headers={"Content-Type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=10).read()
            except Exception as e:
                self.store.event("webhook_failed", {"error": str(e)[:200]}, ts=now)

    # ------------------------------------------------------------------ living
    def run_forever(self, max_cycles=None, serve=True):
        for s in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(s, lambda *_: self.stop.set())
            except ValueError:
                pass                                           # not in the main thread
        if serve and self.cfg.get("port"):
            from .serve import start_server
            self.httpd = start_server(self)
        done = 0
        try:
            while not self.stop.is_set():
                self.cycle()
                done += 1
                if max_cycles is not None and done >= max_cycles:
                    break
                waited, step = 0.0, min(1.0, self.cfg["interval_seconds"])
                while waited < self.cfg["interval_seconds"] and not self.stop.is_set():
                    self.sleep(step); waited += step
        finally:
            if getattr(self, "httpd", None):
                self.httpd.shutdown(); self.httpd.server_close()
            self.shutdown()

    # ------------------------------------------------------------------ introspection
    def status(self):
        st = self.store
        counts = st.counts()
        rows = st.all_answered()
        hits = sum(1 for r in rows if r[1] == 1)
        o = self.org
        ls = st.meta_get("last_stats") or {}
        return {
            "alive": not self.stop.is_set(), "paused": self.paused, "home": self.home,
            "born": st.meta_get("born_ts"), "awake_since": self.boot_ts, "boots": st.meta_get("boots"), "cycle": self.cycle_no,
            "last_cycle_ts": st.meta_get("last_cycle_ts"), "queue": counts,
            "picks_answered": len(rows), "picks_that_were_real_problems": hits,
            "precision_of_picks": (hits / len(rows)) if rows else None,
            "workers": (int(o.W) if o else ls.get("workers")), "mean_energy": (float(o.energy.mean()) if o else ls.get("mean_energy")),
            "total_births": (int(o.generation_births) if o else ls.get("total_births")), "last_cycle": self.last or ls,
            "vocabulary": getattr(self.models, "vocab", 0), "simulated_answers": bool(getattr(self.verifier, "simulated", False)),
        }

    def say(self):
        """The organism describes itself and its latest decisions in plain sentences."""
        s = self.status()
        lines = []
        days = None if not s["born"] else (self.clock() - s["born"]) / DAY
        lines.append(f"I have been alive for {days:.1f} days, woken {s['boots']} time(s), and have lived {s['cycle']} cycles." if days is not None else "I am not born yet.")
        if s["picks_answered"]:
            lines.append(f"I proposed {s['picks_answered']} checks that came back; {s['picks_that_were_real_problems']} ({100*s['precision_of_picks']:.0f}%) were real problems.")
        if s["workers"]:
            lines.append(f"{s['workers']} workers live in me; {s['total_births']} have been born since I woke the first time; their mean energy is {s['mean_energy']:.2f}.")
        ev = self.store.events(1, "cycle")
        for sy in (ev[0]["data"].get("sayings", [])[:5] if ev else []):
            lines.append(sy["why"])
        return lines

    def grade(self, truth):
        """Prospective grading: score every check the organism proposed against what really happened (dict id -> 0/1)."""
        rows = self.store.db.execute("SELECT entity_id,risk,data_version,created FROM queue WHERE worker IS NOT NULL").fetchall()
        picks = [(e, r) for e, r, _, _ in rows if e in truth]
        if not picks:
            return {"graded": 0}
        y = np.array([truth[e] for e, _ in picks])
        base_rate = float(np.mean(list(truth.values())))
        return {"graded": int(len(y)), "hit_rate": float(y.mean()), "base_rate": base_rate, "lift": float(y.mean() / max(base_rate, 1e-9)),
                "mean_predicted_risk": float(np.mean([r for _, r in picks]))}
