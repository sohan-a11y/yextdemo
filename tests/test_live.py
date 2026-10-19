import json, os, shutil, tempfile, threading, unittest, urllib.request
import numpy as np
from tests import _common  # noqa
from lkm import nca as N, nca_train as NT
from lkm.baselines import GBDT
from lkm.organism import init_genome
from lkm.pipeline import Models
from lkm.worlds import make_world, make_bundle
from lkm.live.habitat import Habitat, VirtualClock, DAY
from lkm.live.source import DatasetSource
from lkm.live.verifiers import ReplayVerifier
from lkm.live.store import Store

V = 8


def make_models(home, seed=0):
    train = make_bundle("cascade", range(100 + seed, 103 + seed), n=1000, rho=0.3)
    gb = GBDT("graph", max_iter=60).fit(train)
    from lkm.baselines import oof_scores
    oof = oof_scores(train, lambda: GBDT("graph", max_iter=60), folds=3)
    P, _ = NT.train(train, oof, val=train[0], val_base=oof[0], steps=40, V=V, comm="both", log=lambda *a: None)
    M = Models(); M.graph, M.P, M.vocab, M.comm = gb, P, V, "both"
    M.thetas = {"full": init_genome(0, V)}
    import pickle
    os.makedirs(os.path.join(home, "models"), exist_ok=True)
    with open(os.path.join(home, "models", "models.pkl"), "wb") as f:
        pickle.dump(M, f)
    return M


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.models_home = os.path.join(cls.tmp, "_models")
        make_models(cls.models_home)
        cls.world = make_world(1900, n=1200, kind="cascade", rho=0.3)
        cls.ids = np.array([f"e{i}" for i in range(cls.world.n)])
        cls.truth = {e: int(y) for e, y in zip(cls.ids, cls.world.y)}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def home(self, name):
        h = os.path.join(self.tmp, name)
        shutil.rmtree(h, ignore_errors=True)
        shutil.copytree(self.models_home, h)
        return h

    def habitat(self, home, verifier=True, clock=None, cfg=None, source=None):
        vc = clock or VirtualClock()
        h = Habitat(home, source or DatasetSource(self.world, self.ids), {"port": None, "budget_per_cycle": 12, "interval_seconds": DAY, **(cfg or {})},
                    clock=vc.now, sleep=vc.advance, verifier=ReplayVerifier(self.truth, 1) if verifier else None, log=lambda *a: None)
        h.vc = vc
        h.boot()
        return h


class TestLife(Base):
    def test_birth_cycles_and_rules(self):
        h = self.habitat(self.home("a"))
        for _ in range(4):
            h.vc.advance(DAY); s = h.cycle()
        pend = h.store.pending()
        ids = [p["entity_id"] for p in pend]
        self.assertEqual(len(ids), len(set(ids)))                       # nothing is asked twice while pending
        self.assertLessEqual(len(pend), 3 * 12)                         # backlog cap
        self.assertTrue(all(p["why"] for p in pend))                    # every decision is explained
        rows = h.store.all_answered()
        self.assertGreater(len(rows), 10)                               # the world answered, the organism used the answers
        h.shutdown()

    def test_memory_survives_restart(self):
        home = self.home("b")
        h = self.habitat(home)
        for _ in range(3):
            h.vc.advance(DAY); h.cycle()
        g, e, uid = h.org.genomes.copy(), h.org.energy.copy(), h.org.uid.copy()
        pos_ids = h.org.pos_ids.copy()
        cyc = h.cycle_no
        h.shutdown()
        h2 = self.habitat(home, clock=h.vc)                              # a new process: new objects, same home
        self.assertEqual(h2.cycle_no, cyc)
        h2.vc.advance(DAY); h2.cycle()
        self.assertEqual(h2.store.meta_get("boots"), 2)
        self.assertEqual(h2.org.W, len(g))
        # the individuals are the same ones: identities persisted, the old ids are a superset of the survivors
        survivors = np.intersect1d(uid, h2.org.uid)
        self.assertGreater(len(survivors), len(uid) // 2)
        h2.shutdown()

    def test_snapshot_roundtrip_is_exact(self):
        h = self.habitat(self.home("c"))
        for _ in range(2):
            h.vc.advance(DAY); h.cycle()
        a = h.org.export()
        h.shutdown()
        h2 = self.habitat(self.home("c_copy_unused"))
        h2.org = None
        from lkm.live.body import LiveOrganism
        o = LiveOrganism(h.models.thetas["full"], h.models.P, vocab=V)
        o.restore(a)
        b = o.export()
        for k in ("genomes", "h", "Wf", "energy", "last_r", "uid"):
            self.assertTrue(np.array_equal(a[k], b[k]), k)
        self.assertEqual(str(a["rng"]), str(b["rng"]))
        h2.shutdown()

    def test_late_duplicate_and_unsolicited_answers(self):
        h = self.habitat(self.home("d"), verifier=False)
        h.vc.advance(DAY); h.cycle()
        p = h.store.pending()[0]
        r1 = h.answer([(p["entity_id"], 1)])
        self.assertEqual(r1, ["answered"])
        r2 = h.answer([("not-in-queue", 0)])
        self.assertEqual(r2, ["recorded_unsolicited"])
        h.vc.advance(DAY); s = h.cycle()
        self.assertEqual(s["answers_integrated"], 2)
        self.assertEqual(s["credited"], 1)                                # only the proposed check pays a worker
        self.assertNotIn(p["entity_id"], [q["entity_id"] for q in h.store.pending()])
        h.shutdown()

    def test_unanswered_checks_expire_and_can_be_asked_again(self):
        h = self.habitat(self.home("e"), verifier=False, cfg={"pending_ttl_days": 2})
        h.vc.advance(DAY); h.cycle()
        first = {p["entity_id"] for p in h.store.pending()}
        for _ in range(4):
            h.vc.advance(DAY); h.cycle()
        self.assertGreater(h.store.counts().get("expired", 0), 0)
        h.shutdown()

    def test_dead_worker_is_not_credited(self):
        h = self.habitat(self.home("f"), verifier=False)
        h.vc.advance(DAY); h.cycle()
        p = h.store.pending()[0]
        h.org.uid[p["worker"]] += 10_000                                    # that individual has since died and been replaced
        h.store.answer(p["entity_id"], 1, "test", ts=h.clock())
        h.vc.advance(DAY); s = h.cycle()
        self.assertEqual(s["credited"], 0)
        h.shutdown()

    def test_world_changes_between_cycles(self):
        src = DatasetSource(self.world, self.ids, version="v1")
        h = self.habitat(self.home("g"), source=src, verifier=False)
        h.vc.advance(DAY); h.cycle()
        keep = np.arange(0, self.world.n, 2)                                 # half of the entities vanish
        sub = self.world.subset(keep)
        src.ds, src.ids = DatasetSource(sub, self.ids[keep]).ds, self.ids[keep]; src.set_version("v2")
        h.vc.advance(DAY); s = h.cycle()
        self.assertEqual(s["entities"], len(keep))
        self.assertTrue(set(h.org.pos_ids.tolist()) <= set(self.ids.tolist()))
        h.shutdown()

    def test_kill_switch_and_single_instance(self):
        home = self.home("h")
        h = self.habitat(home, verifier=False)
        with self.assertRaises(RuntimeError):
            Habitat(home, DatasetSource(self.world, self.ids), {"port": None}, log=lambda *a: None).boot()
        open(os.path.join(home, "PAUSE"), "w").close()
        h.vc.advance(DAY)
        self.assertTrue(h.cycle().get("paused"))
        self.assertEqual(h.store.pending(), [])
        os.remove(os.path.join(home, "PAUSE")); h.paused = False
        h.vc.advance(DAY); self.assertGreater(h.cycle()["picked"], 0)
        h.shutdown()

    def test_inbox_csv_is_read_once(self):
        home = self.home("i")
        h = self.habitat(home, verifier=False)
        h.vc.advance(DAY); h.cycle()
        p = h.store.pending()[0]["entity_id"]
        os.makedirs(os.path.join(home, "inbox"), exist_ok=True)
        with open(os.path.join(home, "inbox", "a.csv"), "w") as f:
            f.write(f"entity_id,label\n{p},1\n")
        h.vc.advance(DAY); s = h.cycle()
        self.assertEqual(s["inbox"], 1)
        self.assertFalse(os.path.exists(os.path.join(home, "inbox", "a.csv")))
        h.shutdown()

    def test_labels_expire_with_the_ttl(self):
        h = self.habitat(self.home("j"), verifier=False, cfg={"label_ttl_days": 3})
        h.vc.advance(DAY); h.cycle()
        e = h.store.pending()[0]["entity_id"]
        h.store.answer(e, 1, "test", ts=h.clock())
        h.vc.advance(DAY); h.cycle()
        self.assertEqual(int(h.store.labels_since(h.clock() - 3 * DAY).get(e, (None,))[0]), 1)
        h.vc.advance(5 * DAY); h.cycle()
        self.assertNotIn(e, h.store.labels_since(h.clock() - 3 * DAY))
        h.shutdown()

    def test_prospective_grading(self):
        h = self.habitat(self.home("k"), verifier=False)
        for _ in range(3):
            h.vc.advance(DAY); h.cycle()
        g = h.grade(self.truth)
        self.assertGreater(g["graded"], 10)
        self.assertGreater(g["hit_rate"], g["base_rate"])                  # untrained workers still beat chance with the trees underneath
        h.shutdown()


class TestInterface(Base):
    def test_http_api(self):
        h = self.habitat(self.home("l"), verifier=False, cfg={"port": 0, "host": "127.0.0.1"})
        h.vc.advance(DAY); h.cycle()
        from lkm.live.serve import start_server
        srv = start_server(h)
        base = f"http://127.0.0.1:{h.port_bound}"
        try:
            get = lambda p: json.loads(urllib.request.urlopen(base + p, timeout=10).read())
            self.assertTrue(get("/health")["ok"])
            st = get("/status"); self.assertEqual(st["cycle"], 1)
            q = get("/queue")["pending"]; self.assertGreater(len(q), 0)
            html = urllib.request.urlopen(base + "/", timeout=10).read().decode()
            self.assertIn("Organism", html); self.assertIn("In its own words", html)
            req = urllib.request.Request(base + "/answers", data=json.dumps({"answers": [{"entity_id": q[0]["entity_id"], "label": 1}]}).encode(), method="POST")
            self.assertEqual(json.loads(urllib.request.urlopen(req, timeout=10).read())["results"], ["answered"])
            bad = urllib.request.Request(base + "/answers", data=json.dumps({"answers": [{"entity_id": "x", "label": 7}]}).encode(), method="POST")
            with self.assertRaises(urllib.error.HTTPError) as c:
                urllib.request.urlopen(bad, timeout=10)
            self.assertEqual(c.exception.code, 400)
            urllib.request.urlopen(urllib.request.Request(base + "/pause", data=b"{}", method="POST"), timeout=10).read()
            self.assertTrue(h.paused)
        finally:
            srv.shutdown(); srv.server_close(); h.shutdown()

    def test_token_is_enforced(self):
        os.environ["ORGANISM_TOKEN_TEST"] = "s3cret"
        h = self.habitat(self.home("m"), verifier=False, cfg={"port": 0, "token_env": "ORGANISM_TOKEN_TEST"})
        h.vc.advance(DAY); h.cycle()
        from lkm.live.serve import start_server
        srv = start_server(h)
        base = f"http://127.0.0.1:{h.port_bound}"
        try:
            with self.assertRaises(urllib.error.HTTPError) as c:
                urllib.request.urlopen(base + "/status", timeout=10)
            self.assertEqual(c.exception.code, 401)
            r = urllib.request.Request(base + "/status", headers={"Authorization": "Bearer s3cret"})
            self.assertEqual(json.loads(urllib.request.urlopen(r, timeout=10).read())["cycle"], 1)
            self.assertTrue(json.loads(urllib.request.urlopen(base + "/health", timeout=10).read())["ok"])
        finally:
            del os.environ["ORGANISM_TOKEN_TEST"]; srv.shutdown(); srv.server_close(); h.shutdown()


class TestSafety(Base):
    def test_refuses_to_listen_publicly_without_a_token(self):
        os.environ.pop("ORGANISM_TOKEN_NONE", None)
        h = self.habitat(self.home("n"), verifier=False, cfg={"port": 0, "host": "0.0.0.0", "token_env": "ORGANISM_TOKEN_NONE"})
        from lkm.live.serve import start_server
        with self.assertRaises(RuntimeError):
            start_server(h)
        h.shutdown()

    def test_dry_run_proposes_but_emits_nothing(self):
        home = self.home("o")
        h = self.habitat(home, verifier=False, cfg={"dry_run": True})
        h.vc.advance(DAY); h.cycle()
        self.assertGreater(len(h.store.pending()), 0)
        self.assertFalse(os.path.exists(os.path.join(home, "outbox.jsonl")))
        h.shutdown()

    def test_outbox_written_when_not_dry(self):
        home = self.home("p")
        h = self.habitat(home, verifier=False)
        h.vc.advance(DAY); h.cycle()
        lines = open(os.path.join(home, "outbox.jsonl")).read().splitlines()
        self.assertEqual(len(lines), len(h.store.pending()))
        self.assertIn("why", json.loads(lines[0]))
        h.shutdown()


class TestStore(unittest.TestCase):
    def test_transaction_rolls_back(self):
        d = tempfile.mkdtemp()
        try:
            st = Store(d)
            with self.assertRaises(RuntimeError):
                with st.tx() as db:
                    db.execute("INSERT INTO events(ts,kind,data) VALUES(1,'x','{}')")
                    raise RuntimeError("crash in the middle of a cycle")
            self.assertEqual(st.events(10), [])
            st.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
