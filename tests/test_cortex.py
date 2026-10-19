"""Tests of the cortex: instant concept neurons, the self-growing novelty memory, and their place in the living Sentinel.

Most tests use a tiny stand-in encoder (hashed words) so they are fast and do not need the language model; the last class checks the real
MiniLM cortex and is skipped when the model is not installed.
"""
import contextlib
import io
import os
import re
import shutil
import tempfile
import time
import unittest

import numpy as np

from tests import _common  # noqa
from sentinel.cortex import ConceptNeuron, NoveltyMemory, event_sentence, make_neuron, unit, default_model_dir
from sentinel.service import Sentinel
from sentinel.sensors import Sensor
from sentinel.store import SentinelStore


class HashCortex:
    """Stand-in encoder: hashed word counts in 384 dimensions. Similar wording -> similar vector."""
    PARAMS = 0

    def encode(self, texts, batch=128):
        out = np.zeros((len(texts), 384), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, hash_(w) % 384] += 1.0
        return unit(out + 1e-6)


def hash_(w):
    h = 0
    for ch in w:
        h = (h * 131 + ord(ch)) % 1000003
    return h


class Fake(Sensor):
    name = "fake"

    def __init__(self):
        self.q = []

    def poll(self, now=None):
        q, self.q = self.q, []
        return q


def proc(cmd, pid=100, name="tool"):
    return {"kind": "process_new", "pid": pid, "ppid": 1, "name": name, "parent": "shell", "exe_class": "user", "exe": f"/home/u/{name}", "cmd": cmd,
            "script_host": False, "lolbin": False, "obfuscated": False, "office_parent": False, "novel_program": False}


NORMAL = ["git status", "git commit -m update", "ls -la documents", "python3 script.py", "code notes.md", "ping example.com", "npm install",
          "cat readme.txt", "grep error app.log", "ssh build server", "docker ps", "make test", "vim todo.txt", "cp report.pdf backup"]


class TestConceptNeuron(unittest.TestCase):
    def setUp(self):
        self.cx = HashCortex()
        rng = np.random.default_rng(0)
        self.normal = [f"{NORMAL[i % len(NORMAL)]} {rng.integers(1000)}" for i in range(3000)]
        self.V = self.cx.encode(self.normal)

    def test_born_from_words_and_calibrated(self):
        n = make_neuron(self.cx, "archives", "compress files into a zip archive or extract an archive", normal_vectors=self.V, alpha=0.01)
        fired = (n.score_many(self.V) > n.thr).mean()
        self.assertLessEqual(fired, 0.011)                                   # fires on at most alpha of the normal history
        x = self.cx.encode(["zip archive of all files"])[0]
        self.assertTrue(n.fires(x))

    def test_one_shot_feedback(self):
        n = make_neuron(self.cx, "archives", "compress files into a zip archive", normal_vectors=self.V, alpha=0.01)
        fp = self.cx.encode(["zip files list"])[0]                            # something harmless it fires on
        if not n.fires(fp):
            self.skipTest("stand-in encoder did not produce a false positive here")
        n.learn(fp, real=False)
        self.assertFalse(n.fires(fp))                                         # one answer is enough
        miss = self.cx.encode(["tar czf backup.tgz home"])[0]
        before = n.score(miss)
        n.learn(miss, real=True)
        self.assertGreater(n.score(miss), before)
        self.assertEqual(n.n_feedback, 2)

    def test_roundtrip_and_size(self):
        n = make_neuron(self.cx, "x", "watch for remote logins", ["ssh into server"], self.V, 0.01)
        n.learn(self.V[0], False)
        m = ConceptNeuron.from_dict(n.to_dict())
        np.testing.assert_allclose(n.score_many(self.V[:50]), m.score_many(self.V[:50]), atol=1e-6)
        self.assertEqual(m.thr, n.thr)
        self.assertEqual(n.n_params, 384 * 3 + 1)                             # description + 1 example + 1 counter-example

    def test_uncalibrated_neuron_never_fires(self):
        n = make_neuron(self.cx, "x", "anything at all")
        self.assertIsNone(n.thr)
        self.assertFalse(n.fires(self.V[0]))


class TestNoveltyMemory(unittest.TestCase):
    def test_grows_on_novelty_and_saturates_on_routine(self):
        cx = HashCortex()
        mem = NoveltyMemory(vigilance=0.8, capacity=500)
        routine = cx.encode(NORMAL * 50)
        born = [mem.observe(v) for v in routine]
        self.assertLessEqual(mem.born, len(NORMAL))                          # one prototype per routine activity, not per event
        self.assertTrue(all(not b for b in born[len(NORMAL) * 2:]))           # after two passes nothing new is born
        seen = mem.novelty(cx.encode(["git status"]))[0]
        new = mem.novelty(cx.encode(["format the disk partition now"]))[0]
        self.assertGreater(new, seen + 0.3)

    def test_capacity_is_respected_by_merging(self):
        rng = np.random.default_rng(0)
        mem = NoveltyMemory(vigilance=0.99, capacity=40)
        for _ in range(300):
            mem.observe(unit(rng.normal(size=384)))
        self.assertEqual(len(mem.P), 40)
        self.assertEqual(mem.born - mem.merged, 40)
        S = mem.P @ mem.P.T; np.fill_diagonal(S, -2)
        np.testing.assert_allclose(S.max(1), mem.nn_sim, atol=1e-5)          # the neighbour bookkeeping is exact
        self.assertEqual(int(mem.count.sum()), 300)

    def test_roundtrip(self):
        rng = np.random.default_rng(1)
        mem = NoveltyMemory(vigilance=0.5, capacity=30)
        for _ in range(200):
            mem.observe(unit(rng.normal(size=384)))
        m2 = NoveltyMemory.from_arrays(mem.to_arrays())
        X = unit(rng.normal(size=(20, 384)))
        np.testing.assert_allclose(mem.novelty(X), m2.novelty(X))
        m2.observe(X[0]); mem.observe(X[0])
        np.testing.assert_allclose(mem.nn_sim, m2.nn_sim, atol=1e-6)


class TestEventSentences(unittest.TestCase):
    def test_every_kind(self):
        self.assertIn("tar czf", event_sentence(proc("tar czf a.tgz b")))
        self.assertIn("the internet", event_sentence({"kind": "conn_new", "prog": "x", "exe_class": "temp", "remote_port": 443, "private": False}))
        self.assertIn("all network interfaces", event_sentence({"kind": "listen_new", "prog": "x", "exe_class": "temp", "port": 9, "public_bind": True}))
        self.assertEqual(event_sentence({"kind": "file", "op": "created", "loc": "user", "path": "/home/u/.ssh/id_rsa", "sensitive": True}), None)
        self.assertIn("report.txt", event_sentence({"kind": "file", "op": "created", "loc": "user", "path": "C:\\Users\\u\\report.txt"}))
        self.assertIn("(failed)", event_sentence({"kind": "dns", "name": "abc.example", "nx": True}))
        self.assertIsNone(event_sentence({"kind": "netstat", "bytes_out": 1}))


class TestCortexInTheLivingSentinel(unittest.TestCase):
    def boot(self, home, T, cfg=None):
        c = {"window_s": 30, "port": 0, "reflex": {"fm_windows": 3, "ad_windows": 3, "cal_windows": 3},
             "cortex": {"childhood_events": 30, "min_calibration": 40, "capacity": 200}}
        c.update(cfg or {})
        f = Fake()
        s = Sentinel(home, [f], c, clock=lambda: T[0], log=lambda *a: None, cortex=HashCortex()).boot()
        return s, f

    def live(self, s, f, T, windows, extra=None):
        out = []
        rng = np.random.default_rng(int(T[0]) % 1000)
        for w in range(windows):
            f.q = [proc(f"{NORMAL[int(rng.integers(len(NORMAL)))]} {int(rng.integers(100))}", pid=1000 + w * 10 + j) for j in range(6)]
            if extra and w == windows - 1:
                f.q += extra
            T[0] += 30
            r = s.step()
            out.append(r)
        return out

    def test_grow_a_neuron_while_alive_and_teach_it(self):
        d = tempfile.mkdtemp()
        T = [1.8e9]
        s, f = self.boot(d, T)
        self.live(s, f, T, 15)                                                 # 90 events read; vectors stored
        _, V, _, _ = s.store.recent_vectors()
        self.assertEqual(len(V), 90)
        n = make_neuron(HashCortex(), "archives", "compress files into a zip archive or extract an archive", normal_vectors=V, alpha=0.02)
        cid = s.store.add_concept("archives", n.description, n.to_dict(), "medium", "tester")
        r = self.live(s, f, T, 1, extra=[proc("zip archive compress all files", pid=4242, name="zip")])[-1]
        hits = [a for a in r["alerts"] if a["source"] == "concept:archives"]
        self.assertEqual(len(hits), 1)
        a = s.store.alert(hits[0]["id"])
        self.assertEqual(a["target"]["concept_id"], cid)
        self.assertIn("vec_id", a["target"])
        self.assertEqual(s.store.requests(), [])                               # a grown neuron at medium severity asks for nothing
        # 'it was fine' -> one-shot counter-example; the same thing does not fire again (beyond the cool-down)
        s.feedback(a["id"], "benign")
        data = s.store.concepts()[0]["data"]
        self.assertEqual((len(data["counter"]), data["n_feedback"]), (1, 1))
        T[0] += 3600
        r = self.live(s, f, T, 1, extra=[proc("zip archive compress all files", pid=4243, name="zip")])[-1]
        self.assertEqual([a for a in r["alerts"] if a["source"] == "concept:archives"], [])
        # switching it off stops it, and the record stays
        s.store.set_concept_state(cid, "disabled", "tester")
        T[0] += 3600
        r = self.live(s, f, T, 1, extra=[proc("extract the zip archive files", pid=4244, name="unzip")])[-1]
        self.assertEqual([a for a in r["alerts"] if a["source"].startswith("concept:")], [])
        self.assertEqual(s.store.concepts()[0]["state"], "disabled")
        s.shutdown(); shutil.rmtree(d)

    def test_uncalibrated_neuron_is_calibrated_automatically(self):
        d = tempfile.mkdtemp()
        T = [1.8e9]
        s, f = self.boot(d, T)
        n = make_neuron(HashCortex(), "remote", "log in to a remote server with ssh")
        s.store.add_concept("remote", n.description, n.to_dict(), "low", "tester")
        self.live(s, f, T, 12)                                                 # 72 vectors > min_calibration 40
        self.assertIsNotNone(s.store.concepts()[0]["data"]["thr"])
        s.shutdown(); shutil.rmtree(d)

    def test_memory_grows_counts_novelty_and_survives_restart(self):
        d = tempfile.mkdtemp()
        T = [1.8e9]
        s, f = self.boot(d, T)
        self.live(s, f, T, 12)
        bm = s.brain.brain_map()
        self.assertGreater(bm["memory_prototypes"], 0)
        self.assertLessEqual(bm["memory_prototypes"], len(NORMAL) * 2)
        self.assertEqual(bm["memory_childhood_left"], 0)
        r = self.live(s, f, T, 1, extra=[proc("mkfs format partition sdb1 wipe", pid=7, name="mkfs")])[-1]
        self.assertGreaterEqual(r["raw"]["n_novel_events"], 1)
        protos = len(s.brain.memory.P)
        s.shutdown()
        s2, f2 = self.boot(d, T)
        self.assertEqual(len(s2.brain.memory.P), protos)
        st = s2.status()
        self.assertIn("brain", st)
        self.assertEqual(st["brain"]["cortex"], "on")
        s2.shutdown(); shutil.rmtree(d)

    def test_without_the_model_the_rest_of_the_brain_still_works(self):
        d = tempfile.mkdtemp()
        T = [1.8e9]
        f = Fake()
        s = Sentinel(d, [f], {"window_s": 30, "port": 0, "cortex": {"model_dir": os.path.join(d, "nowhere")}}, clock=lambda: T[0], log=lambda *a: None).boot()
        self.assertTrue(s.status()["brain"]["cortex"].startswith("off"))
        f.q = [dict(proc("powershell -enc AAAA", name="powershell"), script_host=True, obfuscated=True)]
        T[0] += 30
        self.assertEqual(len(s.step()["alerts"]), 1)
        s.shutdown(); shutil.rmtree(d)


@unittest.skipUnless(os.path.exists(os.path.join(default_model_dir(), "model.onnx")), "language model not installed")
class TestRealCortex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sentinel.cortex import Cortex
        cls.cx = Cortex()

    def test_meaning_not_spelling(self):
        n = make_neuron(self.cx, "archives", "compress files into an archive or extract files from an archive")
        on = self.cx.encode(["tar czf backup.tar.gz documents", "7z x photos.7z", "unzip project.zip -d src"])
        off = self.cx.encode(["git status", "ping example.com", "docker ps -a"])
        self.assertGreater(n.score_many(on).min(), n.score_many(off).max())

    def test_creation_is_fast(self):
        V = self.cx.encode(NORMAL * 100)
        t0 = time.perf_counter()
        n = make_neuron(self.cx, "remote", "log in to a remote computer and run commands there", normal_vectors=V, alpha=0.01)
        dt = time.perf_counter() - t0
        self.assertLess(dt, 1.0)
        self.assertIsNotNone(n.thr)

    def test_cli_grow(self):
        d = tempfile.mkdtemp()
        st = SentinelStore(d)
        V = self.cx.encode(NORMAL * 80)
        st.add_event_vectors(1.8e9, [("process_new", s, v) for s, v in zip(NORMAL * 80, V)])
        from sentinel import __main__ as M
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            M.main(["grow", "--home", d, "--name", "remote_login", "log in to a remote computer", "--dry-run"])
            M.main(["grow", "--home", d, "--name", "remote_login", "log in to a remote computer", "--alpha", "0.01"])
            M.main(["concepts", "--home", d])
        txt = out.getvalue()
        self.assertIn("made in", txt)
        self.assertIn("dry run", txt)
        self.assertIn("ssh build server", txt)                                # closest recent event shown
        self.assertEqual(len(st.concepts()), 1)
        with self.assertRaises(SystemExit):
            M.main(["grow", "--home", d, "--name", "Bad Name", "x y z"])
        shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()


class TestGrownNeuronRequests(unittest.TestCase):
    def test_only_high_severity_grown_neurons_ask_for_action(self):
        from sentinel import actuators
        a = {"severity": "medium", "source": "concept:x", "title": "t", "target": {"kind": "process", "pid": 1234}}
        self.assertIsNone(actuators.propose(a))
        self.assertEqual(actuators.propose(dict(a, severity="high"))[0], "end_process")
        self.assertEqual(actuators.propose(dict(a, source="instinct"))[0], "end_process")
