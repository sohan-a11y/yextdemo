import inspect, unittest
import numpy as np, pandas as pd
from tests import _common  # noqa
from tests.test_realdata import fake_tables
from lkm import language as LG, nca as N, nca_train as NT, narrate as NR
from lkm.game import run_game
from lkm.organism import Organism, init_genome, unpack, p_total, P_TOTAL, dims, DD0
from lkm.worlds import make_world


def table_with(web=None, phone=None, name="Alpha Cafe", addr="12 Main St Ste 4", brand=None):
    a, _ = fake_tables()
    a = a.iloc[:1].copy()
    a["names"] = [{"primary": name, "common": None, "rules": None}]
    a["websites"] = [np.array([web]) if web else None]
    a["phones"] = [np.array([phone]) if phone else None]
    a["addresses"] = [np.array([{"freeform": addr, "postcode": "11201-1234", "locality": "Brooklyn"}], dtype=object)]
    a["brand"] = [brand]
    return a


class TestStringFeatures(unittest.TestCase):
    def col(self, F, names, n):
        return float(F[0, names.index(n)])

    def test_website_classes(self):
        F, n = LG.string_features(table_with(web="https://www.facebook.com/alphacafe"))
        self.assertEqual(self.col(F, n, "w_social"), 1.0); self.assertEqual(self.col(F, n, "w_own"), 0.0)
        F, n = LG.string_features(table_with(web="https://alphacafe.com/menu/lunch"))
        self.assertEqual(self.col(F, n, "w_own"), 1.0); self.assertEqual(self.col(F, n, "w_https"), 1.0)
        self.assertGreater(self.col(F, n, "w_namematch"), 0.0); self.assertGreater(self.col(F, n, "w_depth"), 0.0)
        F, n = LG.string_features(table_with(web="http://alpha.wixsite.com/x"))
        self.assertEqual(self.col(F, n, "w_builder"), 1.0); self.assertEqual(self.col(F, n, "w_https"), 0.0)

    def test_phone_address_brand(self):
        F, n = LG.string_features(table_with(phone="+18005551234"))
        self.assertEqual(self.col(F, n, "p_toll"), 1.0)
        F, n = LG.string_features(table_with(phone="+17185551234"))
        self.assertEqual(self.col(F, n, "p_nyc"), 1.0); self.assertEqual(self.col(F, n, "p_toll"), 0.0)
        F, n = LG.string_features(table_with(phone="+15125551234"))
        self.assertEqual(self.col(F, n, "p_other"), 1.0)
        self.assertEqual(self.col(F, n, "a_unit"), 1.0); self.assertEqual(self.col(F, n, "a_zip4"), 1.0)
        self.assertEqual(self.col(F, n, "loc_bk"), 1.0)
        F, n = LG.string_features(table_with(brand={"wikidata": None, "names": {"primary": "Alpha Cafe"}}))
        self.assertEqual(self.col(F, n, "b_namematch"), 1.0)

    def test_empty_fields_are_zero_not_error(self):
        F, n = LG.string_features(table_with())
        self.assertTrue(np.isfinite(F).all()); self.assertEqual(F.shape[1], len(n))

    def test_lexical_flags(self):
        f = LG.lexical_flags(["Smith Dental LLC", "ROCK 42 BAR", "", None])
        self.assertEqual(f[0, 0], 1.0); self.assertEqual(f[1, 1], 1.0); self.assertEqual(f[1, 2], 1.0)
        self.assertEqual(f[2, 6], 1.0); self.assertEqual(f[3, 6], 1.0)


class TestLanguageRelations(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(0)
        self.n, self.d = 300, 24
        base = r.normal(size=(self.n, self.d)).astype(np.float32)
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        self.E = base.copy()
        self.E[1] = self.E[0]                                   # place 1 has the same name as place 0
        for i in range(2, 12):                                   # a chain of 10 with the same name, spread over the city
            self.E[i] = self.E[2]
        self.xy = np.column_stack([r.uniform(0, 1, self.n), r.uniform(0, 1, self.n)]).astype(np.float32)
        self.xy[1] = self.xy[0] + 1e-5                           # 0 and 1 are next door
        self.xy[2:12] += 5.0 + np.arange(10)[:, None] * 0.01     # chain members are far from each other

    def test_next_door_duplicates_are_seen(self):
        f, names = LG.name_neighbour_features(self.E, self.xy, k=6, radius_deg=0.0008)
        self.assertGreater(f[0, names.index("nn_dup")], 0.0); self.assertGreater(f[1, names.index("nn_best")], 0.95)
        self.assertEqual(f[100, names.index("nn_dup")], 0.0)

    def test_semantic_edges_find_chain_far_apart(self):
        src, dst = LG.semantic_edges(self.E, np.ones(self.n, bool), k=3, thr=0.92)
        self.assertTrue((src != dst).all())
        chain = set(range(2, 12))
        inside = [(s, d) for s, d in zip(src, dst) if s in chain]
        self.assertGreater(len(inside), 10); self.assertTrue(all(d in chain for _, d in inside))
        self.assertLessEqual(np.bincount(src, minlength=self.n).max(), 3)
        unrelated = [s for s in src if s not in chain and s not in (0, 1)]
        self.assertEqual(len(unrelated), 0)

    def test_pca_whitens(self):
        X = np.random.default_rng(1).normal(size=(500, 30)).astype(np.float32) * np.linspace(0.2, 3, 30)
        p = LG.Pca().fit(X, k=8)
        Z = p.transform(X)
        self.assertEqual(Z.shape, (500, 8)); self.assertTrue(np.allclose(Z.std(0), 1.0, atol=0.1))


class TestNoLeakage(unittest.TestCase):
    def test_language_layer_never_sees_labels_or_september(self):
        params = list(inspect.signature(LG.build_layer).parameters)
        self.assertEqual(params[:4], ["a", "E", "xy", "train_mask"])
        self.assertFalse(any(p in params for p in ("y", "labels", "b", "sep")))
        a, _ = fake_tables()
        a["brand"] = [None] * len(a)
        F1, _ = LG.string_features(a)
        a2 = a.copy(); a2["september_flag"] = list(range(len(a)))
        F2, _ = LG.string_features(a2)
        self.assertTrue(np.allclose(F1, F2))


class TestLayer(unittest.TestCase):
    def test_build_layer_alignment_and_finiteness(self):
        a, _ = fake_tables()
        a["brand"] = [None] * len(a)
        n = len(a)
        r = np.random.default_rng(0)
        E = r.normal(size=(n, 20)).astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True)
        E[1] = E[0]
        xy = np.column_stack([np.arange(n) * 1e-5, np.zeros(n)]).astype(np.float32)
        out = LG.build_layer(a, E, xy, np.ones(n, bool), log=lambda *x: None)
        self.assertEqual(out["F"].shape, (n, len(out["names"])))
        self.assertTrue(np.isfinite(out["F"]).all())
        self.assertEqual(len(out["text"]), n); self.assertIn("Alpha Cafe", out["text"][0])
        self.assertTrue(((out["src"] >= 0) & (out["src"] < n) & (out["dst"] >= 0) & (out["dst"] < n)).all())
        self.assertGreater(out["F"][0, out["names"].index("nn_dup")], 0)       # places 0 and 1 share a name and sit next door

    def test_language_dataset_has_no_label_proxy(self):
        import os
        p = os.path.join("lkm_data", "real_lang_s0_t0.92.npz")
        q = os.path.join("lkm_data", "real_built.npz")
        if not (os.path.exists(p) and os.path.exists(q)):
            self.skipTest("language layer not built")
        F = np.load(p, allow_pickle=True)["F"]; y = np.load(q, allow_pickle=True)["y"].astype(float)
        idx = np.random.default_rng(0).choice(len(y), 60000, replace=False)
        c = np.nan_to_num([abs(np.corrcoef(F[idx, j], y[idx])[0, 1]) for j in range(F.shape[1]) if F[idx, j].std() > 0])
        self.assertLess(c.max(), 0.45, "a language feature is suspiciously predictive: check for leakage")


class TestLexiconChannel(unittest.TestCase):
    def setUp(self):
        self.ds = make_world(3, n=1000, kind="cascade")
        self.base = np.zeros(self.ds.n, np.float32)
        self.din = self.ds.X.shape[1] + 4

    def test_modes_have_expected_shapes_and_start_as_identity(self):
        t = N.tables(self.ds.graph())
        for comm in ("cont", "lex", "both"):
            P = N.init_params(self.din, V=6, comm=comm)
            self.assertEqual(N.comm_mode(P), comm if comm != "cont" else "cont")
            d, s = N.forward(P, N.build_inp(self.ds.X, self.base), t)
            self.assertEqual(float(np.abs(d).max()), 0.0)
            if comm != "cont":
                w = N.words(P, s); self.assertTrue(((w >= 0) & (w < 6)).all())

    def test_jax_matches_numpy_in_lexicon_modes(self):
        import jax, jax.numpy as jnp
        t = [(jnp.asarray(i), jnp.asarray(m)) for i, m in N.tables_idx(self.ds.graph())]
        inp = N.build_inp(self.ds.X, self.base)
        for comm in ("lex", "both"):
            P = N.init_params(self.din, seed=2, V=5, comm=comm)
            P["wo"] = np.random.default_rng(3).normal(size=P["wo"].shape).astype(np.float32)
            dn, _ = N.forward(P, inp, N.tables(self.ds.graph()))
            dj, _ = NT._forward_jax({k: jnp.asarray(v) for k, v in P.items()}, jnp.asarray(inp), t, jax.random.PRNGKey(0), N.T_DEFAULT, False)
            self.assertTrue(np.allclose(dn, np.asarray(dj), atol=1e-4), comm)

    def test_training_learns_and_uses_the_vocabulary(self):
        P, info = NT.train([self.ds], [self.base], val=self.ds, val_base=self.base, steps=80, V=6, comm="lex", log=lambda *a: None)
        self.assertGreater(np.abs(P["Wt"]).sum(), 0)
        d, s = N.forward(P, N.build_inp(self.ds.X, self.base), N.tables(self.ds.graph()))
        self.assertGreater(len(np.unique(N.words(P, s))), 2)           # words are not collapsed to one
        from lkm.metrics import auc
        self.assertGreater(auc(self.ds.y, d), 0.6)


class TestWorkersHearWords(unittest.TestCase):
    def setUp(self):
        self.ds = make_world(5, n=1200, kind="cascade")
        self.base = np.random.default_rng(0).normal(size=self.ds.n).astype(np.float32)
        self.V = 6
        self.P = N.init_params(self.ds.X.shape[1] + 4, V=self.V, comm="lex", seed=1)
        self.P["wo"] = np.random.default_rng(2).normal(size=self.P["wo"].shape).astype(np.float32) * 0.1

    def test_genome_dimensions(self):
        self.assertEqual(p_total(0), P_TOTAL)
        self.assertGreater(p_total(self.V), P_TOTAL)
        g = init_genome(0, self.V)
        self.assertEqual(len(g), p_total(self.V))
        u = unpack(g[None, :], self.V)
        self.assertEqual(u["wd"].shape[-1], dims(self.V)[0])

    def test_game_runs_with_words_and_respects_rules(self):
        o = Organism(init_genome(1, self.V), P=self.P, base=self.base, vocab=self.V, seed=3, record=True)
        r = run_game(self.ds, o, 0.05, 6, 3)
        self.assertTrue(np.isfinite(r["hit_rate"]))
        d, _ = o._descriptors(type("E", (), {"verified": np.zeros(self.ds.n, np.int8), "rev_pos": np.zeros(self.ds.n, np.float32),
                                              "rev_neg": np.zeros(self.ds.n, np.float32)})())
        self.assertEqual(d.shape[1], DD0 + self.V)
        self.assertTrue(np.allclose(d[:, DD0:].sum(1), 1.0))             # exactly one word per place
        self.assertTrue(all("workers" in e for e in o.rec))


class TestNarrator(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(0)
        n = 600
        a, _ = fake_tables()
        self.ds = make_world(2, n=n, kind="contagion")
        self.ds.text = np.array([f"Place {i} | cafe" for i in range(n)])
        self.V = 5
        self.P = N.init_params(self.ds.X.shape[1] + 4, V=self.V, comm="lex", seed=3)
        self.P["Wt"] = r.normal(size=self.P["Wt"].shape).astype(np.float32)
        self.base = r.normal(size=n)

    def test_dictionary_and_sentences(self):
        nar = NR.Narrator(self.ds, self.base, self.P)
        dic = nar.dictionary()
        self.assertEqual(len(dic), self.V); self.assertAlmostEqual(sum(e["share"] for e in dic), 1.0, places=5)
        used = next(e for e in dic if e["n"] > 0)
        self.assertIn("change_rate", used); self.assertIn("word", nar.say_word(used["word"], used))
        s = nar.explain(7, risk=0.42)
        self.assertIn("'Place 7'", s); self.assertIn("42%", s)

    def test_worker_log_is_grounded(self):
        nar = NR.Narrator(self.ds, self.base, self.P)
        rec = [{"ids": [3, 9], "labels": [1, 0], "workers": [11, 12]}]
        lines = nar.worker_log(rec, None, risk_fn=lambda i: 0.5)
        self.assertEqual(len(lines), 2); self.assertIn("worker 11", lines[0]); self.assertIn("it had changed", lines[0])
        self.assertIn("nothing had changed", lines[1])


if __name__ == "__main__":
    unittest.main()
