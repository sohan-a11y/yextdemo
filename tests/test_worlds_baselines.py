import unittest
import numpy as np
from tests import _common  # noqa
from lkm.worlds import make_world, make_bundle
from lkm.baselines import GBDT, AgeRule, LogReg, fit_offset_logreg, apply_offset_logreg, oof_scores
from lkm.metrics import auc


def neighbour_label_corr(ds):
    G = ds.graph()
    nm = G.Aun @ ds.y.astype(np.float32)
    return float(np.corrcoef(nm, ds.y)[0, 1])


class TestWorlds(unittest.TestCase):
    def test_deterministic(self):
        a, b = make_world(5, n=800), make_world(5, n=800)
        self.assertTrue(np.array_equal(a.X, b.X)); self.assertTrue(np.array_equal(a.y, b.y))

    def test_base_rate(self):
        for kind in ("control", "contagion", "cascade", "shifted"):
            w = make_world(1, n=4000, kind=kind, base_rate=0.1)
            self.assertAlmostEqual(w.y.mean(), 0.1, delta=0.03)

    def test_control_has_no_graph_signal_contagion_has(self):
        c = np.mean([neighbour_label_corr(make_world(s, n=4000, kind="control")) for s in range(3)])
        k = np.mean([neighbour_label_corr(make_world(s, n=4000, kind="contagion")) for s in range(3)])
        self.assertLess(abs(c), 0.06)
        self.assertGreater(k, 0.08)

    def test_features_finite_and_shapes(self):
        w = make_world(2, n=1000)
        self.assertTrue(np.isfinite(w.X).all()); self.assertEqual(w.X.shape[0], len(w.y))


class TestBaselines(unittest.TestCase):
    def test_gbdt_beats_age_rule_on_control(self):
        tr = make_bundle("control", range(10, 13), n=3000); te = make_world(99, n=3000, kind="control")
        g = GBDT("node", max_iter=80).fit(tr)
        self.assertGreater(auc(te.y, g.score(te)), 0.68)
        self.assertGreater(auc(te.y, g.score(te)), auc(te.y, AgeRule().score(te)) + 0.03)

    def test_offset_logreg_recovers_signal(self):
        r = np.random.default_rng(0); F = r.normal(size=(4000, 3)); off = r.normal(size=4000)
        true = np.array([1.5, -1.0, 0.0, 0.3]); p = 1 / (1 + np.exp(-(F @ true[:3] + true[3] + off)))
        y = (r.random(4000) < p).astype(float)
        w = fit_offset_logreg(F, y, off, l2=1.0)
        self.assertTrue(np.allclose(w, true, atol=0.2))
        self.assertEqual(apply_offset_logreg(F, w, off).shape, (4000,))

    def test_offset_logreg_shrinks_with_large_l2(self):
        r = np.random.default_rng(1); F = r.normal(size=(100, 3)); y = (F[:, 0] > 0).astype(float)
        self.assertLess(np.abs(fit_offset_logreg(F, y, np.zeros(100), l2=1e6)[:-1]).max(), 0.01)   # weights shrink (intercept is lightly penalised)

    def test_greedy_reveal_gbdt_trains_and_is_leak_free(self):
        tr = make_bundle("contagion", range(30, 33), n=1500)
        m = GBDT("rev", greedy_aug=True, max_iter=30).fit(tr)
        d = make_world(77, n=1500)
        rp = np.zeros(d.n, np.float32); rp[:30] = d.y[:30]
        s0 = m.score(d); s1 = m.score(d, rev_pos=rp)
        self.assertTrue(np.isfinite(s1).all())
        self.assertGreater(np.abs(s0 - s1).max(), 1e-6)         # revealed information is actually used

    def test_oof_shapes(self):
        tr = make_bundle("control", range(20, 24), n=1500)
        oof = oof_scores(tr, lambda: GBDT("node", max_iter=20), folds=2)
        self.assertEqual([len(o) for o in oof], [d.n for d in tr])


if __name__ == "__main__":
    unittest.main()
