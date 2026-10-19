import unittest
import numpy as np
from tests import _common  # noqa
from lkm.graph import Graph, knn_edges, group_edges, sym_csr, REL, NSLOT
from lkm import metrics as M
from sklearn.metrics import roc_auc_score


class TestGraph(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(0)
        self.xy = r.normal(size=(300, 2))
        grp = np.where(r.random(300) < 0.4, r.integers(0, 20, 300), -1)
        self.edges = {"sp": knn_edges(self.xy, 5), "br": group_edges(grp, 2, r), "ad": (np.array([1, 2]), np.array([2, 3]))}
        self.G = Graph.from_edges(300, self.edges)

    def test_symmetric_no_loops_binary(self):
        for rl in REL:
            A = self.G.A[rl]
            self.assertEqual((A != A.T).nnz, 0)
            self.assertEqual(A.diagonal().sum(), 0)
            self.assertTrue(set(np.unique(A.data)) <= {1.0})

    def test_slots_valid_unique_per_relation(self):
        nb, rel = self.G.nbr, self.G.nbr_rel
        self.assertEqual(nb.shape, (300, NSLOT))
        for i in range(0, 300, 7):
            for r in range(3):
                row = nb[i][(rel[i] == r) & (nb[i] >= 0)]
                self.assertEqual(len(row), len(set(row.tolist())))
                self.assertNotIn(i, row)

    def test_induced_matches_filter(self):
        idx = np.arange(0, 300, 2)
        sub = Graph.induced(300, self.edges, idx)
        self.assertEqual(sub.n, len(idx))
        # every induced edge must exist in the parent graph between the mapped nodes
        A = self.G.Au
        As = sub.Au.tocoo()
        for a, b in zip(As.row[:200], As.col[:200]):
            self.assertEqual(A[idx[a], idx[b]], 1.0)

    def test_row_normalised(self):
        s = np.asarray(self.G.An["sp"].sum(1)).ravel()
        self.assertTrue(np.all((np.abs(s - 1) < 1e-5) | (s == 0)))


class TestMetrics(unittest.TestCase):
    def test_auc_matches_sklearn(self):
        r = np.random.default_rng(1)
        y = (r.random(500) < 0.2).astype(int); s = r.normal(size=500) + y
        self.assertAlmostEqual(M.auc(y, s), roc_auc_score(y, s), places=9)

    def test_lift_perfect_and_random(self):
        y = np.zeros(1000, int); y[:100] = 1
        self.assertAlmostEqual(M.lift_at_frac(y, y.astype(float), 0.05), 10.0)
        r = np.random.default_rng(2)
        self.assertAlmostEqual(M.lift_at_frac(y, r.random(1000), 0.5), 1.0, delta=0.4)

    def test_precision_at_k(self):
        y = np.array([1, 0, 1, 0, 0]); s = np.array([5, 4, 3, 2, 1])
        self.assertAlmostEqual(M.precision_at_k(y, s, 3), 2 / 3)

    def test_bootstrap_contains_mean(self):
        m, lo, hi = M.bootstrap_ci(np.random.default_rng(3).normal(1, 0.1, 30))
        self.assertTrue(lo <= m <= hi)

    def test_paired_diff_sign(self):
        a = np.array([0.5, 0.6, 0.7, 0.55]); b = a - 0.1
        d, lo, hi, p = M.paired_diff_ci(a, b)
        self.assertAlmostEqual(d, 0.1, places=6); self.assertGreater(lo, 0)

    def test_discovery_stats_oracle(self):
        s = M.discovery_stats([5, 5], [5, 5], 100)
        self.assertAlmostEqual(s["frac_of_oracle"], 1.0); self.assertAlmostEqual(s["curve_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
