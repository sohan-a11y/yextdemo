import unittest
import numpy as np
from tests import _common  # noqa
from lkm import nca as N
from lkm.worlds import make_world, make_bundle
from lkm.baselines import GBDT
from lkm.game import Env, run_game, RandomPolicy, RankPolicy, Policy, CellsOnly
from lkm.organism import Organism, init_genome, unpack, P_TOTAL, HD


class TestNCA(unittest.TestCase):
    def setUp(self):
        self.ds = make_world(3, n=1200)
        self.base = np.random.default_rng(0).normal(size=self.ds.n).astype(np.float32)
        self.din = self.ds.X.shape[1] + 4

    def test_init_is_identity_residual(self):
        P = N.init_params(self.din)
        d, _ = N.forward(P, N.build_inp(self.ds.X, self.base), N.tables(self.ds.graph()))
        self.assertEqual(float(np.abs(d).max()), 0.0)

    def test_sparse_aggregation_equals_reference(self):
        G = self.ds.graph(); tabs = N.tables(G); idxm = N.tables_idx(G)
        s = np.random.default_rng(1).normal(size=(self.ds.n, 5)).astype(np.float32)
        for A, (idx, m) in zip(tabs, idxm):
            ref = (s[idx] * m[..., None]).sum(1) / np.maximum(m.sum(1, keepdims=True), 1.0)
            self.assertTrue(np.allclose(A @ s, ref, atol=1e-5))

    def test_jax_forward_matches_numpy(self):
        import jax.numpy as jnp
        import jax
        from lkm import nca_train as NT
        P = N.init_params(self.din, seed=2)
        P["wo"] = np.random.default_rng(3).normal(size=P["wo"].shape).astype(np.float32)
        inp = N.build_inp(self.ds.X, self.base)
        dn, _ = N.forward(P, inp, N.tables(self.ds.graph()))
        tabs = [(jnp.asarray(i), jnp.asarray(m)) for i, m in N.tables_idx(self.ds.graph())]
        dj, _ = NT._forward_jax({k: jnp.asarray(v) for k, v in P.items()}, jnp.asarray(inp), tabs, jax.random.PRNGKey(0), N.T_DEFAULT, False)
        self.assertTrue(np.allclose(dn, np.asarray(dj), atol=1e-4))

    def test_revealed_labels_change_state(self):
        P = N.init_params(self.din, seed=4)
        P["wo"] = np.ones_like(P["wo"])
        t = N.tables(self.ds.graph())
        rp = np.zeros(self.ds.n, np.float32); rp[:50] = 1
        a, _ = N.forward(P, N.build_inp(self.ds.X, self.base), t)
        b, _ = N.forward(P, N.build_inp(self.ds.X, self.base, rp), t)
        self.assertGreater(float(np.abs(a - b).max()), 1e-4)


class Bad(Policy):
    def __init__(self, mode): self.mode = mode
    def pick(self, env, k):
        if self.mode == "over": return np.arange(k + 1)
        if self.mode == "dup": return np.array([0, 0])
        return np.arange(k)       # re-verification on second round


class TestGame(unittest.TestCase):
    def setUp(self):
        self.ds = make_world(4, n=2000)

    def test_budget_sum(self):
        e = Env(self.ds, 0.03, 8)
        self.assertEqual(int(e.B.sum()), 60); self.assertTrue(e.B.max() - e.B.min() <= 1)

    def test_rule_violations_raise(self):
        for m in ("over", "dup", "again"):
            with self.assertRaises(RuntimeError):
                run_game(self.ds, Bad(m), 0.03, 8, 0)

    def test_random_hit_rate_near_base_rate(self):
        r = [run_game(self.ds, RandomPolicy(), 0.1, 8, s)["hit_rate"] for s in range(12)]
        self.assertAlmostEqual(np.mean(r), self.ds.y.mean(), delta=0.03)

    def test_found_bounds_and_determinism(self):
        a = run_game(self.ds, RandomPolicy(), 0.03, 8, 5); b = run_game(self.ds, RandomPolicy(), 0.03, 8, 5)
        self.assertEqual(a["found_rounds"], b["found_rounds"])
        self.assertLessEqual(a["found"], min(sum(a["B"]), a["n_pos"]))

    def test_oracle_policy_hits_one(self):
        oracle = RankPolicy(self.ds.y.astype(float) + 1e-3 * np.random.default_rng(0).random(self.ds.n), "oracle")
        r = run_game(self.ds, oracle, 0.03, 8, 0)
        self.assertAlmostEqual(r["hit_rate"], 1.0)


class TestOrganism(unittest.TestCase):
    def setUp(self):
        self.ds = make_world(6, n=2500)
        self.base = np.random.default_rng(0).normal(size=self.ds.n).astype(np.float32) + 2.0 * self.ds.y

    def test_genome_layout(self):
        g = init_genome(0)
        self.assertEqual(len(g), P_TOTAL)
        u = unpack(np.stack([g, g]))
        self.assertEqual(u["Wxh"].shape[0], 2); self.assertEqual(u["Wm"].shape[-1], HD + 1)

    def test_runs_without_nan_and_respects_rules(self):
        o = Organism(init_genome(1), P=None, base=self.base, use_cells=False, seed=1)
        r = run_game(self.ds, o, 0.04, 8, 3)
        self.assertTrue(np.isfinite(o.energy).all()); self.assertTrue(np.isfinite(o.genomes).all())
        self.assertLessEqual(r["found"], sum(r["B"]))

    def test_workers_find_more_than_random_when_base_informative(self):
        h = [run_game(self.ds, Organism(init_genome(0), P=None, base=self.base, use_cells=False, seed=s), 0.04, 8, s)["hit_rate"] for s in range(3)]
        self.assertGreater(np.mean(h), 3 * self.ds.y.mean())

    def test_population_constant_and_evolution_changes_genomes(self):
        o = Organism(init_genome(1), P=None, base=self.base, use_cells=False, seed=2, evolve=True)
        run_game(self.ds, o, 0.04, 6, 1); n0 = o.W
        g_ev = o.genomes.copy()
        o2 = Organism(init_genome(1), P=None, base=self.base, use_cells=False, seed=2, evolve=False)
        run_game(self.ds, o2, 0.04, 6, 1)
        self.assertEqual(o.genomes.shape[0], n0)
        self.assertGreater(sum(o.log["births"]), 0); self.assertEqual(sum(o2.log["births"]), 0)
        self.assertGreater(np.abs(g_ev - o2.genomes).max(), 1e-3)

    def test_scent_deposited_then_fades(self):
        o = Organism(init_genome(1), P=None, base=self.base, use_cells=False, seed=3)
        run_game(self.ds, o, 0.04, 8, 2)
        self.assertGreater(o.hit.max(), 0)
        self.assertLessEqual(o.hit.max(), 1.0 / (1 - 0.85) + 1e-3)      # scent can never explode
        h = o.hit.copy()
        for _ in range(30):
            o.hit = (0.85 * (0.5 * o.hit + 0.5 * (o.G.Aun @ o.hit))).astype(np.float32)
        self.assertLess(o.hit.sum(), h.sum())

    def test_no_scent_ablation_has_zero_scent(self):
        o = Organism(init_genome(1), P=None, base=self.base, use_cells=False, use_scent=False, seed=3)
        run_game(self.ds, o, 0.04, 6, 2)
        self.assertEqual(float(o.hit.max()), 0.0)

    def test_seeded_determinism(self):
        mk = lambda: Organism(init_genome(2), P=None, base=self.base, use_cells=False, seed=9)
        a = run_game(self.ds, mk(), 0.04, 6, 4)["found_rounds"]; b = run_game(self.ds, mk(), 0.04, 6, 4)["found_rounds"]
        self.assertEqual(a, b)

    def test_risk_field_excludes_verified_and_is_a_distribution(self):
        o = Organism(init_genome(1), P=None, base=self.base, use_cells=False, seed=5)
        run_game(self.ds, o, 0.04, 6, 2)
        # after the last round re-derive the field for the final state
        class E: verified = np.zeros(self.ds.n, np.int8)
        E.verified[:100] = 1
        o._field_cdf(self.base, E)
        self.assertAlmostEqual(float(o._cdf[-1]), 1.0, places=6)
        self.assertEqual(float(o._cdf[99]), 0.0)                 # verified nodes carry no probability mass

    def test_informed_foraging_beats_blind_with_informative_base(self):
        mk = lambda inf, s: Organism(init_genome(0), P=None, base=self.base, use_cells=False, informed=inf, seed=s)
        inf = np.mean([run_game(self.ds, mk(True, s), 0.04, 8, s)["hit_rate"] for s in range(4)])
        bl = np.mean([run_game(self.ds, mk(False, s), 0.04, 8, s)["hit_rate"] for s in range(4)])
        self.assertGreater(inf, bl)

    def test_cells_policy_runs(self):
        P = N.init_params(self.ds.X.shape[1] + 4, seed=1)
        r = run_game(self.ds, CellsOnly(P, self.base, eps=0.0), 0.04, 6, 1)
        self.assertGreater(r["hit_rate"], 2 * self.ds.y.mean())


if __name__ == "__main__":
    unittest.main()
