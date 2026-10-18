import unittest
import numpy as np
from tests import _common  # noqa
from lkm.worlds import make_bundle
from lkm.es import train_es
from lkm.organism import P_TOTAL


class TestES(unittest.TestCase):
    def test_es_runs_and_returns_genome(self):
        ds = make_bundle("contagion", range(1, 3), n=900)
        bases = [np.random.default_rng(i).normal(size=d.n).astype(np.float32) + 2.0 * d.y for i, d in enumerate(ds)]
        th, info = train_es(ds, bases, None, {"use_cells": False}, 0.05, 5, gens=2, pairs=3, episodes_per_eval=2, n_proc=2, log=lambda *a: None)
        self.assertEqual(th.shape, (P_TOTAL,)); self.assertTrue(np.isfinite(th).all())
        self.assertGreaterEqual(info["best"], info["start"])


class TestSpecs(unittest.TestCase):
    def test_e2_specs_follow_available_thetas(self):
        from lkm import pipeline as PL
        pick = {"online": {"m": "online", "adapt": None, "eps": 0.0}, "cells": {"m": "cells", "adapt": None, "eps": 0.0}, "org": {"m": "org", "adapt": None}}
        names = [PL.spec_name(x) for x in PL.e2_specs(pick, {"full": 1})]
        self.assertTrue(any(n.startswith("org:full") for n in names))
        self.assertFalse(any("no_scent" in n or "workers_only" in n or "blind" in n for n in names))
        full = PL.e2_specs(pick, None)
        self.assertTrue(any("blind" in PL.spec_name(x) for x in full))

    def test_primary_organism_detection(self):
        from lkm import pipeline as PL
        self.assertTrue(PL.is_primary_org("org:full[off]"))
        self.assertFalse(PL.is_primary_org("org:full[off,no_adapt]"))
        self.assertFalse(PL.is_primary_org("org:full[l2=40,no_evo]"))


if __name__ == "__main__":
    unittest.main()
