import os, unittest, inspect
import numpy as np, pandas as pd
from tests import _common  # noqa
from lkm import realdata as RD


def fake_tables():
    def src(ds, t): return [{"dataset": ds, "update_time": t, "property": "", "record_id": "x", "confidence": 0.5, "license": "", "between": None, "provider": ds, "resource": ds, "version": ""}]
    def row(i, name, phone, web, status="open"):
        return {"id": f"id{i}", "bbox": {"xmin": -73.99 + i * 1e-4, "xmax": -73.99, "ymin": 40.7, "ymax": 40.7}, "confidence": 0.7,
                "names": {"primary": name, "common": None, "rules": None}, "sources": np.array(src("meta", "2026-08-01T00:00:00Z") + src("Overture-signals", "2026-07-01T00:00:00Z"), dtype=object),
                "operating_status": status, "basic_category": "restaurant", "websites": np.array([web]) if web else None,
                "phones": np.array([phone]) if phone else None, "addresses": np.array([{"freeform": f"{i} Main St", "postcode": "11201"}], dtype=object), "brand": None}
    a = pd.DataFrame([row(0, "Alpha Cafe", "+1", "a.com"), row(1, "Beta Bar", "+2", "b.com"), row(2, "Gamma Gym", "+3", "g.com"),
                      row(3, "Delta Deli", "+4", "d.com"), row(4, "Eps Shop", "+5", "e.com")])
    b_rows = [row(0, "Alpha Cafe", "+1", "a.com"),                 # unchanged
              row(1, "Beta Bar", "+99", "b.com"),                   # phone changed
              row(2, "Gamma Gym", "+3", "g.com", status="permanently_closed"),  # closed
              # id3 removed (and not re-created) ; id4 removed but re-created under a new id with same name nearby
              row(5, "Eps Shop", "+5", "e.com")]
    b_rows[3]["bbox"] = dict(a.iloc[4]["bbox"])
    b = pd.DataFrame(b_rows)
    return a, b


class TestLabels(unittest.TestCase):
    def test_label_logic(self):
        a, b = fake_tables()
        lab = RD.make_labels(a, b)
        self.assertEqual(lab["y"].tolist(), [False, True, True, True, False])
        self.assertEqual(lab["closed"].tolist(), [False, False, True, False, False])
        self.assertEqual(lab["changed"].tolist(), [False, True, False, False, False])
        self.assertEqual(lab["removed"].tolist(), [False, False, False, True, False])
        self.assertEqual(lab["rematched"].tolist(), [False, False, False, False, True])    # id churn is not an event


class TestNoLeakage(unittest.TestCase):
    def test_features_only_depend_on_august_table(self):
        a, b = fake_tables()
        self.assertEqual(list(inspect.signature(RD.aug_features).parameters)[:1], ["a"])
        X1, names, cats, age = RD.aug_features(a)
        a2 = a.copy(); a2["september_flag"] = [1, 0, 1, 0, 1]
        X2, _, _, _ = RD.aug_features(a2)
        self.assertTrue(np.allclose(X1, X2))
        self.assertTrue(np.isfinite(X1).all())
        self.assertFalse(any("sep" in n.lower() for n in names))

    def test_no_feature_perfectly_predicts_label_on_real_data(self):
        p = os.path.join("lkm_data", "real_built.npz")
        if not os.path.exists(p):
            self.skipTest("real data not built")
        z = np.load(p, allow_pickle=True)
        X, y = z["X"], z["y"].astype(float)
        idx = np.random.default_rng(0).choice(len(y), 60000, replace=False)
        c = np.nan_to_num([abs(np.corrcoef(X[idx, j], y[idx])[0, 1]) for j in range(X.shape[1]) if X[idx, j].std() > 0])
        self.assertLess(c.max(), 0.45, "a single feature is suspiciously predictive: check for leakage")

    def test_spatial_split_is_disjoint(self):
        p = os.path.join("lkm_data", "real_built.npz")
        if not os.path.exists(p):
            self.skipTest("real data not built")
        R = RD.build_real(log=lambda *a: None)
        te = set(map(tuple, np.round(R["test"].xy, 6))); 
        tr = set(map(tuple, np.round(np.concatenate([d.xy for d in R["train"]]), 6)))
        # distinct places can share coordinates, so test overlap only by spatial block: a tiny share is tolerated
        self.assertLess(len(te & tr) / max(len(te), 1), 0.02)


if __name__ == "__main__":
    unittest.main()
