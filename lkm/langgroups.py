"""Where does the gain from reading come from? Feature-group ablation for the boosted-tree baseline on the real mesh,
scored on validation chunks only. Each group is added to the 50 original features on its own, and removed from the full set.
Run: python -m lkm.langgroups
"""
import json, os, time
import numpy as np
from .baselines import GBDT
from .realdata import build_real
from .diagnose import pooled, report, boot_auc_diff

GROUPS = {
    "website / phone / address strings": lambda n: n.startswith(("w_", "p_", "a_", "loc_", "b_namematch")),
    "source confidence": lambda n: n in ("s_baseconf", "s_sigconf", "s_confage"),
    "lexical flags in the name": lambda n: n.startswith("lx_"),
    "near-duplicate / same-name neighbours": lambda n: n.startswith("nn_"),
    "MiniLM semantic code of the name": lambda n: n.startswith("tx"),
}


def log(*a):
    print(*a, flush=True)


def main(out_dir="lkm_out_final"):
    R = build_real(language=True, lang_edges=False, log=log)
    train, vch = R["train"], R["val_chunks"]
    names = list(train[0].feat_names)
    es_ix = [i for i, d in enumerate(vch) if d.n <= 13000]
    sel = vch[es_ix[0]]
    ev = [d for d in vch if d is not sel]
    y = np.concatenate([d.y for d in ev]).astype(np.int8)
    orig = [j for j, n in enumerate(names) if not any(g(n) for g in GROUPS.values())]
    assert len(orig) >= 40, len(orig)

    def fit_eval(cols):
        tr = [d.subset(np.arange(d.n)) for d in train]
        for d in tr:
            d.X = d.X[:, cols]; d.agg_cols = [cols.index(c) for c in d.agg_cols if c in cols]; d.age_col = cols.index(d.age_col)
        ve = [d.subset(np.arange(d.n)) for d in ev]
        for d in ve:
            d.X = d.X[:, cols]; d.agg_cols = [cols.index(c) for c in d.agg_cols if c in cols]; d.age_col = cols.index(d.age_col)
        m = GBDT("graph").fit(tr)
        return report(y, pooled(ve, lambda d, i: m.score(d))), pooled(ve, lambda d, i: m.score(d))

    out = {"base_features": len(orig)}
    r0, s0 = fit_eval(orig)
    out["original features only"] = r0
    full_cols = list(range(len(names)))
    rf, sf = fit_eval(full_cols)
    out["all features"] = rf
    out["groups"] = {}
    for gname, g in GROUPS.items():
        cols_g = [j for j, n in enumerate(names) if g(n)]
        if not cols_g:
            continue
        r_add, s_add = fit_eval(orig + cols_g)
        r_add["auc_vs_original"] = boot_auc_diff(y, s_add, s0)
        r_del, s_del = fit_eval([j for j in full_cols if j not in cols_g])
        r_del["auc_vs_all"] = boot_auc_diff(y, s_del, sf)
        out["groups"][gname] = {"n_features": len(cols_g), "added_alone": r_add, "removed_from_all": r_del}
        log(f"[groups] {gname}: alone {r_add['auc']:.4f} (d {r_add['auc_vs_original'][0]:+.4f}); removed {r_del['auc']:.4f} (d {r_del['auc_vs_all'][0]:+.4f})")
        json.dump(out, open(os.path.join(out_dir, "langgroups.json"), "w"), indent=1)
    json.dump(out, open(os.path.join(out_dir, "langgroups.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
