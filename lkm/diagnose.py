"""Diagnostics on the real mesh: is the bottleneck model capacity or information?

Everything here is scored on VALIDATION chunks only (the test pools are not touched).
  A. learning curve of the strongest static baseline (graph-feature GBDT) vs amount of training data
  B. capacity / communication sweep of the cell field (channels C, hidden H, update steps T), and a
     'stand-alone' cell field that gets no baseline logit at all
Run: python -m lkm.diagnose
"""
import json, os, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from .baselines import GBDT, oof_scores
from .metrics import auc, ap
from .realdata import build_real
from . import nca as N, nca_train as NT


def log(*a):
    print(*a, flush=True)


def pooled(chunks, scorer):
    s = [np.asarray(scorer(d, i), np.float64) for i, d in enumerate(chunks)]
    return np.concatenate(s)


def p_at(y, s, frac=0.03):
    k = max(1, int(frac * len(y)))
    return float(y[np.argsort(-s)[:k]].mean())


def report(y, s):
    return {"auc": auc(y, s), "ap": ap(y, s), "p@3%": p_at(y, s)}


def boot_auc_diff(y, a, b, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y); d = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        d.append(auc(y[ix], a[ix]) - auc(y[ix], b[ix]))
    d = np.asarray(d)
    return float(d.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main(out_dir="lkm_out", steps=500):
    t0 = time.time()
    R = build_real(log=log)
    train, vch = R["train"], R["val_chunks"]
    es_ix = [i for i, d in enumerate(vch) if d.n <= 13000]
    sel = vch[es_ix[0]]                                  # used only for early stopping of the NCA
    ev = [d for d in vch if d is not sel]                # evaluation chunks (never used for fitting/selection)
    y = np.concatenate([d.y for d in ev]).astype(np.int8)
    log(f"[diag] train chunks {len(train)}, eval chunks {len(ev)}, eval nodes {len(y)}, base rate {y.mean():.4f}")
    out = {"n_eval": int(len(y)), "base_rate": float(y.mean()), "curve": {}, "nca": {}}

    # ------------------------------------------------------------ A. learning curve
    for k in (2, 5, 11, len(train)):
        m = GBDT("graph").fit(train[:k])
        s = pooled(ev, lambda d, i: m.score(d))
        out["curve"][f"graph k={k} ({sum(d.n for d in train[:k])} nodes)"] = report(y, s)
        log("[diag] curve", k, out["curve"][f"graph k={k} ({sum(d.n for d in train[:k])} nodes)"])
    node = GBDT("node").fit(train)
    out["curve"]["node-only, all data"] = report(y, pooled(ev, lambda d, i: node.score(d)))
    big = GBDT("graph")
    big.m = HistGradientBoostingClassifier(max_iter=900, learning_rate=0.04, max_leaf_nodes=63, l2_regularization=1.0,
                                           early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=0)
    big.fit(train)
    out["curve"]["graph, 3x bigger GBDT"] = report(y, pooled(ev, lambda d, i: big.score(d)))
    log("[diag] node-only", out["curve"]["node-only, all data"], "bigger GBDT", out["curve"]["graph, 3x bigger GBDT"])

    # ------------------------------------------------------------ B. cell-field capacity sweep
    base_gb = GBDT("graph").fit(train)
    oof = oof_scores(train, lambda: GBDT("graph"), folds=4)
    base_eval = pooled(ev, lambda d, i: base_gb.score(d))
    out["nca"]["baseline (GBDT graph)"] = report(y, base_eval)
    sel_base = base_gb.score(sel)
    configs = [("cells C16 H32 T8 (the one we used)", 16, 32, 8, True),
               ("T=1: almost no communication", 16, 32, 1, True),
               ("T=16: longer communication", 16, 32, 16, True),
               ("wide: C48 H96 T8", 48, 96, 8, True),
               ("big: C64 H128 T12", 64, 128, 12, True),
               ("stand-alone cells (no baseline logit) C16 H32 T8", 16, 32, 8, False),
               ("stand-alone big (no baseline logit) C64 H128 T12", 64, 128, 12, False)]
    for name, C, H, T, use_base in configs:
        t1 = time.time()
        bl = oof if use_base else [np.zeros(d.n) for d in train]
        vb = sel_base if use_base else np.zeros(sel.n)
        P, info = NT.train(train, bl, val=sel, val_base=vb, steps=steps, T=T, C=C, H=H, log=lambda *_: None)
        Pn = {k: np.asarray(v) for k, v in P.items()}

        def sc(d, i):
            b = base_gb.score(d) if use_base else np.zeros(d.n)
            dl, _ = N.forward(Pn, N.build_inp(d.X, b), N.tables(d.graph()), T)
            return b + dl
        s = pooled(ev, sc)
        r = report(y, s)
        r["auc_vs_baseline"] = boot_auc_diff(y, s, base_eval)
        r["params"] = int(sum(v.size for v in Pn.values())); r["seconds"] = round(time.time() - t1)
        out["nca"][name] = r
        log("[diag]", name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "auc_vs_baseline"},
            "dAUC vs base", [round(x, 4) for x in r["auc_vs_baseline"]])
        os.makedirs(out_dir, exist_ok=True)
        json.dump(out, open(os.path.join(out_dir, "diagnose.json"), "w"), indent=1)
    out["seconds"] = round(time.time() - t0)
    json.dump(out, open(os.path.join(out_dir, "diagnose.json"), "w"), indent=1)
    log(f"[diag] done in {out['seconds']}s")


if __name__ == "__main__":
    main()
