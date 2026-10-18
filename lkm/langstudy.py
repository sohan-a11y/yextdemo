"""Language study on the real mesh, scored on VALIDATION chunks only (the test pools are not touched).

  R0  the numbers from lkm_out/diagnose.json (no language; same evaluation chunks)
  R1  + text features (string classes, lexical flags, name-neighbour features, MiniLM semantic code)
  R2  + language edges (same-name families joined into the brand relation)
For every variant the strongest static baseline (graph-feature GBDT) gets exactly the same inputs as the cells.
On R2 the cells are trained three ways: continuous messages, words only ('lex'), and both.
The configuration for the final build is picked here by validation AUC with a pre-declared rule:
words-only is chosen if it is within TOL AUC of the best communication mode (the organism should speak).
Run: python -m lkm.langstudy
"""
import json, os, sys, time
import numpy as np
from .baselines import GBDT, oof_scores
from .metrics import auc, ap
from .realdata import build_real
from .diagnose import pooled, report, boot_auc_diff
from . import nca as N, nca_train as NT

TOL = 0.002
V = 16


def log(*a):
    print(*a, flush=True)


def run_variant(name, R, out, steps, comms):
    train, vch = R["train"], R["val_chunks"]
    es_ix = [i for i, d in enumerate(vch) if d.n <= 13000]
    sel = vch[es_ix[0]]
    ev = [d for d in vch if d is not sel]
    y = np.concatenate([d.y for d in ev]).astype(np.int8)
    t0 = time.time()
    gb = GBDT("graph").fit(train)
    base_eval = pooled(ev, lambda d, i: gb.score(d))
    out[name] = {"features": int(train[0].X.shape[1]), "gbdt_graph": report(y, base_eval)}
    log(f"[lang] {name}: GBDT {out[name]['gbdt_graph']} ({time.time()-t0:.0f}s)")
    oof = oof_scores(train, lambda: GBDT("graph"), folds=4)
    sel_base = gb.score(sel)
    for comm in comms:
        t1 = time.time()
        Vc = 0 if comm == "cont" else V
        P, info = NT.train(train, oof, val=sel, val_base=sel_base, steps=steps, V=Vc, comm=comm, log=lambda *_: None)
        Pn = {k: np.asarray(v) for k, v in P.items()}

        def sc(d, i):
            b = gb.score(d)
            dl, _ = N.forward(Pn, N.build_inp(d.X, b), N.tables(d.graph()))
            return b + dl
        s = pooled(ev, sc)
        r = report(y, s)
        r["auc_vs_gbdt"] = boot_auc_diff(y, s, base_eval)
        r["seconds"] = round(time.time() - t1)
        out[name][f"cells_{comm}"] = r
        log(f"[lang] {name} cells[{comm}] auc {r['auc']:.4f} p@3% {r['p@3%']:.4f} dAUC {r['auc_vs_gbdt'][0]:+.4f} ({r['seconds']}s)")
        json.dump(out, open(os.path.join(out_dir_of(out), "langstudy.json"), "w"), indent=1)
    return y, base_eval


def out_dir_of(out):
    return out["_dir"]


def main(out_dir="lkm_out_final", steps=500):
    os.makedirs(out_dir, exist_ok=True)
    out = {"_dir": out_dir, "V": V, "tol": TOL}
    diag = json.load(open(os.path.join("lkm_out", "diagnose.json")))
    out["R0"] = {"features": 50, "gbdt_graph": diag["nca"]["baseline (GBDT graph)"],
                 "cells_cont": diag["nca"]["cells C16 H32 T8 (the one we used)"]}
    R1 = build_real(language=True, lang_edges=False, log=log)
    run_variant("R1", R1, out, steps, ("cont", "lex", "both"))
    R2 = build_real(language=True, lang_edges=True, log=log)
    run_variant("R2", R2, out, steps, ("cont", "lex", "both"))
    # decision rule (validation only)
    variant = "R2" if out["R2"]["cells_cont"]["auc"] >= out["R1"]["cells_cont"]["auc"] - 1e-9 else "R1"
    c = {k: out[variant][f"cells_{k}"]["auc"] for k in ("cont", "lex", "both")}
    best = max(c, key=c.get)
    comm = "lex" if c["lex"] >= c[best] - TOL else best
    out["decision"] = {"comm": comm, "dataset": variant, "val_auc_by_comm": c,
                       "rule": f"words-only if within {TOL} AUC of the best mode; dataset = higher cell-field AUC of R1/R2"}
    log("[lang] decision", out["decision"])
    json.dump(out, open(os.path.join(out_dir, "langstudy.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
