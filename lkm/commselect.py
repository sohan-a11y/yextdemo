"""Choose how the cells talk, on VALIDATION worlds only (no test world is touched).

For each communication mode (continuous messages, words only, words + continuous) a cell field is trained on every
synthetic family and played on that family's validation worlds with the same adaptation grid for every mode; the
shifted family deploys the contagion-trained field on shifted validation worlds. The real mesh enters through the
validation AUC of lkm.langstudy.

Pre-declared rule (written before running):
  candidates are the modes that give the organism a vocabulary (lex, both); choose the one with the higher mean
  validation hit rate over {control, contagion, cascade, lexical (names readable), shifted};
  reject a candidate whose real-data validation AUC is more than 0.003 below the best mode.
Continuous-only is reported as the reference.
Run: python -m lkm.commselect
"""
import json, os, time
import numpy as np
from .baselines import AgeRule, GBDT, oof_scores
from .worlds import make_bundle
from . import nca_train as NT
from . import pipeline as PL

V = 16
CFG = dict(PL.CFG_FULL)
AD = (None, 40.0, 200.0)


def log(*a):
    print(*a, flush=True)


class M:
    pass


def best_cells(model, val_sets):
    scores = {}
    for ad in AD:
        spec = {"m": "cells", "adapt": ad, "eps": 0.0}
        r = PL.play([spec], val_sets, model, CFG, seeds=2, n_proc=1)
        scores[PL.spec_name(spec)] = float(list(r.values())[0]["hit"].mean())
    nm = max(scores, key=scores.get)
    return scores[nm], nm


def main(out_dir="lkm_out_final", steps=600):
    fams = [("control", "control", False, 0.5, 0.10), ("contagion", "contagion", False, 0.5, 0.10), ("cascade", "cascade", False, 0.2, 0.08),
            ("lexical (names readable)", "lexical", True, 0.3, 0.10)]
    out = {"modes": {}}
    shifted_val = make_bundle("shifted", range(700, 706), n=5000)
    for fname, kind, lang, rho, br in fams:
        t0 = time.time()
        train = make_bundle(kind, range(100, 108), n=5000, rho=rho, base_rate=br, lang=lang)
        val = make_bundle(kind, range(500, 506), n=5000, rho=rho, base_rate=br, lang=lang)
        gb = GBDT("graph").fit(train)
        oof = oof_scores(train, lambda: GBDT("graph"), folds=4)
        for comm in ("cont", "lex", "both"):
            P, _ = NT.train(train, oof, val=train[0], val_base=oof[0], steps=steps, V=0 if comm == "cont" else V, comm=comm, log=lambda *_: None)
            m = M(); m.graph, m.P, m.vocab = gb, P, (0 if comm == "cont" else V)
            sc, nm = best_cells(m, val)
            out["modes"].setdefault(comm, {})[fname] = {"hit": sc, "spec": nm}
            if kind == "contagion":
                sc2, nm2 = best_cells(m, shifted_val)
                out["modes"][comm]["shifted"] = {"hit": sc2, "spec": nm2}
            log(f"[commselect] {fname} {comm}: {sc:.4f} ({nm})" + (f" | shifted {sc2:.4f} ({nm2})" if kind == "contagion" else "") + f" [{time.time()-t0:.0f}s]")
            json.dump(out, open(os.path.join(out_dir, "commselect.json"), "w"), indent=1)
    ls = json.load(open(os.path.join(out_dir, "langstudy.json")))
    ds = ls["decision"]["dataset"]
    real = {k: ls[ds][f"cells_{k}"]["auc"] for k in ("cont", "lex", "both")}
    best_real = max(real.values())
    mean = {c: float(np.mean([v["hit"] for v in out["modes"][c].values()])) for c in out["modes"]}
    ok = {c: real[c] >= best_real - 0.003 for c in ("lex", "both")}
    cands = [c for c in ("lex", "both") if ok[c]] or ["both"]
    comm = max(cands, key=lambda c: mean[c])
    out["real_val_auc"] = real; out["mean_val_hit"] = mean; out["real_ok"] = ok
    out["decision"] = {"comm": comm, "rule": "higher mean validation hit over synthetic families among {lex, both}; reject if real-data validation AUC is >0.003 below the best mode"}
    log("[commselect] decision", out["decision"], mean, real)
    json.dump(out, open(os.path.join(out_dir, "commselect.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
