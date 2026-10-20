"""Two more ways to make a neuron from a sentence, suggested by prior work, on exactly the same splits as research/instant_neurons.py.

  contrast       the direction from a generic sentence to the description (contrast-pair probes, representation engineering):
                 p = e(description) - e("an ordinary computer command")
  selfrefine_m   description prototype refined with the m local-history events closest to it, no labels (in the spirit of Lbl2Vec /
                 cold-start adaptation): p = e(d) + mean(top-m calibration events). Tests whether the machine's own unlabeled history
                 helps a zero-shot neuron.
Run: python -m research.extra_methods  (needs the embedding cache written by research.instant_neurons)
"""
import json
import os

import numpy as np

from sentinel.cortex import unit
from research.instant_neurons import CONCEPTS, CACHE, OUT, MODEL, load_tldr, membership, auroc, at_alpha, paired_boot, log

GENERIC = "an ordinary computer command"


def main(seeds=5):
    from sentinel.cortex import Cortex
    rows = load_tldr()
    z = np.load(os.path.join(CACHE, "tldr_EA.npz"))
    E = z["E"]
    cx = Cortex(MODEL, threads=4)
    mem = membership(rows)
    g = cx.encode([GENERIC])[0]
    res = []
    for key, (desc, para, tools, rx) in CONCEPTS.items():
        m = mem[key]
        pos_all = np.where(m["pos"])[0]
        neg_all = np.where(~m["pos"] & ~m["excl"])[0]
        tools_present = sorted(set(m["tool"][pos_all]))
        dv = cx.encode([desc])[0]
        for seed in range(seeds):
            rng = np.random.default_rng(1000 * seed + sum(map(ord, key)))          # same calls, same order: same splits as the main run
            perm = rng.permutation(tools_present)
            n_seen = max(1, len(perm) // 2)
            unseen_t = set(perm[n_seen:])
            unseen_idx = np.array([i for i in pos_all if m["tool"][i] in unseen_t])
            negp = rng.permutation(neg_all)
            cal, tneg = negp[: len(negp) // 2], negp[len(negp) // 2:]
            if not len(unseen_idx):
                continue
            protos = {"desc": dv, "contrast": unit(dv - g)}
            sc = E[cal] @ dv
            for mm in (10, 30, 100):
                top = cal[np.argsort(-sc)[:mm]]
                protos[f"selfrefine_{mm}"] = unit(dv + E[top].mean(0))
            for name, p in protos.items():
                su, sn, scal = E[unseen_idx] @ p, E[tneg] @ p, E[cal] @ p
                tpr, fpr = at_alpha(scal, su, sn, 0.01)
                res.append({"concept": key, "seed": seed, "method": name, "auroc_unseen": auroc(su, sn), "tpr_unseen@0.01": tpr, "fpr_test@0.01": fpr})
        log(f"[extra] {key}")
    out = {}
    names = sorted({r["method"] for r in res})
    per = {n: [np.mean([r["auroc_unseen"] for r in res if r["method"] == n and r["concept"] == c]) for c in CONCEPTS] for n in names}
    for n in names:
        out[n] = {"auroc_unseen": float(np.mean(per[n])), "tpr_unseen@0.01": float(np.mean([r["tpr_unseen@0.01"] for r in res if r["method"] == n]))}
    out["paired_vs_desc"] = {n: paired_boot(per[n], per["desc"]) for n in names if n != "desc"}
    json.dump({"summary": out, "rows": res}, open(os.path.join(OUT, "extra_methods.json"), "w"), indent=1)
    for n in names:
        log(f"{n:16s} AUROC unseen {out[n]['auroc_unseen']:.3f}  TPR@1% {out[n]['tpr_unseen@0.01']:.3f}" +
            (f"   vs desc {out['paired_vs_desc'][n][0]:+.3f} [{out['paired_vs_desc'][n][1]:+.3f},{out['paired_vs_desc'][n][2]:+.3f}]" if n != "desc" else ""))


if __name__ == "__main__":
    main()
