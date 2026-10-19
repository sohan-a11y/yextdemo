"""Figures for the paper, from research/results/instant_neurons.json.  Run: python -m research.figures"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join("research", "results", "figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a2"      # validated light-mode slots 1-3; gray = reference

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8})

NAMES = {"desc|k=0": "new neuron from the sentence (0 examples)", "proto|k=3": "sentence + 3 examples", "proto|k=10": "sentence + 10 examples",
         "logreg|k=10": "trained classifier, 10 examples", "keyword|k=0": "keyword match of the sentence", "rule|k=3": "rule from 3 examples (tool names)",
         "tfidf|k=10": "character n-grams, 10 examples", "lmn0_16|k=0": "16 existing LM neurons the sentence excites",
         "lmn_16|k=10": "16 existing LM neurons picked with 10 examples", "lmn0_1|k=0": "1 existing LM neuron the sentence excites"}
FAMILY = {"desc|k=0": BLUE, "proto|k=3": BLUE, "proto|k=10": BLUE, "logreg|k=10": BLUE, "keyword|k=0": AQUA, "rule|k=3": AQUA, "tfidf|k=10": AQUA,
          "lmn0_16|k=0": ORANGE, "lmn_16|k=10": ORANGE, "lmn0_1|k=0": ORANGE}


def bars(r, field, fname, title, xlabel, xlim):
    t = r["concepts_table"]
    keys = [k for k in NAMES if k in t and t[k].get(field) is not None]
    keys.sort(key=lambda k: t[k][field])
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(keys) + 1.4))
    y = np.arange(len(keys))
    vals = [t[k][field] for k in keys]
    ax.barh(y, [v - xlim[0] for v in vals], left=xlim[0], height=0.55, color=[FAMILY[k] for k in keys])
    for yi, v in zip(y, vals):
        ax.text(v + (xlim[1] - xlim[0]) * 0.01, yi, f"{v:.2f}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, [NAMES[k] for k in keys]); ax.set_xlim(*xlim); ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="new neuron built in the model's meaning space"), Patch(color=ORANGE, label="neurons that already exist inside the model"),
                       Patch(color=AQUA, label="lexical baselines (no language model)")], loc="lower right", frameon=False, fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=160); plt.close(fig)


def feedback(r):
    fb = r["feedback"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for learn, color, label in ((False, ORANGE, "neuron left as it was born"), (True, BLUE, "neuron learning from every answer")):
        curves = [e["curve"] for e in fb if e["learn"] == learn]
        t = [c["t"] for c in curves[0]]
        fp = np.mean([[c["fp"] for c in cv] for cv in curves], axis=0)
        ax.plot(t, fp, color=color, lw=2, solid_capstyle="round", marker="o", ms=4.5, mec=SURFACE, mew=1.5, label=label)
        ax.text(t[-1] + 80, fp[-1], f"{fp[-1]:.1f}", va="center", color=INK2, fontsize=9)
    ax.set_xlabel("events read"); ax.set_ylabel("false alarms so far (mean per neuron)")
    ax.legend(frameon=False, loc="upper left"); ax.set_xlim(0, t[-1] * 1.08)
    ax.set_title("One-shot learning from 'it was fine' cuts false alarms", loc="left", fontsize=11.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_feedback.png"), dpi=160); plt.close(fig)


def growth(r):
    g = [e for e in r["growth"] if e["vigilance"] != "exact-kNN"]
    knn = [e for e in r["growth"] if e["vigilance"] == "exact-kNN"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    v8 = sorted({e["capacity"] for e in g if e["vigilance"] == 0.8})
    au = [np.mean([e["auroc_new_vs_familiar"] for e in g if e["vigilance"] == 0.8 and e["capacity"] == c]) for c in v8]
    ax.plot(v8, au, color=BLUE, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label="growing memory (vigilance 0.8)")
    for c, a in zip(v8, au):
        ax.text(c, a + 0.008, f"{a:.3f}", ha="center", color=INK2, fontsize=9)
    if knn:
        kv = np.mean([e["auroc_new_vs_familiar"] for e in knn]); n = int(np.mean([e["prototypes"] for e in knn]))
        ax.axhline(kv, color=GRAY, lw=1.5)
        ax.text(v8[0], kv - 0.016, f"reference: keep every event ({n:,} vectors) {kv:.3f}", color=INK2, fontsize=9)
    ax.set_xscale("log"); ax.set_xticks(v8, [f"{c:,}" for c in v8]); ax.minorticks_off()
    ax.set_xlabel("memory capacity (prototype neurons)"); ax.set_ylabel("AUROC: new tools vs familiar tools")
    ax.set_title("A bounded, self-growing memory recognises what is new", loc="left", fontsize=11.5)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_growth.png"), dpi=160); plt.close(fig)


def per_concept(r):
    d = r["per_concept_desc_auroc_unseen"]
    ks = sorted(d, key=d.get)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    y = np.arange(len(ks))
    ax.hlines(y, 0.5, [d[k] for k in ks], color=GRID, lw=1)
    ax.plot([d[k] for k in ks], y, "o", color=BLUE, ms=7, mec=SURFACE, mew=1.5)
    for yi, k in zip(y, ks):
        ax.text(d[k] + 0.006, yi, f"{d[k]:.2f}", va="center", color=INK2, fontsize=8.5)
    ax.set_yticks(y, ks); ax.set_xlim(0.5, 1.03); ax.set_xlabel("AUROC on tools never shown (neuron made from the sentence alone)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Zero-shot neurons, concept by concept", loc="left", fontsize=11.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_per_concept.png"), dpi=160); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    r = json.load(open(os.path.join("research", "results", "instant_neurons.json")))
    bars(r, "auroc_unseen", "fig1_methods_auroc.png", "Recognising an activity on tools never shown", "AUROC (0.5 = chance) - mean over 20 concepts x 5 splits", (0.45, 1.0))
    bars(r, "tpr_unseen@0.01", "fig1b_methods_tpr.png", "Share caught when 1% false alarms are allowed", "true positive rate on tools never shown, alarm line set for 1% false alarms", (0.0, 0.75))
    per_concept(r)
    if r.get("feedback"):
        feedback(r)
    if r.get("growth"):
        growth(r)
    print("figures in", OUT)


if __name__ == "__main__":
    main()
