"""Evaluation metrics. All pure NumPy/sklearn so they can be unit-tested against references."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def auc(y, s):
    y = np.asarray(y);
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def ap(y, s):
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def precision_at_k(y, s, k):
    k = int(max(1, min(k, len(y))))
    top = np.argpartition(-np.asarray(s), k - 1)[:k]
    return float(np.asarray(y)[top].mean())


def lift_at_frac(y, s, frac):
    """Precision in the top `frac` of ranked items divided by the base rate."""
    y = np.asarray(y); base = y.mean()
    if base == 0:
        return float("nan")
    return precision_at_k(y, s, max(1, int(round(frac * len(y))))) / base


def forecast_report(y, s, top_frac=0.05):
    return {"auc": auc(y, s), "ap": ap(y, s), "lift": lift_at_frac(y, s, top_frac),
            "p_at_top": precision_at_k(y, s, max(1, int(round(top_frac * len(y)))))}


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    v = np.asarray([x for x in values if x == x], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_diff_ci(a, b, n_boot=4000, alpha=0.05, seed=0):
    """Mean of (a-b) with a bootstrap CI over paired entries. Returns (mean, lo, hi, p_pos)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2)), float((m > 0).mean())


def discovery_stats(found_per_round, budget_per_round, n_pos):
    """found_per_round: positives found each round. Returns total, hit-rate, normalised AUC of the
    discovery curve vs an oracle that finds min(budget, positives) as early as possible."""
    f = np.asarray(found_per_round, float)
    cum = np.cumsum(f)
    total_budget = float(np.sum(budget_per_round))
    oracle = np.minimum(np.cumsum(budget_per_round), n_pos).astype(float)
    return {"found": float(cum[-1]), "hit_rate": float(cum[-1] / max(total_budget, 1)),
            "frac_of_oracle": float(cum[-1] / max(oracle[-1], 1)),
            "curve_auc": float(cum.sum() / max(oracle.sum(), 1))}
