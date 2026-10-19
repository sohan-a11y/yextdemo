"""Scenario synthesis: take real windows of THIS machine's normal life and apply a spec's effects to them."""
import numpy as np

from ..features import BOUNDS, FEATURES, to_vector


def apply_effects(raw, effects, rng):
    r = dict(raw)
    for e in effects:
        f, v = e["feature"], e["value"] * float(np.exp(rng.normal(0.0, e["jitter"])))
        base = float(r.get(f, 0.0))
        if e["mode"] == "add":
            base += v
        elif e["mode"] == "multiply":
            base *= max(v, 1.0)
        else:
            base = max(base, v)
        r[f] = float(min(base, BOUNDS[f]))
    return r


def synth(scenario, base_windows, n, rng):
    """n threat (or lookalike) vectors, each built on a different real normal window."""
    idx = rng.integers(0, len(base_windows), n)
    return np.array([to_vector(apply_effects(base_windows[i], scenario["effects"], rng)) for i in idx])


def baseline_stats(base_windows):
    """Per-feature median and 95th percentile of raw counts: the only machine data the optional LLM call may carry."""
    M = np.array([[w.get(f, 0.0) for f in FEATURES] for w in base_windows])
    return {f: {"median": round(float(np.median(M[:, j])), 3), "p95": round(float(np.quantile(M[:, j], 0.95)), 3)} for j, f in enumerate(FEATURES)}


def weaken(scenario, k=0.5):
    """The same threat at a fraction of the strength (adds and floors scaled by k, multipliers pulled towards 1)."""
    eff = []
    for e in scenario["effects"]:
        v = 1.0 + (e["value"] - 1.0) * k if e["mode"] == "multiply" else e["value"] * k
        eff.append({**e, "value": v})
    return {**scenario, "effects": eff}
