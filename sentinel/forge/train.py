"""Train a candidate neuron from a validated spec and the machine's own normal windows, then test it honestly.

Training data: the machine's normal windows (label 0), the spec's harmless lookalikes applied to normal windows (label 0), and the spec's threat
scenarios applied to normal windows (label 1). The threat windows are SYNTHETIC. Therefore the report always carries:
  * false alarms are measured on real, later, untouched normal windows (time-ordered split),
  * recall is measured on freshly drawn threat windows built on those later windows,
  * leave-one-scenario-out: a copy trained WITHOUT one scenario is asked to catch it (does the neuron generalise to a threat it was not shown?),
  * the always-on reflex is scored on the same threats, so a neuron that adds nothing over it is visible.
Nothing here says the neuron catches real attacks; that can only be learned from real alerts and your feedback.
"""
import numpy as np

from ..features import to_vector
from ..neurons import ForgeNeuron, TinyMLP, KitEnsemble, SD_FLOOR
from .generate import synth, weaken

GATE = {"min_test_windows": 960, "max_fpr_ub": 0.004, "min_recall": 0.90, "min_scenario_recall": 0.50, "max_lookalike_fire": 0.20, "min_baseline": 3800}
# 960 untouched normal windows with no false alarm bound the per-window false-alarm rate at 0.4% (95% confidence): about 1.5 days at 30 s per window.

def wilson_ub(k, n, z=1.96):
    if n == 0:
        return 1.0
    p = k / n
    return float((p + z * z / (2 * n) + z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n))


def _fit(Zb, Zl, Zp_by_group, rng_seed, epochs):
    parts = [Zb] + ([Zl] if len(Zl) else []) + list(Zp_by_group)
    X = np.vstack(parts)
    y = np.concatenate([np.zeros(len(Zb)), np.zeros(len(Zl))] + [np.ones(len(z)) for z in Zp_by_group])
    w = np.concatenate([np.full(len(Zb), 0.4 / len(Zb))] + ([np.full(len(Zl), 0.2 / len(Zl))] if len(Zl) else []) +
                       [np.full(len(z), 0.4 / len(Zp_by_group) / len(z)) for z in Zp_by_group])
    return TinyMLP(X.shape[1], 16, seed=rng_seed).fit(X, y, w, epochs=epochs)


def _threshold(mlp, Zcal):
    """Alarm line = just above the highest score any CALIBRATION window got (never below 0.5). The test windows are not used for this."""
    return float(np.clip(mlp.predict(Zcal).max() + 0.01, 0.5, 0.99))


def reflex_reference(Vtr, Vval, fresh):
    """Recall of the always-on reflex (trained by replaying the normal training windows) on the same fresh threats."""
    n = len(Vtr)
    if n < 300:
        return None
    k = KitEnsemble(fm=n // 4, ad=n // 2, cal=n - n // 4 - n // 2, seed=0)
    for v in Vtr:
        k.process(v)
    if k.phase != "armed":
        return None
    fa = float(np.mean([k.process(v, learn=False)[0] > k.thr for v in Vval]))
    rec = {name: float(np.mean([k.process(v, learn=False)[0] > k.thr for v in V])) for name, V in fresh.items()}
    return {"false_alarm_rate_on_validation": fa, "recall": rec, "mean_recall": float(np.mean(list(rec.values())))}


def forge(spec, baseline, baseline_kind="machine", seed=0, n_pos=400, epochs=500, gate=None, compare_reflex=True):
    """-> (ForgeNeuron, metrics). `baseline` is a list of raw window dicts in time order.
    Time-ordered split: 60% train, 15% calibration (picks the alarm line), 25% test (touched once, measures false alarms and recall)."""
    g = {**GATE, **(gate or {})}
    rng = np.random.default_rng(seed)
    n = len(baseline)
    c1, c2 = int(0.6 * n), int(0.75 * n)
    tr_raw, cal_raw, te_raw = baseline[:c1], baseline[c1:c2], baseline[c2:]
    Vtr, Vcal, Vte = (np.array([to_vector(w) for w in part]) for part in (tr_raw, cal_raw, te_raw))
    mu = Vtr.mean(0)
    sd = np.sqrt(Vtr.var(0) + SD_FLOOR ** 2)
    Z = lambda V: np.clip((V - mu) / sd, -8.0, 8.0)
    Ztr, Zcal, Zte = Z(Vtr), Z(Vcal), Z(Vte)
    Zl = np.vstack([Z(synth(s, tr_raw, n_pos // 2, rng)) for s in spec["lookalikes"]])
    pos_tr = {s["name"]: Z(synth(s, tr_raw, n_pos, rng)) for s in spec["scenarios"]}
    fresh_pos = {s["name"]: Z(synth(s, te_raw, 300, rng)) for s in spec["scenarios"]}
    half_pos = {s["name"]: Z(synth(weaken(s), te_raw, 300, rng)) for s in spec["scenarios"]}
    fresh_look = {s["name"]: Z(synth(s, te_raw, 300, rng)) for s in spec["lookalikes"]}

    mlp = _fit(Ztr, Zl, list(pos_tr.values()), seed, epochs)
    thr = _threshold(mlp, Zcal)
    fires = int((mlp.predict(Zte) >= thr).sum())
    recall = {k: float((mlp.predict(z) >= thr).mean()) for k, z in fresh_pos.items()}
    half = {k: float((mlp.predict(z) >= thr).mean()) for k, z in half_pos.items()}
    look = {k: float((mlp.predict(z) >= thr).mean()) for k, z in fresh_look.items()}
    loso = {}
    if len(spec["scenarios"]) >= 2:
        for s in spec["scenarios"]:
            rest = [z for k, z in pos_tr.items() if k != s["name"]]
            m2 = _fit(Ztr, Zl, rest, seed + 1, epochs)
            loso[s["name"]] = float((m2.predict(fresh_pos[s["name"]]) >= _threshold(m2, Zcal)).mean())

    m = {"baseline_kind": baseline_kind, "n_baseline": n, "n_train": len(Vtr), "n_calibration": len(Vcal), "n_test": len(Vte), "threshold": thr, "n_params": mlp.n_params,
         "test_false_alarms": fires, "test_fpr_upper_bound": wilson_ub(fires, len(Vte)), "recall_fresh": recall, "mean_recall": float(np.mean(list(recall.values()))),
         "recall_half_strength": half, "recall_leave_one_scenario_out": loso, "lookalike_fire_rate": look, "synthetic_threats_only": True}
    if compare_reflex:
        fresh_vec = {k: z * sd + mu for k, z in fresh_pos.items()}
        m["reflex_reference"] = reflex_reference(np.vstack([Vtr, Vcal]), Vte, fresh_vec)
    checks = [
        ("untouched later normal windows to test false alarms", len(Vte), g["min_test_windows"], len(Vte) >= g["min_test_windows"]),
        ("false-alarm rate per window, 95% upper bound", m["test_fpr_upper_bound"], g["max_fpr_ub"], m["test_fpr_upper_bound"] <= g["max_fpr_ub"]),
        ("mean recall on fresh threat windows", m["mean_recall"], g["min_recall"], m["mean_recall"] >= g["min_recall"]),
        ("weakest scenario's recall", min(recall.values()), g["min_scenario_recall"], min(recall.values()) >= g["min_scenario_recall"]),
        ("worst lookalike wrongly fired on", max(look.values()), g["max_lookalike_fire"], max(look.values()) <= g["max_lookalike_fire"]),
        ("baseline is this machine's real history", baseline_kind, "machine", baseline_kind == "machine"),
        ("enough history in total", n, g["min_baseline"], n >= g["min_baseline"]),
    ]
    m["gate"] = {"passed": all(c[3] for c in checks), "checks": [{"check": a, "value": b, "need": c, "ok": bool(d)} for a, b, c, d in checks]}
    return ForgeNeuron(spec["name"], mlp, mu, sd, thr, spec), m
