"""Command line of the Neuron Forge.   python -m sentinel.forge <command> --home DIR

  design   ask a teacher (mock or Claude) for a spec from your purpose; saves it for you to read and edit; trains nothing
  create   purpose (or a spec file) -> validated spec -> synthetic threats on THIS machine's normal history -> trained neuron -> gate -> stored as a proposal
  list / show ID          see the neurons and the full report on one
  approve ID --by NAME    a person lets a proposal into the live brain      reject / disable / enable ID --by NAME
Nothing is ever deleted: a neuron you no longer want is disabled and stays on record. The running Sentinel picks up changes without a restart.
"""
import argparse
import json
import os
import sys
import time

from ..constitution import own_path
from ..store import SentinelStore
from . import spec as S
from .generate import baseline_stats
from .llm import MockTeacher, ClaudeTeacher, TeacherError
from .train import forge, GATE
from .demo import synthetic_baseline


def make_teacher(a):
    if a.teacher == "mock":
        return MockTeacher()
    return ClaudeTeacher(model=a.model, effort=a.effort, fallbacks=a.fallbacks, base_url=a.base_url)


def get_baseline(store, a):
    if getattr(a, "synthetic_baseline", None):
        return synthetic_baseline(a.synthetic_baseline, a.seed), "synthetic"
    return store.baseline_windows(), "machine"


def design_spec(store, a):
    if a.spec:
        try:
            raw = json.load(open(a.spec))
        except (OSError, json.JSONDecodeError) as ex:
            raise SystemExit(f"cannot read the spec file {a.spec}: {ex}")
        who = "file"
    else:
        base, kind = get_baseline(store, a)
        stats = baseline_stats(base) if (a.share_baseline_stats and len(base) >= 100) else None
        raw = make_teacher(a).design(a.purpose, stats)
        who = a.teacher
    return S.validate(raw), who


def report(n, m=None):
    m = m or n["metrics"]
    sp = n["spec"]
    L = [f"#{n['id']} {n['name']} v{n['version']}  [{n['state']}]  teacher: {n['teacher']}", f"purpose: {n['purpose']}", f"summary: {sp.get('summary', '')}", ""]
    L.append(f"history used: {m['n_baseline']} windows ({m['baseline_kind']}); train {m['n_train']} / calibrate {m['n_calibration']} / test {m['n_test']}; {m['n_params']} numbers")
    L.append(f"alarm line {m['threshold']:.2f}; false alarms on {m['n_test']} untouched normal windows: {m['test_false_alarms']} (per-window rate at most {100*m['test_fpr_upper_bound']:.2f}%, 95% confidence)")
    L.append("caught on freshly built SYNTHETIC threats: " + ", ".join(f"{k} {100*v:.0f}%" for k, v in m["recall_fresh"].items()))
    L.append("same threats at half strength:               " + ", ".join(f"{k} {100*v:.0f}%" for k, v in m["recall_half_strength"].items()))
    if m["recall_leave_one_scenario_out"]:
        L.append("threat NOT shown during training (leave-one-out): " + ", ".join(f"{k} {100*v:.0f}%" for k, v in m["recall_leave_one_scenario_out"].items()))
    L.append("harmless lookalikes it fires on: " + ", ".join(f"{k} {100*v:.0f}%" for k, v in m["lookalike_fire_rate"].items()))
    rr = m.get("reflex_reference")
    if rr:
        L.append(f"for comparison the always-on reflex alone catches {100*rr['mean_recall']:.0f}% of these threats (its false-alarm rate on the test windows: {100*rr['false_alarm_rate_on_validation']:.2f}%)")
    L.append("")
    L.append("gate: " + ("PASSED" if m["gate"]["passed"] else "NOT PASSED"))
    for c in m["gate"]["checks"]:
        v, need = c["value"], c["need"]
        L.append(f"  [{'ok' if c['ok'] else 'NO'}] {c['check']}: {v:.4g}" + (f" (need {need:.4g})" if not isinstance(need, str) else f" (need {need})") if isinstance(v, (int, float)) else
                 f"  [{'ok' if c['ok'] else 'NO'}] {c['check']}: {v} (need {need})")
    L.append("")
    L.append("Remember: the threats are synthetic. This says the neuron is quiet on your normal days and reacts to the described changes; it does not prove it catches real attacks.")
    a = sp.get("alert", {})
    L.append(f"when it fires the alert will read: [{a.get('severity')}] {a.get('title')} - {a.get('explanation')}")
    return "\n".join(L)


def create(store, spec, purpose, teacher, baseline, kind, seed=0, gate=None):
    neuron, m = forge(spec, baseline, baseline_kind=kind, seed=seed, gate=gate)
    state = "proposed" if m["gate"]["passed"] else ("demo" if kind != "machine" else "failed")
    nid, ver = store.add_neuron(spec["name"], purpose, spec, neuron.to_blob(), m, teacher, {"kind": kind, "n": len(baseline)}, state)
    return nid, ver, state, m


def approve(store, nid, by):
    n = next((x for x in store.neurons() if x["id"] == nid), None)
    if n is None:
        raise SystemExit(f"no neuron #{nid}")
    if n["state"] != "proposed":
        raise SystemExit(f"neuron #{nid} is {n['state']!r}; only a proposal that passed the gate on this machine's real history can be approved")
    if not by or not by.strip():
        raise SystemExit("say who is approving: --by NAME")
    store.set_neuron_state(nid, "active", by.strip())


def _store(a):
    return SentinelStore(os.path.abspath(a.home))


def cmd_design(a):
    st = _store(a)
    spec, who = design_spec(st, a)
    path = own_path(os.path.abspath(a.home), "specs", spec["name"] + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(spec, open(path, "w"), indent=1)
    print(json.dumps(spec, indent=1))
    print(f"\nsaved to {path}. Read it, edit it if you like, then:  python -m sentinel.forge create --home {a.home} --spec {path} --purpose \"...\"")


def cmd_create(a):
    st = _store(a)
    spec, who = design_spec(st, a)
    base, kind = get_baseline(st, a)
    if kind == "machine" and len(base) < GATE["min_baseline"]:
        print(f"note: only {len(base)} normal windows recorded so far (the gate wants {GATE['min_baseline']}); the neuron will be trained but cannot pass yet.")
    if not base:
        sys.exit("no normal history yet: let the Sentinel run first (or use --synthetic-baseline N for a demonstration neuron that can never be approved)")
    nid, ver, state, m = create(st, spec, a.purpose or spec["summary"], who, base, kind, a.seed)
    n = next(x for x in st.neurons() if x["id"] == nid)
    print(report(n))
    print(f"\nstored as #{nid} v{ver} [{state}]." + (f" To let it into the live brain:  python -m sentinel.forge approve {nid} --home {a.home} --by YOURNAME" if state == "proposed" else ""))


def cmd_list(a):
    for n in _store(a).neurons():
        m = n["metrics"]
        print(f"#{n['id']:<3} {n['name']:<30} v{n['version']} {n['state']:<9} recall {100*m.get('mean_recall', 0):.0f}%  gate {'ok' if m.get('gate', {}).get('passed') else 'no'}  by {n['decided_by'] or '-'}")


def cmd_show(a):
    n = next((x for x in _store(a).neurons() if x["id"] == a.id), None)
    print(report(n) if n else "no such neuron")


def _decide(state_to, allowed_from):
    def fn(a):
        st = _store(a)
        n = next((x for x in st.neurons() if x["id"] == a.id), None)
        if n is None:
            sys.exit("no such neuron")
        if n["state"] not in allowed_from:
            sys.exit(f"#{a.id} is {n['state']!r}; this needs one of {allowed_from}")
        if state_to == "active" and n["decided_by"] is None:
            sys.exit("this neuron was never approved; use approve")
        st.set_neuron_state(a.id, state_to, a.by or "unknown")
        print(f"#{a.id} is now {state_to}")
    return fn


def cmd_approve(a):
    approve(_store(a), a.id, a.by)
    print(f"#{a.id} is now active; the running Sentinel loads it at its next window.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sentinel.forge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("design", cmd_design), ("create", cmd_create), ("list", cmd_list), ("show", cmd_show), ("approve", cmd_approve),
                     ("reject", _decide("rejected", ("proposed", "failed", "demo"))), ("disable", _decide("disabled", ("active",))),
                     ("enable", _decide("active", ("disabled",)))):
        p = sub.add_parser(name)
        p.add_argument("--home", default="sentinel_home")
        p.set_defaults(fn=fn)
        if name in ("design", "create"):
            p.add_argument("--purpose"); p.add_argument("--spec"); p.add_argument("--teacher", choices=["mock", "claude"], default="mock")
            p.add_argument("--model", default="claude-opus-5-5"); p.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
            p.add_argument("--fallbacks", action="store_true", help="let the API retry a declined request on another model")
            p.add_argument("--base-url", dest="base_url"); p.add_argument("--share-baseline-stats", action="store_true",
                                                                        help="send per-feature median/95th percentile counts of this machine to the model (no names, paths or addresses)")
            p.add_argument("--synthetic-baseline", type=int, help="demonstration only: a made-up normal history of N windows; the neuron cannot be approved")
            p.add_argument("--seed", type=int, default=0)
        if name in ("show", "approve", "reject", "disable", "enable"):
            p.add_argument("id", type=int)
        if name in ("approve", "reject", "disable", "enable"):
            p.add_argument("--by", required=True)
    a = ap.parse_args(argv)
    if a.cmd in ("design", "create") and not (a.purpose or a.spec):
        ap.error("give --purpose or --spec")
    try:
        a.fn(a)
    except TeacherError as ex:
        sys.exit(f"teacher: {ex}")
    except S.SpecError as ex:
        sys.exit(f"the spec is not acceptable: {ex}")


if __name__ == "__main__":
    main()
