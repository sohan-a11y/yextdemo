"""Command line of the Sentinel.   python -m sentinel <command> --home DIR

  init      write the configuration       run       let it watch (boots, watches, sleeps, until stopped)
  status    what it is doing              alerts    recent alerts           requests   things it asks you to do (it will not do them)
  pause / resume   the kill switch (answering an alert 'real' / 'fine' is done on the dashboard)
  grow      make a new neuron from one sentence, in milliseconds:  grow --name NAME "what to watch for" [--example "..."]
  concepts  list the neurons you grew         concept-off / concept-on ID     switch one off or on (nothing is deleted)
  fetch-model   install the language model the cortex borrows (run once, by you; the Sentinel itself never downloads)
  The Neuron Forge (making new neurons from a purpose) is a separate tool:  python -m sentinel.forge --help
"""
import argparse
import json
import os
import re
import sys

from .constitution import own_path
from .service import DEFAULTS, Sentinel


def _home(a):
    return os.path.abspath(a.home)


def _cfg(home):
    p = os.path.join(home, "sentinel.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def build_sensors(cfg, sniff=None):
    from .sensors.processes import ProcessSensor
    from .sensors.connections import ConnectionSensor
    from .sensors.files import FileSensor
    from .sensors.lan import LanSensor
    from .sensors.netflow import NetflowSensor
    return [ProcessSensor(), ConnectionSensor(), FileSensor(cfg.get("watch_dirs")), LanSensor(), NetflowSensor(sniff=cfg.get("dns_sniff", False) if sniff is None else sniff)]


def cmd_init(a):
    home = _home(a)
    os.makedirs(home, exist_ok=True)
    cfg = {k: DEFAULTS[k] for k in ("window_s", "poll_s", "host", "port", "max_alerts_per_hour", "watch_dirs", "dns_sniff", "reflex")}
    for k in ("window_s", "poll_s", "port", "host"):
        if getattr(a, k, None) is not None:
            cfg[k] = getattr(a, k)
    if a.watch:
        cfg["watch_dirs"] = a.watch
    json.dump(cfg, open(own_path(home, "sentinel.json"), "w"), indent=1)
    print("wrote", os.path.join(home, "sentinel.json"))


def cmd_run(a):
    home = _home(a)
    cfg = _cfg(home)
    s = Sentinel(home, build_sensors(cfg), cfg).boot()
    s.run_forever(max_steps=a.max_steps)


def _db(a):
    from .store import SentinelStore
    return SentinelStore(_home(a))


def cmd_status(a):
    st = _db(a)
    print(json.dumps({"windows": st.n_windows(), "alerts": st.alert_counts(), "neurons": [(n["name"], n["version"], n["state"]) for n in st.neurons()],
                      "paused_flag": st.meta_get("paused", False), "boots": st.meta_get("boots", 0)}, indent=1))


def cmd_alerts(a):
    for x in _db(a).alerts(a.status, a.n):
        print(f"#{x['id']} [{x['severity']}] {x['status']:6s} {x['source']:14s} {x['title']}\n      {x['why']}")


def cmd_requests(a):
    for x in _db(a).requests("open", 50):
        print(f"#{x['id']} {x['kind']}: {x['suggestion']}")


def cmd_pause(a):
    _db(a).meta_set("paused", True); print("paused (a running Sentinel notices at its next step; creating a file named PAUSE in the home folder does the same)")


def cmd_resume(a):
    _db(a).meta_set("paused", False); print("resumed")


NAME_OK = re.compile(r"^[a-z][a-z0-9_]{2,39}$")


def cmd_grow(a):
    import time
    import numpy as np
    from .cortex import Cortex, make_neuron
    if not NAME_OK.match(a.name):
        sys.exit("name: lowercase letters, digits and underscores, 3-40 characters")
    if not (3 <= len(a.description) <= 300):
        sys.exit("description: 3-300 characters")
    st = _db(a)
    t0 = time.perf_counter()
    cx = Cortex(a.model_dir)
    t_load = time.perf_counter() - t0
    ids, V, sents, ts = st.recent_vectors(20000)
    per_day = None
    if len(ts) > 1 and max(ts) > min(ts):
        per_day = len(V) / max((max(ts) - min(ts)) / 86400.0, 1.0 / 24)
    if a.alpha:
        alpha = a.alpha
    elif per_day and len(V) >= 1000:
        alpha = float(min(0.01, max(a.per_day / per_day, 1.0 / len(V))))
    else:
        alpha = 0.001
    t1 = time.perf_counter()
    n = make_neuron(cx, a.name, a.description, a.example or [], V if len(V) >= 1000 else None, alpha)
    t_make = time.perf_counter() - t1
    print(f"neuron '{a.name}' made in {1000 * t_make:.0f} ms ({n.n_params} numbers; loading the language model took {t_load:.1f} s, once per run)")
    if n.thr is None:
        print(f"only {len(V)} events in memory so far: the alarm line will be set automatically once the Sentinel has read 1000 events")
    else:
        est = f", about {alpha * per_day:.1f} alerts per day on your normal activity" if per_day else ""
        print(f"alarm line {n.thr:.3f}: fires on {100 * alpha:.2f}% of the {len(V)} recent events it was checked against{est}")
    if len(V):
        sc = n.score_many(V)
        print("closest recent events (would it fire?):")
        for i in np.argsort(-sc)[:a.show]:
            mark = "FIRE" if (n.thr is not None and sc[i] > n.thr) else "    "
            print(f"  {mark} {sc[i]:.3f}  {sents[i][:110]}")
    if a.dry_run:
        print("dry run: nothing saved")
        return
    cid = st.add_concept(a.name, a.description, n.to_dict(), a.severity, a.by)
    print(f"saved as concept #{cid}, active now; a running Sentinel picks it up at its next window. Switch off with: concept-off {cid}")


def cmd_concepts(a):
    for c in _db(a).concepts():
        d = c["data"]
        thr = "not yet calibrated" if d.get("thr") is None else f"line {d['thr']:.3f}"
        print(f"#{c['id']:<3} {c['name']:<28} {c['state']:<8} {thr:<20} examples {len(d['examples'])} counter {len(d['counter'])} answers {d.get('n_feedback', 0)}  - {c['description']}")


def _concept_state(state):
    def fn(a):
        st = _db(a)
        if not any(c["id"] == a.id for c in st.concepts()):
            sys.exit("no such concept neuron")
        st.set_concept_state(a.id, state, a.by)
        print(f"#{a.id} is now {state}")
    return fn


def cmd_fetch_model(a):
    from lkm.language import ensure_model
    from .cortex import default_model_dir
    d = a.model_dir or default_model_dir()
    print("installed in", ensure_model(d))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sentinel", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("init", cmd_init), ("run", cmd_run), ("status", cmd_status), ("alerts", cmd_alerts), ("requests", cmd_requests),
                     ("pause", cmd_pause), ("resume", cmd_resume),
                     ("grow", cmd_grow), ("concepts", cmd_concepts), ("concept-off", _concept_state("disabled")), ("concept-on", _concept_state("active")),
                     ("fetch-model", cmd_fetch_model)):
        p = sub.add_parser(name)
        p.add_argument("--home", default="sentinel_home")
        p.set_defaults(fn=fn)
        if name == "init":
            p.add_argument("--window_s", type=float); p.add_argument("--poll_s", type=float); p.add_argument("--port", type=int); p.add_argument("--host")
            p.add_argument("--watch", nargs="*", help="folders to watch (default: Downloads, Desktop, Documents, Temp, Startup)")
        if name == "run":
            p.add_argument("--max_steps", type=int)
        if name == "grow":
            p.add_argument("description"); p.add_argument("--name", required=True); p.add_argument("--example", action="append")
            p.add_argument("--severity", default="medium", choices=["info", "low", "medium", "high"]); p.add_argument("--alpha", type=float)
            p.add_argument("--per-day", dest="per_day", type=float, default=1.0, help="target false alarms per day on your normal activity")
            p.add_argument("--dry-run", action="store_true"); p.add_argument("--show", type=int, default=10); p.add_argument("--by", default="owner")
        if name in ("grow", "fetch-model"):
            p.add_argument("--model-dir", dest="model_dir")
        if name in ("concept-off", "concept-on"):
            p.add_argument("id", type=int); p.add_argument("--by", default="owner")
        if name == "alerts":
            p.add_argument("--status"); p.add_argument("-n", type=int, default=20)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
