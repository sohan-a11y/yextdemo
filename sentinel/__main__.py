"""Command line of the Sentinel.   python -m sentinel <command> --home DIR

  init      write the configuration       run       let it watch (boots, watches, sleeps, until stopped)
  status    what it is doing              alerts    recent alerts           requests   things it asks you to do (it will not do them)
  pause / resume   the kill switch (answering an alert 'real' / 'fine' is done on the dashboard)
  The Neuron Forge (making new neurons from a purpose) is a separate tool:  python -m sentinel.forge --help
"""
import argparse
import json
import os
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sentinel", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("init", cmd_init), ("run", cmd_run), ("status", cmd_status), ("alerts", cmd_alerts), ("requests", cmd_requests),
                     ("pause", cmd_pause), ("resume", cmd_resume)):
        p = sub.add_parser(name)
        p.add_argument("--home", default="sentinel_home")
        p.set_defaults(fn=fn)
        if name == "init":
            p.add_argument("--window_s", type=float); p.add_argument("--poll_s", type=float); p.add_argument("--port", type=int); p.add_argument("--host")
            p.add_argument("--watch", nargs="*", help="folders to watch (default: Downloads, Desktop, Documents, Temp, Startup)")
        if name == "run":
            p.add_argument("--max_steps", type=int)
        if name == "alerts":
            p.add_argument("--status"); p.add_argument("-n", type=int, default=20)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
