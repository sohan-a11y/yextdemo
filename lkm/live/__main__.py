"""Command line of the living organism.  python -m lkm.live <command> --home DIR

  init       write the habitat configuration            bootstrap  give the organism its first brain
  run        let it live (boots, cycles, sleeps, until stopped)
  status     who it is and what it did                  queue      the checks it is waiting on
  answer     tell it what really happened               say        it describes itself
  grade      score its past proposals against the truth (prospective test)
  selftest   build a tiny world, live, kill and wake it, and verify continuity
"""
import argparse, csv, json, os, sys, time


def _home(a):
    return os.path.abspath(a.home)


def _cfg(home):
    return json.load(open(os.path.join(home, "habitat.json")))


def cmd_init(a):
    from .habitat import DEFAULTS
    os.makedirs(_home(a), exist_ok=True)
    cfg = dict(DEFAULTS)
    cfg["source"] = {"type": a.source, **({"path": a.csv} if a.csv else {}), **({"pool": a.pool} if a.source == "overture" else {})}
    for k in ("interval_seconds", "port", "webhook_url", "budget_per_cycle"):
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
    json.dump(cfg, open(os.path.join(_home(a), "habitat.json"), "w"), indent=1)
    print("wrote", os.path.join(_home(a), "habitat.json"))


def cmd_bootstrap(a):
    from . import bootstrap as B
    home = _home(a)
    if a.models:
        B.from_models(home, a.models); print("models installed")
        return
    import pandas as pd
    from .source import CsvSource
    cfgd = _cfg(home)
    dfs = [pd.read_csv(p) for p in a.history]
    spec = CsvSource.fit_spec(dfs[0], id_col=a.id_col, age_col=a.age_col, lat_col=a.lat_col, lon_col=a.lon_col,
                              group_cols=dict(kv.split("=") for kv in (a.group or [])), text_col=a.text_col, url_col=a.url_col, label_col=a.label_col)
    src = CsvSource(a.history[0], spec)
    datasets = [src.build(d, labels=True)[0] for d in dfs]
    B.from_history(home, datasets, quick=a.quick)
    src.spec = spec; src.save_spec(home)
    print("organism has its first brain; spec saved")


def _habitat(a, verifier=None, clock=None, sleep=None):
    from .habitat import Habitat
    from .source import load_source
    home = _home(a); cfg = _cfg(home)
    kw = {}
    if clock:
        kw = {"clock": clock.now, "sleep": clock.advance}
    return Habitat(home, load_source(home, cfg), cfg, verifier=verifier, **kw)


def cmd_run(a):
    from .habitat import VirtualClock
    from .verifiers import ReplayVerifier, HttpLivenessVerifier
    home = _home(a); cfg = _cfg(home)
    clock = VirtualClock() if a.virtual else None
    if clock is not None:                                 # a simulated calendar must continue where the last life stopped
        from .store import Store
        st = Store(home); last = st.meta_get("last_cycle_ts"); st.close()
        if last:
            clock.t = last + cfg.get("interval_seconds", 86400.0)
    h = _habitat(a, clock=clock)
    if a.replay:
        h.verifier = ReplayVerifier(h.source.truth(), delay_cycles=a.replay_delay)
    elif cfg.get("verifier") == "http" or a.http_verifier:
        h.verifier = HttpLivenessVerifier(h.source.url_of)
    if a.port is not None:
        h.cfg["port"] = a.port
    if a.interval is not None:
        h.cfg["interval_seconds"] = a.interval
    h.boot()
    h.run_forever(max_cycles=a.cycles)


def _store(a):
    from .store import Store
    return Store(_home(a))


def cmd_status(a):
    from .habitat import Habitat
    h = Habitat(_home(a), None); h.store = _store(a); h.boot_ts = None
    h.cycle_no = int(h.store.meta_get("cycle_no", 0))
    print(json.dumps(h.status(), indent=1, default=str))


def cmd_queue(a):
    for p in _store(a).pending(a.n):
        print(f"{p['entity_id']}\trisk {100*(p['risk'] or 0):.0f}%\t{p['why']}")


def cmd_answer(a):
    st = _store(a)
    if a.csv:
        with open(a.csv, newline="") as f:
            for r in csv.DictReader(f):
                print(r["entity_id"], st.answer(str(r["entity_id"]), int(r["label"]), "cli"))
    else:
        print(st.answer(str(a.id), int(a.label), "cli"))


def cmd_say(a):
    from .habitat import Habitat
    h = Habitat(_home(a), None); h.store = _store(a)
    h.clock = lambda: (h.store.meta_get("last_cycle_ts") or time.time())
    h.cycle_no = int(h.store.meta_get("cycle_no", 0))
    print("\n".join(h.say()))


def cmd_grade(a):
    from .habitat import Habitat
    h = Habitat(_home(a), None); h.store = _store(a)
    truth = {}
    with open(a.truth, newline="") as f:
        for r in csv.DictReader(f):
            truth[str(r["entity_id"])] = int(r["label"])
    print(json.dumps(h.grade(truth), indent=1))


def cmd_selftest(a):
    from .selftest import run
    sys.exit(0 if run(a.home) else 1)


def main():
    ap = argparse.ArgumentParser(prog="lkm.live")
    sp = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        p = sp.add_parser(name, **kw); p.add_argument("--home", required=(name != "selftest"), default="./organism_selftest"); p.set_defaults(fn=fn); return p
    p = add("init", cmd_init); p.add_argument("--source", choices=["csv", "overture", "custom"], default="csv"); p.add_argument("--csv")
    p.add_argument("--pool", type=int, default=0); p.add_argument("--interval-seconds", dest="interval_seconds", type=float)
    p.add_argument("--port", type=int); p.add_argument("--webhook-url", dest="webhook_url"); p.add_argument("--budget-per-cycle", dest="budget_per_cycle", type=int)
    p = add("bootstrap", cmd_bootstrap); p.add_argument("--models"); p.add_argument("--history", nargs="+")
    p.add_argument("--id-col", default="id"); p.add_argument("--age-col", default="age"); p.add_argument("--label-col", default="label")
    p.add_argument("--lat-col"); p.add_argument("--lon-col"); p.add_argument("--text-col"); p.add_argument("--url-col")
    p.add_argument("--group", nargs="*", help="relation=column, e.g. br=chain_id ad=address_key"); p.add_argument("--quick", action="store_true")
    p = add("run", cmd_run); p.add_argument("--cycles", type=int); p.add_argument("--virtual", action="store_true", help="one cycle = one simulated day, no waiting")
    p.add_argument("--replay", action="store_true", help="answer from known outcomes (SIMULATION of the outside world)"); p.add_argument("--replay-delay", type=int, default=1)
    p.add_argument("--http-verifier", action="store_true"); p.add_argument("--port", type=int); p.add_argument("--interval", type=float)
    add("status", cmd_status)
    p = add("queue", cmd_queue); p.add_argument("-n", type=int, default=50)
    p = add("answer", cmd_answer); p.add_argument("--csv"); p.add_argument("--id"); p.add_argument("--label", type=int)
    add("say", cmd_say)
    p = add("grade", cmd_grade); p.add_argument("--truth", required=True)
    add("selftest", cmd_selftest)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
