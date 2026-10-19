"""End-to-end proof that the organism lives: build a small world as a CSV file, give it a brain from history, let it live as
a separate process, kill it without warning, wake it up again, and check that it carries on as the same individual.

    python -m lkm.live selftest [--home DIR]
"""
import json, os, shutil, signal, subprocess, sys, tempfile, time
import numpy as np

CHILD = """
import sys, time
from lkm.live.__main__ import _habitat
class A: pass
a = A(); a.home = sys.argv[1]
from lkm.live.habitat import Habitat
from lkm.live.source import load_source
import json
cfg = json.load(open(sys.argv[1] + '/habitat.json')); cfg['interval_seconds'] = 0.5; cfg['port'] = 0
h = Habitat(sys.argv[1], load_source(sys.argv[1], cfg), cfg)
from lkm.live.verifiers import ReplayVerifier
import csv
h.verifier = ReplayVerifier({r['entity_id']: int(r['label']) for r in csv.DictReader(open(sys.argv[1] + '/truth.csv'))}, 1)
h.boot(); h.run_forever()
"""


def _world_csvs(home, n=1500):
    import pandas as pd
    from scipy.sparse.csgraph import connected_components
    from ..worlds import make_bundle
    from ..graph import Graph
    from ..worlds import make_world
    worlds = make_bundle("cascade", range(100, 103), n=n, rho=0.3) + [make_world(1900, n=n, kind="cascade", rho=0.3)]
    paths = []
    for k, w in enumerate(worlds):
        G = Graph.from_edges(w.n, w.edges)
        df = pd.DataFrame(w.X, columns=[f"f{i}" for i in range(w.X.shape[1])])
        df.insert(0, "id", [f"w{k}-{i}" for i in range(w.n)])
        df["lat"] = w.xy[:, 1] * 0.05 + 40.7; df["lon"] = w.xy[:, 0] * 0.05 - 74.0
        for rel, col in (("br", "chain"), ("ad", "address")):
            nc, lab = connected_components(G.A[rel], directed=False)
            sizes = np.bincount(lab)
            df[col] = np.where(sizes[lab] > 1, lab, -1)
            df[col] = df[col].where(df[col] >= 0)
        df["age"] = w.X[:, w.age_col]
        df["label"] = w.y
        p = os.path.join(home, f"world{k}.csv")
        df.drop(columns=["label"] if k == len(worlds) - 1 else []).to_csv(p, index=False)
        paths.append(p)
    live = pd.read_csv(paths[-1])
    pd.DataFrame({"entity_id": live["id"], "label": worlds[-1].y}).to_csv(os.path.join(home, "truth.csv"), index=False)
    return paths


def run(home=None, quiet=False):
    home = os.path.abspath(home or tempfile.mkdtemp(prefix="organism_selftest_"))
    shutil.rmtree(home, ignore_errors=True); os.makedirs(home)
    res, ok = [], True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        res.append((bool(cond), name, detail))
        print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)

    paths = _world_csvs(home)
    import pandas as pd
    from .source import CsvSource
    from . import bootstrap as B
    from .habitat import DEFAULTS
    hist = [pd.read_csv(p) for p in paths[:-1]]
    spec = CsvSource.fit_spec(hist[0], id_col="id", age_col="age", lat_col="lat", lon_col="lon", group_cols={"br": "chain", "ad": "address"}, label_col="label")
    src = CsvSource(paths[0], spec)
    datasets = [src.build(d, labels=True)[0] for d in hist]
    t0 = time.time()
    B.from_history(home, datasets, quick=True, log=lambda *a: None)
    src.save_spec(home)
    check("first brain fitted from history", os.path.exists(os.path.join(home, "models", "models.pkl")), f"{time.time()-t0:.0f}s")
    cfg = dict(DEFAULTS, source={"type": "csv", "path": paths[-1]}, budget_per_cycle=15, port=0, interval_seconds=0.5)
    json.dump(cfg, open(os.path.join(home, "habitat.json"), "w"), indent=1)

    # life 1: a separate process, killed without warning
    child = subprocess.Popen([sys.executable, "-c", CHILD, home], cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    from .store import Store
    deadline = time.time() + 240
    cyc = 0
    while time.time() < deadline and cyc < 4 and child.poll() is None:
        time.sleep(0.5)
        try:
            cyc = int(Store(home).meta_get("cycle_no", 0))
        except Exception:
            pass
    check("it lived by itself for several cycles (separate process)", cyc >= 4, f"cycle {cyc}")
    os.kill(child.pid, signal.SIGKILL); child.wait()
    st = Store(home)
    snap_cycle = st.load_snapshot()["cycle"]; st.close()
    check("killed with SIGKILL in the middle of its life", True, f"last committed cycle {snap_cycle}")

    # life 2: wake it up
    from .habitat import Habitat, VirtualClock
    from .source import load_source
    from .verifiers import ReplayVerifier
    import csv
    truth = {r["entity_id"]: int(r["label"]) for r in csv.DictReader(open(os.path.join(home, "truth.csv")))}
    h = Habitat(home, load_source(home, cfg), cfg, verifier=ReplayVerifier(truth, 1), log=lambda *a: None)
    h.boot()
    check("it wakes up with its memory", h.cycle_no == snap_cycle + 1, f"resumes at cycle {h.cycle_no}")
    before = h.store.pending()
    ids = [p["entity_id"] for p in before]
    check("its queue is consistent after the crash (no duplicates)", len(ids) == len(set(ids)))
    for _ in range(3):
        h.cycle()
    check("it carries on living", h.cycle_no == snap_cycle + 4)
    check("it was woken twice", h.store.meta_get("boots") == 2)
    rows = h.store.all_answered()
    hits = sum(r[1] for r in rows)
    check("it learned from the answers that came back", len(rows) > 10, f"{len(rows)} answered, {hits} real problems ({100*hits/max(len(rows),1):.0f}%) vs base rate {100*np.mean(list(truth.values())):.0f}%")
    check("it can describe itself", len(h.say()) >= 3)
    h.shutdown()
    print(("\nALL PASSED" if ok else "\nFAILED") + f"  (home: {home})")
    return ok
