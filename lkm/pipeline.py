"""Experiment pipeline: fit everything for one 'family' (a world type or the real data), then evaluate.

Experiments
  E1  one-shot forecasting of which entities need attention (AUC, AP, lift)
  E2  active verification game (the operational problem): positives found per verification budget
  E3  regime shift: models trained on one world, deployed on a shifted one (adaptation)
  E4  missing data: random feature entries missing at test time
  E5  efficiency: wall-clock per game / per forward pass
  E6  sim-to-real: workers evolved on synthetic worlds, deployed on real data
  E7  life statistics: births, deaths, energy, jump rate, behaviour of evolved genomes
"""
import os, time, pickle, json
import numpy as np
import multiprocessing as mp
from . import nca as N
from .baselines import AgeRule, LogReg, GBDT, oof_scores
from .game import run_game, RandomPolicy, RankPolicy, OnlineGBDT, CellsOnly
from .organism import Organism
from .metrics import forecast_report, bootstrap_ci, paired_diff_ci

CFG_FULL = dict(n_train=8, n_val=6, n_test=20, world_n=5000, nca_steps=600, es_gens=40, es_gens_ablation=30, es_pairs=10, es_eps=4, es_proc=4,
                seeds=3, budget_frac=0.03, rounds=8, rhos=(0.1, 0.25, 0.5, 1.0), real_nca_steps=500, real_seeds=3)
CFG_FINAL = dict(CFG_FULL, vocab=16, comm="lex", lang_real=True, lang_edges=True)
CFG_SMOKE = dict(n_train=3, n_val=3, n_test=2, world_n=1500, nca_steps=40, es_gens=2, es_pairs=3, es_eps=2, es_proc=2,
                 seeds=1, budget_frac=0.03, rounds=6, rhos=(0.25, 1.0), real_nca_steps=30, real_seeds=1)

VARIANTS = {"full": {}, "no_scent": {"use_scent": False}, "no_plastic": {"plastic": False}, "workers_only": {"use_cells": False},
            "blind": {"informed": False}, "no_words": {}}


def variant_flags(key, vocab):
    """Organism switches for a variant. Workers hear the cell field's words unless the variant is 'no_words'."""
    f = dict(VARIANTS[key]) if key in VARIANTS else {}
    if vocab and key not in ("no_words", "workers_only"):
        f["vocab"] = vocab
    return f


def log0(*a):
    print(*a, flush=True)


class Models:
    pass


# ------------------------------------------------------------------------------------------ fitting
def fit_models(tag, train, es_sets, cache_dir, cfg, nca_steps, variants=("full", "no_scent", "no_plastic", "workers_only", "blind"),
               log=log0, val_for_nca=None, vocab=0, comm="cont"):
    from . import nca_train as NT
    from .es import train_es
    path = os.path.join(cache_dir, f"models_{tag}.pkl")
    M = pickle.load(open(path, "rb")) if os.path.exists(path) else Models()
    done = getattr(M, "stage", set())
    t0 = time.time()
    if "base" not in done:
        log(f"[{tag}] fitting baselines on {len(train)} datasets, {sum(d.n for d in train)} nodes")
        M.age = AgeRule(); M.logreg = LogReg().fit(train)
        M.node = GBDT("node").fit(train); M.graph = GBDT("graph").fit(train); M.rev = GBDT("rev").fit(train)
        M.rev_g = GBDT("rev", greedy_aug=True, name="gbdt_rev_greedy").fit(train)
        M.graph_aug = GBDT("graph", mask_aug=True, name="gbdt_graph_aug").fit(train)
        done.add("base")
    if "nca" not in done:
        log(f"[{tag}] OOF base logits + NCA training")
        oof = oof_scores(train, lambda: GBDT("graph"), folds=4)
        vds = val_for_nca if val_for_nca is not None else es_sets[0]
        M.P, M.nca_info = NT.train(train, oof, val=vds, val_base=M.graph.score(vds), steps=nca_steps, log=log, V=vocab, comm=comm)
        M.vocab, M.comm = vocab, comm
        done.add("nca")
    M.thetas = getattr(M, "thetas", {}); M.es_info = getattr(M, "es_info", {})
    for v in variants:
        if v in M.thetas:
            continue
        log(f"[{tag}] evolving workers: variant={v}")
        bases = [M.graph.score(d) for d in es_sets]
        fl = variant_flags(v, getattr(M, "vocab", 0))
        th, info = train_es(es_sets, bases, M.P, fl, cfg["budget_frac"], cfg["rounds"], gens=cfg["es_gens"] if v == "full" else cfg.get("es_gens_ablation", cfg["es_gens"]),
                            pairs=cfg["es_pairs"], episodes_per_eval=cfg["es_eps"], n_proc=cfg["es_proc"], log=log, vocab=fl.get("vocab", 0))
        M.thetas[v] = th; M.es_info[v] = info
        M.stage = done; pickle.dump(M, open(path, "wb"))
    M.stage = done
    M.fit_seconds = time.time() - t0
    pickle.dump(M, open(path, "wb"))
    return M


# ------------------------------------------------------------------------------------------ policies
def spec_name(s):
    m = s["m"]
    if m in ("random", "age", "static_node", "static_graph"):
        return m
    ad = "off" if s.get("adapt") is None else f"l2={s['adapt']:g}"
    if m == "online":
        return f"online[{s.get('model', 'rev')},{ad},eps={s.get('eps', 0):g}]"
    if m == "cells":
        return f"cells[{ad},eps={s.get('eps', 0):g}]"
    flags = ([] if s.get("evolve", True) else ["no_evo"]) + ([s["tag"]] if s.get("tag") else [])
    return f"org:{s.get('theta', 'full')}[{ad}{(',' + ','.join(flags)) if flags else ''}]"


def build_policy(s, ds, base, M, seed=0, record=False, theta_override=None):
    m = s["m"]
    if m == "random":
        return RandomPolicy()
    if m == "age":
        return RankPolicy(M.age.score(ds), "age")
    if m == "static_node":
        return RankPolicy(M.node.score(ds), "static_node")
    if m == "static_graph":
        return RankPolicy(base, "static_graph")
    ad = s.get("adapt")
    if m == "online":
        return OnlineGBDT(getattr(M, s.get("model", "rev")), adapt=ad is not None, eps=s.get("eps", 0.0), l2=ad if ad else 40.0)
    if m == "cells":
        return CellsOnly(M.P, base, adapt=ad is not None, eps=s.get("eps", 0.0), l2=ad if ad else 40.0)
    if m == "org":
        key = s.get("theta", "full")
        flags = variant_flags(key, getattr(M, "vocab", 0))
        theta = theta_override if theta_override is not None else M.thetas[key]
        return Organism(theta, P=M.P, base=base, cell_adapt=ad is not None, adapt_l2=ad if ad else 40.0,
                        evolve=s.get("evolve", True), seed=seed, record=record, **flags)
    raise ValueError(m)


_PL = {}


def _init_play(payload):
    _PL.update(payload); _PL["base"] = {}


def _play_task(args):
    spec, di, seed = args
    ds = _PL["datasets"][di]
    if di not in _PL["base"]:
        _PL["base"][di] = _PL["M"].graph.score(ds)
    pol = build_policy(spec, ds, _PL["base"][di], _PL["M"], seed=seed, theta_override=_PL.get("theta_override"))
    r = run_game(ds, pol, _PL["budget_frac"], _PL["rounds"], seed)
    return {"hit_rate": r["hit_rate"], "frac_of_oracle": r["frac_of_oracle"], "curve_auc": r["curve_auc"],
            "found_rounds": r["found_rounds"], "seconds": r["seconds"], "stats": r["stats"], "n_pos": r["n_pos"], "B": r["B"]}


def play(specs, datasets, M, cfg, seeds, n_proc=1, theta_override=None, log=log0):
    """Run every spec on every dataset for every seed. Returns {name: {'hit': (n_ds,n_seeds), ...}}."""
    tasks = [(s, di, 1000 + 17 * si) for s in specs for di in range(len(datasets)) for si in range(seeds)]
    payload = {"datasets": datasets, "M": M, "budget_frac": cfg["budget_frac"], "rounds": cfg["rounds"], "theta_override": theta_override}
    if n_proc > 1:
        from .es import single_thread_env
        single_thread_env()
        with mp.get_context("spawn").Pool(n_proc, initializer=_init_play, initargs=(payload,)) as pool:
            res = pool.map(_play_task, tasks, chunksize=1)
    else:
        _init_play(payload)
        res = [_play_task(t) for t in tasks]
    out, k = {}, 0
    for s in specs:
        nm = spec_name(s)
        hit = np.zeros((len(datasets), seeds)); sec = np.zeros((len(datasets), seeds)); fo = np.zeros((len(datasets), seeds))
        cur = np.zeros((len(datasets), seeds)); stats = []
        rounds = []
        for di in range(len(datasets)):
            for si in range(seeds):
                r = res[k]; k += 1
                hit[di, si] = r["hit_rate"]; sec[di, si] = r["seconds"]; fo[di, si] = r["frac_of_oracle"]; cur[di, si] = r["curve_auc"]
                stats.append(r["stats"]); rounds.append(r["found_rounds"])
        out[nm] = {"hit": hit, "sec": sec, "frac_oracle": fo, "curve_auc": cur, "stats": stats, "found_rounds": rounds}
    return out


ADAPTS = [None, 40.0, 200.0]


def tuning_specs():
    sp = []
    for ad in ADAPTS:
        for eps in (0.0, 0.1):
            sp.append({"m": "online", "adapt": ad, "eps": eps, "model": "rev"}); sp.append({"m": "online", "adapt": ad, "eps": eps, "model": "rev_g"})
            sp.append({"m": "cells", "adapt": ad, "eps": eps})
        sp.append({"m": "org", "theta": "full", "adapt": ad})
    return sp


def tune(val_sets, M, cfg, n_proc=1, log=log0):
    """Pick the best variant of each adaptive method on validation datasets (same grid for every method)."""
    res = play(tuning_specs(), val_sets, M, cfg, seeds=max(1, cfg["seeds"]), n_proc=n_proc)
    best = {}
    for nm, r in res.items():
        kind = nm.split("[")[0]
        kind = "org" if kind.startswith("org") else kind
        sc = float(r["hit"].mean())
        if kind not in best or sc > best[kind][0]:
            best[kind] = (sc, nm)
    pick = {}
    for kind, (sc, nm) in best.items():
        for s in tuning_specs():
            if spec_name(s) == nm:
                pick[kind] = s
    log(f"[tune] picked: " + ", ".join(f"{k}->{spec_name(v)}" for k, v in pick.items()))
    return pick, {nm: float(r["hit"].mean()) for nm, r in res.items()}


# ------------------------------------------------------------------------------------------ experiments
def exp_e1(test_sets, M, cfg, nca_missing=None):
    """One-shot forecasting. Returns per-method arrays of metrics over the test datasets."""
    out = {}
    names = ["age_rule", "logreg", "gbdt_node", "gbdt_graph", "nca_residual"]
    rows = {n: [] for n in names}
    for d in test_sets:
        base = M.graph.score(d)
        inp = N.build_inp(d.X, base)
        dl, _ = N.forward(M.P, inp, N.tables(d.graph()))
        sc = {"age_rule": M.age.score(d), "logreg": M.logreg.score(d), "gbdt_node": M.node.score(d),
              "gbdt_graph": base, "nca_residual": base + dl}
        for n in names:
            rows[n].append(forecast_report(d.y, sc[n]))
    for n in names:
        out[n] = {k: [r[k] for r in rows[n]] for k in ("auc", "ap", "lift", "p_at_top")}
    return out


def exp_e4(test_sets, M, ps=(0.0, 0.3, 0.6)):
    """Missing data: each feature entry missing with prob p (NaN for trees; zero + missing-fraction for the NCA)."""
    res = {}
    for p in ps:
        rows = {"gbdt_graph": [], "gbdt_graph_aug": [], "nca_residual": []}
        for di, d in enumerate(test_sets):
            seed = 100 + di
            b_plain = M.graph.score(d, feat_mask_p=p, mask_seed=seed)
            b_aug = M.graph_aug.score(d, feat_mask_p=p, mask_seed=seed)
            rng = np.random.default_rng(seed)
            keep = rng.random(d.X.shape) >= p
            Xm = np.where(keep, d.X, 0.0).astype(np.float32); miss = 1.0 - keep.mean(1)
            inp = N.build_inp(Xm, b_plain, None, None, miss)
            dl, _ = N.forward(M.P, inp, N.tables(d.graph()))
            for n, s in (("gbdt_graph", b_plain), ("gbdt_graph_aug", b_aug), ("nca_residual", b_plain + dl)):
                rows[n].append(forecast_report(d.y, s))
        res[str(p)] = {n: {k: [r[k] for r in rr] for k in ("auc", "ap", "lift")} for n, rr in rows.items()}
    return res


def summarize_play(res, key="hit"):
    out = {}
    for nm, r in res.items():
        per_ds = r[key].mean(1)
        m, lo, hi = bootstrap_ci(per_ds)
        out[nm] = {"mean": m, "lo": lo, "hi": hi, "per_ds": per_ds.tolist(), "sec": float(r["sec"].mean())}
    return out


def e2_specs(pick, thetas=None):
    have = set(thetas) if thetas is not None else {"full", "no_scent", "no_plastic", "workers_only", "blind"}
    sp = [{"m": "random"}, {"m": "age"}, {"m": "static_node"}, {"m": "static_graph"}, pick["online"], pick["cells"]]
    org = pick["org"]
    sp.append(dict(org, theta="full"))
    for v in ("no_scent", "no_plastic", "blind"):
        if v in have:
            sp.append(dict(org, theta=v))
    if "no_words" in have:
        sp.append(dict(org, theta="no_words"))
    if "workers_only" in have:
        sp.append({"m": "org", "theta": "workers_only", "adapt": None})
    sp.append(dict(org, theta="full", evolve=False))
    sp.append({"m": "org", "theta": "full", "adapt": None, "tag": "no_adapt"})       # without cell-readout adaptation
    return sp


def is_primary_org(n):
    return n.startswith("org:full") and "no_evo" not in n and "no_adapt" not in n


BASELINES = ("random", "age", "static_node", "static_graph")


def verdicts(summ, res, pick_names, val_scores):
    """Compare the full organism with the strongest baseline (chosen on VALIDATION) and with cells-only."""
    base_names = [n for n in summ if n.split("[")[0] in ("random", "age", "static_node", "static_graph", "online")]
    # strongest baseline = best validation score among baselines (online variants use their val score)
    def vs(n):
        return val_scores.get(n, None)
    cand = [(vs(n), n) for n in base_names if vs(n) is not None]
    strongest = max(cand)[1] if cand else max(base_names, key=lambda n: summ[n]["mean"])
    full = [n for n in summ if is_primary_org(n)]
    cells = [n for n in summ if n.startswith("cells[")]
    out = {"strongest_baseline": strongest}
    if full:
        f = full[0]
        out["organism"] = f
        d = paired_diff_ci(res[f]["hit"].mean(1), res[strongest]["hit"].mean(1))
        out["org_vs_strongest"] = dict(zip(("diff", "lo", "hi", "p_pos"), d))
        out["org_vs_strongest"]["verdict"] = "WIN" if d[1] > 0 else ("LOSS" if d[2] < 0 else "TIE")
        if cells:
            d2 = paired_diff_ci(res[f]["hit"].mean(1), res[cells[0]]["hit"].mean(1))
            out["org_vs_cells"] = dict(zip(("diff", "lo", "hi", "p_pos"), d2))
            out["org_vs_cells"]["verdict"] = "WIN" if d2[1] > 0 else ("LOSS" if d2[2] < 0 else "TIE")
    return out
