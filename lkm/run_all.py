"""Run the whole study: unit tests, synthetic families, shift, real data, report.

    python -m lkm.run_all            # full study (about 1-2 hours on 4 cores)
    python -m lkm.run_all --smoke    # tiny end-to-end check (a few minutes)
Outputs go to ./lkm_out (report.html, results.json, logs, cache). Every stage is cached and resumable.
"""
import os, sys, json, time, platform, traceback, argparse, io, unittest
import numpy as np
from . import pipeline as PL
from .pipeline import CFG_FULL, CFG_SMOKE, log0
from .worlds import make_bundle
from .organism import unpack, HD, DD


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if (o != o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def run_tests(log=log0):
    suite = unittest.defaultTestLoader.discover(os.path.join(os.path.dirname(__file__), "..", "tests"))
    buf = io.StringIO()
    res = unittest.TextTestRunner(stream=buf, verbosity=2).run(suite)
    log(f"[tests] run={res.testsRun} failures={len(res.failures)} errors={len(res.errors)}")
    return {"run": res.testsRun, "failures": len(res.failures), "errors": len(res.errors), "output": buf.getvalue()[-6000:]}


def life_summary(M):
    out = {}
    for v, th in M.thetas.items():
        pr = unpack(th[None, :])
        wm = pr["Wm"][0]
        out[v] = {"instinct_wd": np.round(pr["wd"][0], 2).tolist(),
                  "move_bias": {n: round(float(wm[i, HD]), 2) for i, n in enumerate(
                      ["cell_risk", "base_risk", "scent_hit", "scent_visited", "verified_pos", "verified_neg", "age", "degree", "nbr_risk", "nbr_scent", "rel_sp", "rel_br", "rel_ad"])},
                  "jump_bias": round(float(pr["wj"][0][HD]), 2), "log_temp": round(float(pr["misc"][0][0]), 2),
                  "plasticity_rate": round(float(np.log1p(np.exp(pr["misc"][0][1])) * 0.5), 3),
                  "trace_decay": round(float(1 / (1 + np.exp(-pr["misc"][0][2]))), 2)}
    return out


def life_stats(res):
    """Aggregate births/deaths/energy/jump-rate logs from organism games."""
    out = {}
    for nm, r in res.items():
        if not nm.startswith("org"):
            continue
        st = [s for s in r["stats"] if s]
        if not st:
            continue
        out[nm] = {k: np.mean([s[k] for s in st], axis=0).tolist() for k in ("births", "deaths", "energy", "jump_rate", "unused")}
    return out


def family_synthetic(kind, cfg, cache, log, rho=0.5, base_rate=0.10):
    tag = f"syn-{kind}"
    n = cfg["world_n"]
    train = make_bundle(kind, range(100, 100 + cfg["n_train"]), n=n, rho=rho, base_rate=base_rate)
    val = make_bundle(kind, range(500, 500 + cfg["n_val"]), n=n, rho=rho, base_rate=base_rate)
    test = make_bundle(kind, range(1900, 1900 + cfg["n_test"]), n=n, rho=rho, base_rate=base_rate)
    k = max(1, cfg["n_val"] // 2)
    es_sets, tune_sets = val[:k], val[k:]
    variants = ("full",) if kind == "control" else ("full", "no_scent", "no_plastic", "workers_only", "blind")
    M = PL.fit_models(tag, train, es_sets, cache, cfg, cfg["nca_steps"], variants=variants, log=log, val_for_nca=tune_sets[0])
    pick, tune_scores = PL.tune(tune_sets, M, cfg, n_proc=1, log=log)
    return dict(kind=kind, rho=rho, train=train, val=val, test=test, M=M, pick=pick, tune_scores=tune_scores, tune_sets=tune_sets)


def evaluate_family(fam, cfg, log, n_proc=1, seeds=None):
    M, pick, test = fam["M"], fam["pick"], fam["test"]
    seeds = seeds or cfg["seeds"]
    out = {"kind": fam["kind"], "pick": {k: PL.spec_name(v) for k, v in pick.items()},
           "base_rate": float(np.mean([d.y.mean() for d in test])), "n_test": len(test),
           "n_nodes": [d.n for d in test], "edges": test[0].graph().edge_count(),
           "nca": {"info": M.nca_info}, "es": {v: {"start": i["start"], "best": i["best"], "seconds": i["seconds"], "hist": i["hist"]} for v, i in M.es_info.items()},
           "life_genomes": life_summary(M)}
    t = time.time()
    out["e1"] = PL.exp_e1(test, M, cfg)
    out["e4"] = PL.exp_e4(test, M)
    res = PL.play(PL.e2_specs(pick, M.thetas), test, M, cfg, seeds, n_proc=n_proc, log=log)
    out["e2"] = PL.summarize_play(res); out["e2_curve_auc"] = PL.summarize_play(res, "curve_auc")
    out["e2_frac_oracle"] = PL.summarize_play(res, "frac_oracle")
    out["e2_rounds"] = {nm: np.mean(r["found_rounds"], axis=0).tolist() for nm, r in res.items()}
    val_scores = dict(fam["tune_scores"])
    for b in ("random", "age", "static_node", "static_graph"):
        vr = PL.play([{"m": b}], fam["tune_sets"], M, cfg, 1, n_proc=1)
        val_scores[b] = float(vr[b]["hit"].mean())
    out["e2_verdict"] = PL.verdicts(out["e2"], res, pick, val_scores)
    out["e7_life"] = life_stats(res)
    out["e5_seconds"] = {nm: out["e2"][nm]["sec"] for nm in out["e2"]}
    log(f"[{fam['kind']}] evaluation done in {time.time()-t:.0f}s; verdict {out['e2_verdict'].get('org_vs_strongest')}")
    return out, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default="lkm_out")
    ap.add_argument("--only", default="", help="comma list: syn,real,tests")
    ap.add_argument("--no-real", action="store_true")
    a = ap.parse_args()
    cfg = CFG_SMOKE if a.smoke else CFG_FULL
    out = a.out + ("_smoke" if a.smoke else "")
    cache = os.path.join(out, "cache"); os.makedirs(cache, exist_ok=True)
    logf = open(os.path.join(out, "run.log"), "a")

    def log(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    t_all = time.time()
    rpath = os.path.join(out, "results.json")
    results = json.load(open(rpath)) if os.path.exists(rpath) else {}
    results["system"] = {"python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count(),
                         "smoke": a.smoke, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    results["config"] = jsonable(cfg)
    only = set(a.only.split(",")) if a.only else {"tests", "syn", "real"}

    def save():
        json.dump(jsonable(results), open(rpath, "w"))

    if "tests" in only:
        try:
            results["tests"] = run_tests(log)
        except Exception:
            results["tests"] = {"error": traceback.format_exc()}
        save()
    fams = {}
    if "syn" in only:
        for kind, rho, br in (("control", 0.5, 0.10), ("contagion", 0.5, 0.10), ("cascade", 0.2, 0.08)):
            key = f"syn_{kind}"
            if key in results and "e2" in results[key]:
                log(f"[skip] {key} cached"); continue
            try:
                fam = family_synthetic(kind, cfg, cache, log, rho=rho, base_rate=br)
                res, _ = evaluate_family(fam, cfg, log)
                results[key] = res; fams[kind] = fam
                if kind == "contagion":
                    results[key]["e1_rho_sweep"] = rho_sweep(cfg, cache, log)
                    results["syn_shifted"] = shifted(fam, cfg, log)
                save()
            except Exception:
                results[key] = {"error": traceback.format_exc()}; log(results[key]["error"]); save()
    if "real" in only and not a.no_real:
        try:
            results["real"] = real_study(cfg, cache, log, a.smoke, results)
        except Exception:
            results["real"] = {"error": traceback.format_exc()}; log(results["real"]["error"])
        save()
    try:
        results["e5_timing"] = timing_benchmark(cache, cfg, log)
    except Exception:
        results["e5_timing"] = {"error": traceback.format_exc()}
    results["total_seconds"] = time.time() - t_all
    save()
    from . import report
    try:
        report.build(out, results, cfg)
    except Exception:
        log("[report] FAILED\n" + traceback.format_exc())
    log(f"[done] total {time.time()-t_all:.0f}s -> {out}/report.html")


def rho_sweep(cfg, cache, log):
    out = {}
    for rho in cfg["rhos"]:
        n = cfg["world_n"]
        train = make_bundle("contagion", range(100, 100 + cfg["n_train"]), n=n, rho=rho)
        test = make_bundle("contagion", range(1900, 1900 + cfg["n_test"]), n=n, rho=rho)
        M = PL.fit_models(f"syn-contagion-rho{rho}", train, train[:1], cache, cfg, cfg["nca_steps"], variants=(), log=log, val_for_nca=train[0])
        out[str(rho)] = PL.exp_e1(test, M, cfg)
    return out


def shifted(fam, cfg, log):
    """E3: models trained on the contagion world, deployed on the shifted world."""
    n = cfg["world_n"]
    val = make_bundle("shifted", range(700, 700 + cfg["n_val"]), n=n)
    test = make_bundle("shifted", range(1950, 1950 + cfg["n_test"]), n=n)
    M = fam["M"]
    pick, tune_scores = PL.tune(val, M, cfg, n_proc=1, log=log)
    res = PL.play(PL.e2_specs(pick, M.thetas), test, M, cfg, cfg["seeds"], n_proc=1)
    summ = PL.summarize_play(res)
    val_scores = dict(tune_scores)
    for b in ("random", "age", "static_node", "static_graph"):
        val_scores[b] = float(PL.play([{"m": b}], val, M, cfg, 1)[b]["hit"].mean())
    return {"pick": {k: PL.spec_name(v) for k, v in pick.items()}, "e2": summ, "e2_rounds": {nm: np.mean(r["found_rounds"], axis=0).tolist() for nm, r in res.items()},
            "e2_verdict": PL.verdicts(summ, res, pick, val_scores), "e1": PL.exp_e1(test, M, cfg), "base_rate": float(np.mean([d.y.mean() for d in test]))}


def real_study(cfg, cache, log, smoke, results):
    from .realdata import build_real
    R = build_real(log=log)
    train, vch, pools = R["train"], R["val_chunks"], R["test_pools"]
    es_ix = [i for i, d in enumerate(vch) if d.n <= 13000]
    tune_ix = [i for i in range(len(vch)) if i not in es_ix and vch[i].n <= 34000]
    if smoke:
        train, es_ix, tune_ix, pools = train[:4], es_ix[:2], tune_ix[:1], pools[:1]
    es_sets = [vch[i] for i in es_ix]; tune_sets = [vch[i] for i in tune_ix] or es_sets[:1]
    M = PL.fit_models("real", train, es_sets, cache, cfg, cfg["real_nca_steps"], log=log, val_for_nca=es_sets[0])
    pick, tune_scores = PL.tune(tune_sets, M, cfg, n_proc=cfg["es_proc"], log=log)
    cfg_r = dict(cfg)
    out = {"info": R["info"], "diag": R["diag"], "pick": {k: PL.spec_name(v) for k, v in pick.items()},
           "n_test": len(pools), "n_nodes": [d.n for d in pools], "base_rate": float(np.mean([d.y.mean() for d in pools])),
           "base_rate_gone": float(np.mean([d.y_alt.mean() for d in pools])),
           "nca": {"info": M.nca_info}, "es": {v: {"start": i["start"], "best": i["best"], "seconds": i["seconds"], "hist": i["hist"]} for v, i in M.es_info.items()},
           "life_genomes": life_summary(M), "features": R["full"].feat_names}
    out["e1"] = PL.exp_e1(pools, M, cfg)
    out["e1_gone_label"] = _e1_gone(pools, M)
    out["e4"] = PL.exp_e4(pools, M)
    seeds = cfg["real_seeds"]
    specs = PL.e2_specs(pick, M.thetas)
    res = PL.play(specs, pools, M, cfg, seeds, n_proc=cfg["es_proc"], log=log)
    out["e2"] = PL.summarize_play(res); out["e2_curve_auc"] = PL.summarize_play(res, "curve_auc")
    out["e2_rounds"] = {nm: np.mean(r["found_rounds"], axis=0).tolist() for nm, r in res.items()}
    val_scores = dict(tune_scores)
    for b in ("random", "age", "static_node", "static_graph"):
        val_scores[b] = float(PL.play([{"m": b}], tune_sets, M, cfg, 1)[b]["hit"].mean())
    out["e2_verdict"] = PL.verdicts(out["e2"], res, pick, val_scores)
    out["e7_life"] = life_stats(res)
    out["e5_seconds"] = {nm: out["e2"][nm]["sec"] for nm in out["e2"]}
    # E6 sim-to-real: genome evolved on synthetic contagion worlds, deployed on the real mesh without retraining
    syn = results.get("syn_contagion")
    cpath = os.path.join(cache, "models_syn-contagion.pkl")
    if os.path.exists(cpath):
        import pickle
        Ms = pickle.load(open(cpath, "rb"))
        spec = dict(pick["org"], theta="full")
        tr = PL.play([spec], pools, M, cfg, seeds, n_proc=cfg["es_proc"], theta_override=Ms.thetas["full"])
        nm = PL.spec_name(spec)
        sm = PL.summarize_play(tr)[nm]
        full_nm = [k for k in out["e2"] if PL.is_primary_org(k)]
        out["e6_transfer"] = {"transfer": sm, "native": out["e2"][full_nm[0]] if full_nm else None}
    return out


def timing_benchmark(cache, cfg, log):
    """E5 efficiency, measured alone in a quiet single process (the study itself runs processes in parallel)."""
    import pickle
    from . import nca as N
    from .game import run_game
    out = {}
    for tag, key in (("syn-contagion", "synthetic (5,000 entities)"),):
        p = os.path.join(cache, f"models_{tag}.pkl")
        if not os.path.exists(p):
            continue
        M = pickle.load(open(p, "rb"))
        ds = make_bundle("contagion", [4321], n=cfg["world_n"])[0]
        t = time.time(); base = M.graph.score(ds); t_gb = time.time() - t
        tabs = N.tables(ds.graph())
        t = time.time(); N.forward(M.P, N.build_inp(ds.X, base), tabs); t_nca = time.time() - t
        rows = {}
        pick = {"online": {"m": "online", "adapt": 40.0, "eps": 0.0, "model": "rev"}, "cells": {"m": "cells", "adapt": 40.0, "eps": 0.0},
                "org": {"m": "org", "adapt": 40.0}}
        for spec in PL.e2_specs(pick, M.thetas):
            nm = PL.spec_name(spec)
            if "no_" in nm or "blind" in nm:
                continue
            pol = PL.build_policy(spec, ds, base, M, seed=0)
            t = time.time(); run_game(ds, pol, cfg["budget_frac"], cfg["rounds"], 0); rows[nm] = time.time() - t
        out[key] = {"n": ds.n, "gbdt_graph_score_s": t_gb, "nca_forward_s": t_nca, "game_s": rows}
    log(f"[e5] timing: {json.dumps(jsonable(out))[:300]}")
    return out


def _e1_gone(pools, M):
    """Secondary real label: place removed or closed (a stricter 'business gone' event)."""
    from . import nca as N
    from .metrics import forecast_report
    rows = {k: [] for k in ("age_rule", "gbdt_node", "gbdt_graph", "nca_residual")}
    for d in pools:
        base = M.graph.score(d)
        dl, _ = N.forward(M.P, N.build_inp(d.X, base), N.tables(d.graph()))
        for k, s in (("age_rule", M.age.score(d)), ("gbdt_node", M.node.score(d)), ("gbdt_graph", base), ("nca_residual", base + dl)):
            rows[k].append(forecast_report(d.y_alt, s))
    return {k: {m: [r[m] for r in v] for m in ("auc", "ap", "lift")} for k, v in rows.items()}


if __name__ == "__main__":
    main()
