"""Evolution strategies for the workers' ancestral genome (NumPy only; safe to run in spawned processes).

Fitness of a genome = mean fraction of the oracle's discoveries achieved over episodes on held-out *validation*
worlds (never the test worlds, never the data the baselines/NCA were fitted on). Candidates in a generation share
the same episodes and seeds (common random numbers) to cut noise.
"""
import time
import numpy as np
import multiprocessing as mp
from .organism import Organism, init_genome, p_total

_CTX = {}


def single_thread_env():
    """Worker processes must not each spawn BLAS threads (4 procs x 4 threads on 4 cores is ~10x slower)."""
    import os
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"


def _init(payload):
    _CTX.update(payload)


def _episode(args):
    from .game import run_game
    theta, ei, seed, flags = args
    ds = _CTX["datasets"][ei]
    org = Organism(theta, P=_CTX["P"], base=_CTX["bases"][ei], miss=None, seed=seed, **flags)
    r = run_game(ds, org, _CTX["budget_frac"], _CTX["rounds"], seed)
    return r["frac_of_oracle"]


def train_es(datasets, bases, P, flags, budget_frac=0.03, rounds=8, gens=40, pairs=12, sigma=0.08, lr=0.03,
             episodes_per_eval=3, seed=0, n_proc=4, log=print, theta0=None, check_every=5, vocab=0):
    rng = np.random.default_rng(seed)
    theta = init_genome(seed, vocab) if theta0 is None else np.asarray(theta0, float).copy()
    payload = {"datasets": datasets, "bases": bases, "P": P, "budget_frac": budget_frac, "rounds": rounds}
    single_thread_env()
    ctx = mp.get_context("spawn")
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    best, best_theta, hist = -1.0, theta.copy(), []
    t0 = time.time()
    fixed = [(i, 1000 + i) for i in range(len(datasets))]
    with ctx.Pool(n_proc, initializer=_init, initargs=(payload,)) as pool:
        def evaluate(th):
            return float(np.mean(pool.map(_episode, [(th, i, s, flags) for i, s in fixed])))
        f0 = evaluate(theta)
        log(f"[es] start fitness {f0:.3f}")
        best, best_theta = f0, theta.copy()
        for g in range(1, gens + 1):
            eps = rng.normal(size=(pairs, p_total(vocab)))
            cands = [theta + sigma * e for e in eps] + [theta - sigma * e for e in eps]
            ep = rng.choice(len(datasets), episodes_per_eval, replace=len(datasets) < episodes_per_eval)
            sd = rng.integers(1 << 30, size=episodes_per_eval)
            tasks = [(c, int(e_), int(s_), flags) for c in cands for e_, s_ in zip(ep, sd)]
            res = np.array(pool.map(_episode, tasks, chunksize=2)).reshape(len(cands), episodes_per_eval).mean(1)
            ranks = res.argsort().argsort().astype(float)
            ranks = ranks / (len(res) - 1) - 0.5
            grad = ((ranks[:pairs] - ranks[pairs:])[:, None] * eps).sum(0) / (2 * pairs * sigma)
            m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad ** 2
            theta = theta + lr * (m / (1 - 0.9 ** g)) / (np.sqrt(v / (1 - 0.999 ** g)) + 1e-8)
            hist.append(float(res.mean()))
            if g % check_every == 0 or g == gens:
                f = evaluate(theta)
                if f > best:
                    best, best_theta = f, theta.copy()
                log(f"[es] gen {g}/{gens} pop-mean {np.mean(hist[-check_every:]):.3f} theta-fitness {f:.3f} best {best:.3f} ({time.time()-t0:.0f}s)")
    return best_theta, {"start": f0, "best": best, "hist": hist, "seconds": time.time() - t0}
