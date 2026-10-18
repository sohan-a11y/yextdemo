"""Active verification game.

A pool of entities has hidden labels y (needs attention). Over R rounds an agent may verify B_r entities per
round (a verification reveals the true label). Goal: find as many positives as possible within the total
budget. Everything the agent learns (revealed labels) is available to it afterwards.
This is the operational problem behind 'stale data': you can only check so many facts per day.
"""
import time
import numpy as np
from .baselines import sigmoid, fit_offset_logreg, apply_offset_logreg
from .metrics import discovery_stats
from . import nca as N


class Env:
    def __init__(self, ds, budget_frac=0.03, rounds=8, seed=0):
        self.ds, self.G, self.n, self.y = ds, ds.graph(), ds.n, ds.y
        total = max(rounds, int(round(budget_frac * self.n)))
        base = total // rounds
        self.B = np.full(rounds, base, int)
        self.B[: total - base * rounds] += 1
        self.rounds = rounds
        self.round = 0
        self.rng = np.random.default_rng(seed)
        self.verified = np.zeros(self.n, np.int8)      # 0 unknown, 1 negative, 2 positive

    @property
    def rev_pos(self):
        return (self.verified == 2).astype(np.float32)

    @property
    def rev_neg(self):
        return (self.verified == 1).astype(np.float32)

    def verify(self, ids):
        labels = self.y[ids].astype(int)
        self.verified[ids] = 1 + labels
        return labels


def run_game(ds, policy, budget_frac=0.03, rounds=8, seed=0):
    env = Env(ds, budget_frac, rounds, seed)
    policy.reset(env)
    found, picked = [], []
    t0 = time.time()
    for r in range(rounds):
        env.round = r
        ids = np.asarray(policy.pick(env, int(env.B[r])), dtype=np.int64)
        if len(ids) > env.B[r] or len(np.unique(ids)) != len(ids) or (len(ids) and (env.verified[ids] != 0).any()):
            raise RuntimeError("policy violated budget / uniqueness / re-verification rule")
        labels = env.verify(ids) if len(ids) else np.zeros(0, int)
        policy.observe(env, ids, labels)
        found.append(int(labels.sum())); picked.append(len(ids))
    out = discovery_stats(found, env.B, int(ds.y.sum()))
    out.update({"found_rounds": found, "picked": picked, "B": env.B.tolist(), "n": ds.n, "n_pos": int(ds.y.sum()),
                "seconds": time.time() - t0, "stats": policy.stats()})
    return out


# ------------------------------------------------------------------------------------------- policies
class Policy:
    name = "policy"

    def reset(self, env):
        pass

    def observe(self, env, ids, labels):
        pass

    def stats(self):
        return {}

    @staticmethod
    def top_unverified(score, env, k, eps=0.0):
        cand = np.where(env.verified == 0)[0]
        if len(cand) <= k:
            return cand
        k_greedy = k - int(round(eps * k))
        sc = score[cand]
        top = cand[np.argpartition(-sc, k_greedy - 1)[:k_greedy]] if k_greedy > 0 else np.zeros(0, int)
        if k_greedy < k:
            rest = np.setdiff1d(cand, top, assume_unique=False)
            top = np.concatenate([top, env.rng.choice(rest, k - k_greedy, replace=False)])
        return top


class RandomPolicy(Policy):
    name = "random"

    def pick(self, env, k):
        return env.rng.choice(np.where(env.verified == 0)[0], k, replace=False)


class RankPolicy(Policy):
    """Static ranking: scores never change as labels are revealed."""

    def __init__(self, score, name):
        self.score, self.name = score, name

    def pick(self, env, k):
        return self.top_unverified(self.score, env, k)


class OnlineGBDT(Policy):
    """Strong baseline: GBDT with graph + revealed-neighbour features, plus online fine-tuning of a linear
    correction (ridge logistic with offset) on the labels revealed so far, plus epsilon-greedy exploration."""
    name = "gbdt_online"

    def __init__(self, model, adapt=True, eps=0.1, name=None, score_kw=None, l2=40.0):
        self.model, self.adapt, self.eps, self.l2 = model, adapt, eps, l2
        self.kw = dict(score_kw or {})
        if name:
            self.name = name
        self.w = None

    def reset(self, env):
        self.w = None

    def scores(self, env):
        s = self.model.score(env.ds, rev_pos=env.rev_pos, rev_neg=env.rev_neg, **self.kw)
        if self.adapt and self.w is not None:
            s = apply_offset_logreg(np.nan_to_num(env.ds.X), self.w, s)
        return s

    def pick(self, env, k):
        return self.top_unverified(self.scores(env), env, k, self.eps)

    def observe(self, env, ids, labels):
        if not self.adapt:
            return
        rev = np.where(env.verified > 0)[0]
        if len(rev) < 60 or env.verified[rev].min() == env.verified[rev].max():
            return
        s0 = self.model.score(env.ds, rev_pos=env.rev_pos, rev_neg=env.rev_neg, **self.kw)
        F = np.nan_to_num(env.ds.X)
        self.w = fit_offset_logreg(F[rev], (env.verified[rev] == 2).astype(float), s0[rev], l2=self.l2)


class CellsOnly(Policy):
    """The NCA field alone: relax the cells with the revealed labels, rank by readout, verify the top."""
    name = "cells_only"

    def __init__(self, P, base, ds_tabs=None, adapt=True, eps=0.1, miss=None, name=None, l2=40.0):
        self.P, self.base, self.adapt, self.eps, self.miss, self.l2 = P, base, adapt, eps, miss, l2
        self.tabs = ds_tabs
        if name:
            self.name = name

    def reset(self, env):
        if self.tabs is None:
            self.tabs = N.tables(env.G)
        self.readout = None
        self.Xn = np.nan_to_num(env.ds.X)
        self._relax(env)

    def _relax(self, env):
        inp = N.build_inp(self.Xn, self.base, env.rev_pos, env.rev_neg, self.miss)
        self.s = N.forward(self.P, inp, self.tabs)[1]

    def logits(self, env):
        wo, bo = (self.P["wo"], self.P["bo"]) if self.readout is None else self.readout
        return self.base + self.s @ wo + bo

    def pick(self, env, k):
        return self.top_unverified(self.logits(env), env, k, self.eps)

    def observe(self, env, ids, labels):
        self._relax(env)
        if not self.adapt:
            return
        rev = np.where(env.verified > 0)[0]
        if len(rev) < 60 or env.verified[rev].min() == env.verified[rev].max():
            return
        off = self.base + self.s @ self.P["wo"] + self.P["bo"]
        w = fit_offset_logreg(self.s[rev], (env.verified[rev] == 2).astype(float), off[rev], l2=self.l2)
        self.readout = (self.P["wo"] + w[:-1].astype(np.float32), self.P["bo"] + np.float32(w[-1]))
