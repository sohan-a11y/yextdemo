"""Baselines. The strong ones receive exactly the information the organism receives."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from .graph import graph_features, reveal_features


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------------- feature builders
def node_feats(ds, **_):
    return ds.X


def graph_feats(ds, **_):
    return np.concatenate([ds.X, graph_features(ds.graph(), ds.X, ds.agg_cols)], 1)


def rev_feats(ds, rev_pos=None, rev_neg=None, **_):
    G = ds.graph()
    z = np.zeros(ds.n, np.float32)
    rp = z if rev_pos is None else rev_pos
    rn = z if rev_neg is None else rev_neg
    return np.concatenate([graph_feats(ds), reveal_features(G, rp, rn),
                           rp[:, None].astype(np.float32), rn[:, None].astype(np.float32)], 1)


def mask_features(F, p, rng):
    """Randomly drop feature entries (set NaN) with probability p (GBDT handles NaN natively)."""
    if p <= 0:
        return F
    F = F.copy()
    F[rng.random(F.shape) < p] = np.nan
    return F


# --------------------------------------------------------------------------- models
class AgeRule:
    """Rank by age since last source update. The classical staleness heuristic."""
    name = "age_rule"

    def __init__(self, **_):
        pass

    def fit(self, train, **_):
        return self

    def score(self, ds, **_):
        return ds.X[:, ds.age_col].astype(np.float64)


class LogReg:
    name = "logreg"

    def __init__(self, **_):
        self.m = LogisticRegression(C=1.0, max_iter=400)

    def fit(self, train, **_):
        X = np.concatenate([np.nan_to_num(d.X) for d in train]); y = np.concatenate([d.y for d in train])
        self.m.fit(X, y); return self

    def score(self, ds, **_):
        return self.m.decision_function(np.nan_to_num(ds.X))


class GBDT:
    """Gradient-boosted trees on node features, optionally with graph aggregates and
    revealed-neighbour features. `augment` trains with random reveals / random feature masking."""

    def __init__(self, kind="graph", seed=0, max_iter=250, mask_aug=False, name=None, greedy_aug=False, **_):
        self.kind, self.seed, self.max_iter, self.mask_aug, self.greedy_aug = kind, seed, max_iter, mask_aug, greedy_aug
        self.name = name or f"gbdt_{kind}"
        self.m = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.07, max_leaf_nodes=31,
                                                l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                                n_iter_no_change=15, random_state=seed)

    def _build(self, ds, rev_pos=None, rev_neg=None):
        if self.kind == "node":
            return node_feats(ds)
        if self.kind == "graph":
            return graph_feats(ds)
        return rev_feats(ds, rev_pos=rev_pos, rev_neg=rev_neg)

    def fit(self, train, **_):
        rng = np.random.default_rng(self.seed)
        Xs, ys = [], []
        for d in train:
            if self.kind == "rev":
                for rep in range(3):
                    qp, qn = rng.uniform(0.0, 0.35), rng.uniform(0.0, 0.06)
                    rp = ((d.y == 1) & (rng.random(d.n) < qp)).astype(np.float32)
                    rn = ((d.y == 0) & (rng.random(d.n) < qn)).astype(np.float32)
                    keep = (rp + rn) == 0
                    F = self._build(d, rp, rn)
                    Xs.append(F[keep]); ys.append(d.y[keep])
            else:
                F = self._build(d)
                if self.mask_aug:
                    Xs.append(mask_features(F, rng.uniform(0, 0.5), rng)); ys.append(d.y)
                Xs.append(F); ys.append(d.y)
        self.m.fit(np.concatenate(Xs), np.concatenate(ys))
        return self

    def score(self, ds, rev_pos=None, rev_neg=None, feat_mask_p=0.0, mask_seed=0, **_):
        F = self._build(ds, rev_pos, rev_neg)
        if feat_mask_p > 0:
            F = mask_features(F, feat_mask_p, np.random.default_rng(mask_seed))
        return self.m.decision_function(F)


def oof_scores(train, factory, folds=4):
    """Out-of-fold logits for each training dataset (cross-fitting by dataset/chunk)."""
    k = max(2, min(folds, len(train)))
    fold = np.arange(len(train)) % k
    out = [None] * len(train)
    for f in range(k):
        tr = [d for d, g in zip(train, fold) if g != f]
        m = factory().fit(tr)
        for i, (d, g) in enumerate(zip(train, fold)):
            if g == f:
                out[i] = m.score(d)
    return out


# --------------------------------------------------------------------------- online fine-tuning
def fit_offset_logreg(F, y, offset, l2=5.0, iters=12):
    """Ridge logistic regression with a fixed offset (the base logit), by Newton steps.
    Returns weights w (incl. intercept as last element). Used for online adaptation."""
    n, d = F.shape
    A = np.concatenate([F, np.ones((n, 1))], 1).astype(np.float64)
    w = np.zeros(d + 1)
    reg = np.full(d + 1, l2); reg[-1] = 0.5
    for _ in range(iters):
        p = sigmoid(A @ w + offset)
        g = A.T @ (p - y) + reg * w
        H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(reg) + 1e-6 * np.eye(d + 1)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-4:
            break
    return w


def apply_offset_logreg(F, w, offset):
    A = np.concatenate([F, np.ones((F.shape[0], 1))], 1)
    return A @ w + offset
