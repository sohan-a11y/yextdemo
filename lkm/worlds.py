"""Synthetic worlds with hidden ground truth.

Every world has entities with attributes, three relation types (space, brand, address) and a hidden
"needs attention" label y (the fact will change or go stale in the next period). Hidden shocks shared
along relations make the graph informative: a chain-wide event, a regional event, a building event.

World kinds
  control    no graph effects at all (a graph can only add noise)  -> sanity check
  contagion  hidden regional/brand/address shocks; recent events are observed with probability rho
  shifted    like contagion but with *different* parameters (age effect reversed, new category risks,
             stronger brand shocks) -> tests adaptation of models trained on 'contagion'
The generator is deliberately documented because results on synthetic data depend on its assumptions;
the report says so.
"""
from dataclasses import dataclass, field
import numpy as np
from .graph import Graph, knn_edges, group_edges, REL

FEATS = (["age", "nsrc", "conf", "has_web", "has_phone"] + [f"cat{i}" for i in range(8)] +
         ["ev1", "ev2", "brand"])
AGG_COLS = [FEATS.index(c) for c in ("age", "conf", "ev1", "ev2")]
AGE_COL = FEATS.index("age")


@dataclass
class Dataset:
    name: str
    X: np.ndarray                 # (n,F) float32 features available at forecasting time
    y: np.ndarray                 # (n,) int8 hidden label
    edges: dict                   # rel -> (src,dst)
    xy: np.ndarray                # (n,2)
    feat_names: list
    agg_cols: list
    age_col: int
    meta: dict = field(default_factory=dict)
    y_alt: np.ndarray = None      # optional secondary label
    _g: Graph = None

    @property
    def n(self):
        return len(self.y)

    def graph(self):
        if self._g is None:
            self._g = Graph.from_edges(self.n, self.edges)
        return self._g

    def subset(self, idx, name=None):
        idx = np.asarray(idx)
        edges = {}
        loc = np.full(self.n, -1, np.int64); loc[idx] = np.arange(len(idx))
        for r in REL:
            s, d = self.edges[r]
            ls, ld = loc[s], loc[d]; ok = (ls >= 0) & (ld >= 0)
            edges[r] = (ls[ok], ld[ok])
        return Dataset(name or self.name, self.X[idx], self.y[idx], edges, self.xy[idx],
                       self.feat_names, self.agg_cols, self.age_col, dict(self.meta),
                       None if self.y_alt is None else self.y_alt[idx])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


PARAMS = {
    "control":   dict(a_age=0.6, a_src=-0.4, a_conf=-0.7, cat_sd=0.6, s_sp=0.0, s_br=0.0, s_ad=0.0, s_id=0.7, own=0.0),
    "contagion": dict(a_age=0.5, a_src=-0.3, a_conf=-0.5, cat_sd=0.5, s_sp=1.0, s_br=1.1, s_ad=0.7, s_id=0.5, own=0.4),
    "cascade":   dict(a_age=0.4, a_src=-0.3, a_conf=-0.4, cat_sd=0.4, s_sp=2.0, s_br=2.6, s_ad=1.0, s_id=0.4, own=0.3),
    "shifted":   dict(a_age=-0.4, a_src=0.3, a_conf=0.2, cat_sd=0.9, s_sp=0.7, s_br=1.8, s_ad=0.4, s_id=0.5, own=0.4),
}


def make_world(seed, n=5000, kind="contagion", rho=0.5, base_rate=0.10, cat_seed=None):
    """Generate one world. `cat_seed` fixes category risks across worlds of the same family."""
    rng = np.random.default_rng(seed)
    P = PARAMS[kind]
    # ---- geometry: clustered city
    k_cl = 12
    centers = rng.normal(0, 1.0, (k_cl, 2))
    cl = rng.integers(0, k_cl, n)
    xy = centers[cl] + rng.normal(0, 0.18, (n, 2))
    # ---- brands (chains scattered across the city) and addresses (co-located groups)
    brand = np.full(n, -1)
    free = rng.permutation(n)[: int(0.30 * n)]
    b_id, p = 0, 0
    while p < len(free):
        size = int(min(25, 2 + rng.geometric(0.18)))
        brand[free[p:p + size]] = b_id; b_id += 1; p += size
    addr = np.full(n, -1)
    free = rng.permutation(n)[: int(0.15 * n)]
    a_id, p = 0, 0
    while p < len(free):
        size = int(rng.integers(2, 4))
        grp = free[p:p + size]
        xy[grp] = xy[grp[0]] + rng.normal(0, 0.002, (len(grp), 2))
        addr[grp] = a_id; a_id += 1; p += size
    edges = {"sp": knn_edges(xy, 6), "br": group_edges(brand, 2, rng), "ad": group_edges(addr, 2, rng)}
    G = Graph.from_edges(n, edges)
    # ---- attributes
    age_raw = rng.lognormal(4.0, 0.8, n)
    age = (np.log(age_raw) - 4.0) / 0.8
    nsrc_raw = 1 + np.minimum(rng.poisson(1.2, n), 4)
    nsrc = (nsrc_raw - 2.2) / 1.0
    conf_raw = rng.beta(5, 2, n)
    conf = (conf_raw - 0.71) / 0.16
    has_web = (rng.random(n) < 0.6).astype(np.float32)
    has_phone = (rng.random(n) < 0.8).astype(np.float32)
    cat = rng.integers(0, 8, n)
    crng = np.random.default_rng(cat_seed if cat_seed is not None else 12345)
    cat_risk = crng.normal(0, P["cat_sd"], 8)
    # ---- hidden shocks
    u = rng.normal(0, 1, n)
    for _ in range(3):
        u = G.An["sp"] @ u
    u = (u - u.mean()) / (u.std() + 1e-9)
    u_br = rng.normal(0, 1, max(b_id, 1))[np.maximum(brand, 0)] * (brand >= 0)
    u_ad = rng.normal(0, 1, max(a_id, 1))[np.maximum(addr, 0)] * (addr >= 0)
    shock = P["s_sp"] * u + P["s_br"] * u_br + P["s_ad"] * u_ad
    base_lin = P["a_age"] * age + P["a_src"] * nsrc + P["a_conf"] * conf + cat_risk[cat]

    def z_of(extra_noise):
        return base_lin + shock + rng.normal(0, P["s_id"], n) + extra_noise

    # recent events from earlier periods share the hidden shocks (this is what neighbours reveal)
    ev = []
    own = rng.normal(0, 1, n) * P["own"]          # persistent idiosyncratic propensity
    for _ in range(2):
        zk = cat_risk[cat] + shock + own + rng.normal(0, P["s_id"], n)
        ev.append((rng.random(n) < _sigmoid(zk + _bias(zk, base_rate))) * (rng.random(n) < rho))
    z = z_of(own)
    bias = _bias(z, base_rate)
    y = (rng.random(n) < _sigmoid(z + bias)).astype(np.int8)
    X = np.column_stack([age, nsrc, conf, has_web, has_phone] + [(cat == i).astype(np.float32) for i in range(8)]
                        + [ev[0].astype(np.float32), ev[1].astype(np.float32), (brand >= 0).astype(np.float32)])
    return Dataset(f"{kind}-{seed}", X.astype(np.float32), y, edges, xy.astype(np.float32), FEATS,
                   AGG_COLS, AGE_COL, {"kind": kind, "rho": rho, "seed": seed, "base_rate": float(y.mean())})


def _bias(z, target, iters=40):
    lo, hi = -10.0, 10.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def make_bundle(kind, seeds, n=5000, rho=0.5, base_rate=0.10):
    return [make_world(s, n=n, kind=kind, rho=rho, base_rate=base_rate) for s in seeds]
