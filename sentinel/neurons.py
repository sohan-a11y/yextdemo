"""The neurons of the Sentinel (NumPy only, CPU).

  KitEnsemble  the REFLEX: learns what normal looks like on THIS machine and scores how strange a window is. A KitNET-style online
               ensemble (Mirsky et al., 2018): features are grouped by correlation, each group has a small autoencoder trained one
               window at a time, and an output autoencoder reads the group errors. It never sees an attack example.
  TinyMLP      the shape of a neuron the FORGE builds from a written purpose: a 16-unit network trained on the machine's own normal
               windows plus synthetic windows of the described threat.

Safety properties built into the reflex: a childhood phase with no alerts from it, a calibrated threshold, and learning only from windows
it scored as normal (a slowly escalating attacker cannot teach it that the attack is normal in one step; a human can tell it "that was
fine" through feedback, which is the only other way in).
"""
import json
import numpy as np

from .features import FEATURES, DESCRIPTIONS

SD_FLOOR = 0.25          # features are log1p-scaled counts; below this spread a change of one event still counts
ZCLIP = 6.0
MIN_THR = 0.2            # a perfectly quiet machine must not make every small change an emergency


class Running:
    """Welford running mean/variance for a vector."""

    def __init__(self, d):
        self.n = 0
        self.mu = np.zeros(d)
        self.m2 = np.zeros(d)

    def update(self, x):
        self.n += 1
        dlt = x - self.mu
        self.mu += dlt / self.n
        self.m2 += dlt * (x - self.mu)

    def sd(self, floor=SD_FLOOR):
        var = self.m2 / max(self.n - 1, 1)
        return np.sqrt(var + floor ** 2)

    def z(self, x, floor=SD_FLOOR):
        return np.clip((x - self.mu) / self.sd(floor), -ZCLIP, ZCLIP)


class OnlineAE:
    """Tied-weight autoencoder with tanh hidden layer, trained by one SGD step per sample."""

    def __init__(self, d, rng, ratio=0.75, lr=0.05):
        k = max(1, int(np.ceil(ratio * d)))
        self.W = rng.normal(0.0, 1.0 / np.sqrt(d), (d, k))
        self.b = np.zeros(k)
        self.c = np.zeros(d)
        self.lr = lr

    def _fwd(self, x):
        h = np.tanh(x @ self.W + self.b)
        return h, h @ self.W.T + self.c

    def rmse(self, x):
        _, xr = self._fwd(x)
        return float(np.sqrt(np.mean((xr - x) ** 2)))

    def train(self, x):
        h, xr = self._fwd(x)
        e = (xr - x) / len(x)
        dpre = (e @ self.W) * (1.0 - h ** 2)
        self.W -= self.lr * (np.outer(e, h) + np.outer(x, dpre))
        self.b -= self.lr * dpre
        self.c -= self.lr * e
        return float(np.sqrt(np.mean((xr - x) ** 2)))

    def arrays(self, p):
        return {p + "W": self.W, p + "b": self.b, p + "c": self.c}

    @classmethod
    def from_arrays(cls, a, p, lr):
        o = cls.__new__(cls)
        o.W, o.b, o.c, o.lr = a[p + "W"].copy(), a[p + "b"].copy(), a[p + "c"].copy(), lr
        return o


def cluster_features(corr, m):
    """Greedy average-linkage grouping of features by correlation, groups of at most m."""
    d = corr.shape[0]
    groups = [[i] for i in range(d)]
    dist = 1.0 - np.abs(corr)
    while True:
        best, pair = 1e9, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if len(groups[a]) + len(groups[b]) > m:
                    continue
                dd = float(np.mean([dist[i, j] for i in groups[a] for j in groups[b]]))
                if dd < best:
                    best, pair = dd, (a, b)
        if pair is None or best > 0.9:
            break
        a, b = pair
        groups[a] = groups[a] + groups[b]
        del groups[b]
    return [sorted(g) for g in groups]


class KitEnsemble:
    PHASES = ("mapping", "training", "calibrating", "armed")

    def __init__(self, d=len(FEATURES), fm=120, ad=240, cal=240, m=6, lr=0.05, q=0.995, margin=1.5, seed=0, reservoir=4000):
        self.d, self.fm, self.ad, self.cal, self.m, self.lr, self.q, self.margin, self.seed = d, fm, ad, cal, m, lr, q, margin, seed
        self.cap = reservoir
        self.rng = np.random.default_rng(seed)
        self.n = 0
        self.phase = "mapping"
        self.norm = Running(d)
        self.sx = np.zeros(d)
        self.sxx = np.zeros((d, d))
        self.groups, self.aes, self.out = [], [], None
        self.rm = None
        self.scores = []
        self.thr = None
        self.thr0 = None
        self.since_thr = 0

    # ------------------------------------------------------------------ pipeline
    def _build(self):
        mu = self.sx / max(self.n, 1)
        cov = self.sxx / max(self.n, 1) - np.outer(mu, mu)
        sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        corr = cov / np.outer(sd, sd)
        corr[~np.isfinite(corr)] = 0.0
        self.groups = cluster_features(corr, self.m)
        self.aes = [OnlineAE(len(g), self.rng, lr=self.lr) for g in self.groups]
        self.out = OnlineAE(len(self.groups), self.rng, lr=self.lr)
        self.rm = Running(len(self.groups))
        self.sx = self.sxx = None

    def _group_rmse(self, z, train):
        return np.array([ae.train(z[g]) if train else ae.rmse(z[g]) for ae, g in zip(self.aes, self.groups)])

    def _score(self, z, train):
        r = self._group_rmse(z, train)
        if train:
            self.rm.update(r)
        zr = (r - self.rm.mu) / (self.rm.sd(floor=0.0) + 0.1 * np.abs(self.rm.mu) + 1e-6)
        zr = np.clip(zr, -ZCLIP, ZCLIP)
        s = self.out.train(zr) if train else self.out.rmse(zr)
        return s, r

    def process(self, x, learn=True):
        """Feed one window. Returns (score or None while still in childhood, info)."""
        x = np.asarray(x, np.float64)
        self.n += 1
        info = {"phase": self.phase}
        if self.phase == "mapping":
            self.norm.update(x)
            self.sx += x
            self.sxx += np.outer(x, x)
            if self.n >= self.fm:
                self._build()
                self.phase = "training"
            return None, info
        z = self.norm.z(x)
        if self.phase == "training":
            self.norm.update(x)
            self._score(z, train=True)
            if self.n >= self.fm + self.ad:
                self.phase = "calibrating"
            return None, info
        if self.phase == "calibrating":
            s, r = self._score(z, train=True)
            self.norm.update(x)
            self._keep(s)
            if self.n >= self.fm + self.ad + self.cal:
                self._set_threshold()
                self.thr0 = self.thr
                self.phase = "armed"
            return None, info
        s, r = self._score(z, train=False)
        info.update(score=s, threshold=self.thr, z=z, group_err=r)
        normal = s < 0.9 * self.thr
        if learn and normal:
            self._score(z, train=True)
            self.norm.update(x)
            self._keep(s)
            self.since_thr += 1
            if self.since_thr >= 100:
                self._set_threshold()
        return s, info

    def teach_normal(self, x, repeat=5):
        """A human said 'this was fine': learn it, whatever it scored."""
        x = np.asarray(x, np.float64)
        if self.phase in ("mapping",):
            return
        self.norm.update(x)
        z = self.norm.z(x)
        for _ in range(repeat):
            self._score(z, train=True)
        s, _ = self._score(z, train=False)
        self._keep(s)

    def _keep(self, s):
        self.scores.append(float(s))
        if len(self.scores) > self.cap:
            del self.scores[: len(self.scores) - self.cap]

    def _set_threshold(self):
        base = float(np.quantile(self.scores, self.q)) * self.margin
        floor = 0.5 * self.thr0 if self.thr0 else 1e-3
        self.thr = max(base, floor, MIN_THR)
        self.since_thr = 0

    def explain(self, info, raw, top=3):
        """Plain-language reasons: which signals were far from their usual level."""
        z = info["z"]
        order = np.argsort(-np.abs(z))[:top]
        out = []
        for j in order:
            if abs(z[j]) < 1.5:
                break
            name = FEATURES[j]
            usual = float(np.expm1(self.norm.mu[j])) if name != "dns_entropy" else float(self.norm.mu[j] * 4.0)
            out.append(f"{DESCRIPTIONS[name]}: {raw.get(name, 0):.4g} (usually about {usual:.2g})")
        return out

    # ------------------------------------------------------------------ persistence (no pickle)
    def to_arrays(self):
        a = {"n": np.array(self.n), "phase": np.array(self.PHASES.index(self.phase)), "norm_n": np.array(self.norm.n), "norm_mu": self.norm.mu,
             "norm_m2": self.norm.m2, "scores": np.array(self.scores, np.float64),
             "thr": np.array(-1.0 if self.thr is None else self.thr), "thr0": np.array(-1.0 if self.thr0 is None else self.thr0),
             "since_thr": np.array(self.since_thr), "cfg": np.frombuffer(json.dumps(
                 dict(d=self.d, fm=self.fm, ad=self.ad, cal=self.cal, m=self.m, lr=self.lr, q=self.q, margin=self.margin, seed=self.seed, cap=self.cap)).encode(), np.uint8)}
        if self.phase == "mapping":
            a["sx"], a["sxx"] = self.sx, self.sxx
        else:
            a["groups"] = np.frombuffer(json.dumps(self.groups).encode(), np.uint8)
            for i, ae in enumerate(self.aes):
                a.update(ae.arrays(f"ae{i}_"))
            a.update(self.out.arrays("out_"))
            a["rm_n"], a["rm_mu"], a["rm_m2"] = np.array(self.rm.n), self.rm.mu, self.rm.m2
        return a

    @classmethod
    def from_arrays(cls, a):
        cfg = json.loads(bytes(a["cfg"].astype(np.uint8)).decode())
        cap = cfg.pop("cap")
        o = cls(reservoir=cap, **cfg)
        o.n, o.phase = int(a["n"]), cls.PHASES[int(a["phase"])]
        o.norm.n, o.norm.mu, o.norm.m2 = int(a["norm_n"]), a["norm_mu"].copy(), a["norm_m2"].copy()
        o.scores = [float(v) for v in a["scores"]]
        thr, thr0 = float(a["thr"]), float(a["thr0"])
        o.thr, o.thr0 = (None if thr < 0 else thr), (None if thr0 < 0 else thr0)
        o.since_thr = int(a["since_thr"])
        if o.phase == "mapping":
            o.sx, o.sxx = a["sx"].copy(), a["sxx"].copy()
        else:
            o.groups = json.loads(bytes(a["groups"].astype(np.uint8)).decode())
            o.aes = [OnlineAE.from_arrays(a, f"ae{i}_", o.lr) for i in range(len(o.groups))]
            o.out = OnlineAE.from_arrays(a, "out_", o.lr)
            o.rm = Running(len(o.groups))
            o.rm.n, o.rm.mu, o.rm.m2 = int(a["rm_n"]), a["rm_mu"].copy(), a["rm_m2"].copy()
            o.sx = o.sxx = None
        return o


# ---------------------------------------------------------------------------------------- Forge neuron
class TinyMLP:
    """One hidden layer, tanh, sigmoid output; trained with Adam on full batches. Small enough to read, fast enough for a phone."""

    def __init__(self, d, h=16, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(d), (d, h)); self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, 1 / np.sqrt(h), (h, 1)); self.b2 = np.zeros(1)

    def predict(self, X):
        X = np.atleast_2d(X)
        H = np.tanh(X @ self.W1 + self.b1)
        return 1.0 / (1.0 + np.exp(-np.clip((H @ self.W2 + self.b2)[:, 0], -30, 30)))

    def fit(self, X, y, sample_weight=None, epochs=400, lr=0.02, l2=1e-3):
        X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, np.float64)
        w = w / w.sum()
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        for t in range(1, epochs + 1):
            H = np.tanh(X @ self.W1 + self.b1)
            p = 1.0 / (1.0 + np.exp(-np.clip((H @ self.W2 + self.b2)[:, 0], -30, 30)))
            g = (p - y) * w
            gW2 = H.T @ g[:, None] + l2 * self.W2
            gb2 = g.sum(keepdims=True)
            gH = (g[:, None] @ self.W2.T) * (1 - H ** 2)
            gW1 = X.T @ gH + l2 * self.W1
            gb1 = gH.sum(0)
            for i, (p_, g_) in enumerate(zip(params, (gW1, gb1, gW2, gb2))):
                m[i] = 0.9 * m[i] + 0.1 * g_
                v[i] = 0.999 * v[i] + 0.001 * g_ ** 2
                p_ -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        return self

    def arrays(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    @classmethod
    def from_arrays(cls, a):
        o = cls.__new__(cls)
        o.W1, o.b1, o.W2, o.b2 = (a[k].copy() for k in ("W1", "b1", "W2", "b2"))
        return o

    @property
    def n_params(self):
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)


class ForgeNeuron:
    """A trained, approved (or proposed) neuron: standardiser fitted to the machine's own normal windows + a TinyMLP + a decision threshold."""

    def __init__(self, name, mlp, mu, sd, thr, spec=None, version=1):
        self.name, self.mlp, self.mu, self.sd, self.thr, self.spec, self.version = name, mlp, np.asarray(mu), np.asarray(sd), float(thr), spec or {}, version

    def z(self, vec):
        return np.clip((np.asarray(vec, np.float64) - self.mu) / self.sd, -8.0, 8.0)

    def score(self, vec):
        return float(self.mlp.predict(self.z(vec))[0])

    def score_many(self, V):
        return self.mlp.predict(np.clip((np.asarray(V) - self.mu) / self.sd, -8.0, 8.0))

    def explain(self, vec, raw, top=3):
        z = self.z(vec)
        out = []
        for j in np.argsort(-np.abs(z))[:top]:
            if abs(z[j]) < 1.5:
                break
            out.append(f"{DESCRIPTIONS[FEATURES[j]]}: {raw.get(FEATURES[j], 0):.4g}")
        return out

    def to_blob(self):
        from lkm.live.store import pack
        a = self.mlp.arrays()
        a.update(mu=self.mu, sd=self.sd, thr=np.array(self.thr))
        return pack(a)

    @classmethod
    def from_blob(cls, blob, name, spec=None, version=1):
        from lkm.live.store import unpack_blob
        a = unpack_blob(blob)
        return cls(name, TinyMLP.from_arrays(a), a["mu"], a["sd"], float(a["thr"]), spec, version)
