"""The organism: a population of artificial individuals ("workers") that live on the mesh.

Body    each worker has a small recurrent brain (16 cells) whose wiring is its genome (~800 numbers).
Senses  at its node it perceives a descriptor: cell-field risk, baseline risk, scent (hit/visited), whether the
        node is already verified, age, degree, and neighbourhood averages. It also feels its own energy.
Actions move along an edge (or jump to a random patch), and decide whether to ask for verification here.
Purpose a verification costs energy; a confirmed positive pays energy. Workers that starve are replaced by mutated
        copies of workers that thrive (evolution at run time). Reward-modulated Hebbian 'fast weights' let each
        brain adapt within its lifetime (plasticity).
Mesh    successful discoveries leave scent that diffuses to neighbouring nodes and fades; the cell field (NCA)
        relaxes around what has been verified.
Switches (for ablations): use_cells, use_scent, plastic, evolve.
"""
import numpy as np
from .baselines import fit_offset_logreg
from . import nca as N
from .graph import NSLOT

HD, DD0 = 16, 10


def dims(V=0):
    """Descriptor sizes. V > 0: the worker also hears the word the cell field is 'saying' at its node (one-hot over V words)."""
    dd = DD0 + V
    return dd, dd + 3, dd + 4


def shapes(V=0):
    DD_, DC_, DIN_ = dims(V)
    return [("Wxh", (HD, DIN_)), ("Whh", (HD, HD)), ("bh", (HD,)), ("wi", (HD + 1,)), ("wd", (DD_,)), ("Wm", (DC_, HD + 1)),
            ("wj", (HD + 1,)), ("misc", (4,))]          # misc: log-temperature, plasticity rate, trace decay, trust in the risk field


def p_total(V=0):
    return int(sum(int(np.prod(s)) for _, s in shapes(V)))


DD, DC, DIN = dims(0)
SHAPES = shapes(0)
SIZES = [int(np.prod(s)) for _, s in SHAPES]
P_TOTAL = p_total(0)


def unpack(G, V=0):
    """G: (...,P) -> dict of arrays (...,*shape)."""
    out, o = {}, 0
    for name, shp in shapes(V):
        sz = int(np.prod(shp))
        out[name] = G[..., o:o + sz].reshape(G.shape[:-1] + shp)
        o += sz
    return out


def init_genome(seed=0, V=0):
    r = np.random.default_rng(seed)
    DD_, DC_, DIN_ = dims(V)
    parts = {"Wxh": r.normal(0, 0.3, (HD, DIN_)), "Whh": r.normal(0, 0.2, (HD, HD)), "bh": np.zeros(HD),
             "wi": np.zeros(HD + 1), "wd": np.zeros(DD_), "Wm": np.zeros((DC_, HD + 1)), "wj": np.zeros(HD + 1),
             "misc": np.array([0.0, -2.0, 2.0, 1.0])}
    # instincts: seek high estimated risk, avoid verified nodes, mild preference for scent
    parts["wd"][[0, 1]] = [1.5, 1.5]; parts["wd"][[4, 5]] = [-8.0, -8.0]
    parts["Wm"][0, HD] = 1.5; parts["Wm"][1, HD] = 1.0; parts["Wm"][4, HD] = -5.0; parts["Wm"][5, HD] = -5.0
    parts["Wm"][2, HD] = 0.5
    parts["wj"][HD] = -2.0
    return np.concatenate([parts[n].ravel() for n, _ in shapes(V)]).astype(np.float64)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Organism:
    """Policy-compatible organism (see game.Policy)."""
    name = "organism"

    def __init__(self, theta, P=None, base=None, miss=None, use_cells=True, use_scent=True, plastic=True, evolve=True,
                 cell_adapt=True, adapt_l2=40.0, ticks=4, workers_per_slot=4, mut_sigma=0.03, replace_frac=0.15, jump_cands=24, field_beta=1.5, informed=True,
                 r_hit=1.0, r_miss=-0.15, name=None, record=False, seed=0, vocab=0):
        self.theta = np.asarray(theta, np.float64)
        self.V = int(vocab)
        self.DD, self.DC, self.DIN = dims(self.V)
        self.PT = p_total(self.V)
        self.P, self.base, self.miss = P, base, miss
        self.use_cells, self.use_scent, self.plastic, self.evolve_on, self.cell_adapt = use_cells, use_scent, plastic, evolve, cell_adapt
        self.adapt_l2 = adapt_l2
        self.field_beta, self.informed = field_beta, informed
        self.ticks, self.wps, self.mut, self.repl, self.M = ticks, workers_per_slot, mut_sigma, replace_frac, jump_cands
        self.r_hit, self.r_miss = r_hit, r_miss
        if name:
            self.name = name
        self.record, self.rec = record, []
        self.seed = seed
        self.log = {"births": [], "deaths": [], "energy": [], "jump_rate": [], "unused": []}

    # ------------------------------------------------------------------ lifecycle
    def reset(self, env):
        self.rng = np.random.default_rng(self.seed + 7919 * (env.rng.integers(1 << 20)))
        self.n, self.G = env.n, env.G
        self.Xn = np.nan_to_num(env.ds.X)
        self.tabs = N.tables(self.G) if (self.use_cells and self.P is not None) else None
        self.W = int(max(8, self.wps * int(env.B.max())))
        self.genomes = self.theta[None, :] + self.rng.normal(0, 0.02, (self.W, self.PT))
        self.pos = self.rng.integers(0, self.n, self.W)
        self.h = np.zeros((self.W, HD)); self.Wf = np.zeros((self.W, HD, self.DIN)); self.energy = np.zeros(self.W)
        self.last_r = np.zeros(self.W)
        self.hit = np.zeros(self.n, np.float32); self.vis = np.zeros(self.n, np.float32)
        age = env.ds.X[:, env.ds.age_col]
        self.age_n = np.clip(np.nan_to_num(age), -3, 3).astype(np.float32) / 3.0
        deg = self.G.deg.sum(1)
        self.deg_n = (np.log1p(deg) / 3.0).astype(np.float32)
        self.readout = None
        self._cache = None
        if self.base is None:
            self.base = np.zeros(self.n, np.float32)
        self.pb = (np.clip(self.base, -6, 6) / 3.0).astype(np.float32)

    def stats(self):
        return {k: [float(x) for x in v] for k, v in self.log.items()}

    # ------------------------------------------------------------------ mesh perception
    def _cell_logit(self, env):
        if not (self.use_cells and self.P is not None):
            return self.base
        if self._cache is None:
            inp = N.build_inp(self.Xn, self.base, env.rev_pos, env.rev_neg, self.miss)
            self._cache = N.forward(self.P, inp, self.tabs)[1]
        s = self._cache
        wo, bo = (self.P["wo"], self.P["bo"]) if self.readout is None else self.readout
        return self.base + s @ wo + bo

    def _descriptors(self, env):
        z = self._cell_logit(env)
        zc = (np.clip(z, -6, 6) / 3.0).astype(np.float32)
        d = np.zeros((self.n, self.DD), np.float32)
        d[:, 0] = zc if self.use_cells else self.pb
        d[:, 1] = self.pb
        if self.use_scent:
            d[:, 2] = np.minimum(self.hit * 3.0, 3.0); d[:, 3] = np.minimum(self.vis, 3.0)
        d[:, 4] = (env.verified == 2); d[:, 5] = (env.verified == 1)
        d[:, 6] = self.age_n; d[:, 7] = self.deg_n
        d[:, 8] = self.G.Aun @ d[:, 0]
        if self.use_scent:
            d[:, 9] = np.minimum(self.G.Aun @ self.hit * 3.0, 3.0)
        if self.V and self._cache is not None and self.P is not None and "Wt" in self.P:
            d[np.arange(self.n), DD0 + np.minimum(N.words(self.P, self._cache), self.V - 1)] = 1.0     # the word the cell field says here
        return d, z

    def _field_cdf(self, z, env):
        """Shared 'risk chemistry': a sampling distribution over unverified nodes, proportional to exp(beta * logit)."""
        zz = np.clip(np.asarray(z, np.float64), -8, 8)
        w = np.exp(self.field_beta * (zz - zz.max()))
        w = np.where(env.verified == 0, w, 0.0)
        c = np.cumsum(w)
        self._cdf = c / max(c[-1], 1e-300)

    # ------------------------------------------------------------------ brains and movement
    def _think(self, pr, d_all, round_frac):
        d_pos = d_all[self.pos]                                       # (W,DD)
        x = np.concatenate([d_pos, np.tanh(self.energy)[:, None], self.last_r[:, None],
                            np.full((self.W, 1), round_frac), np.ones((self.W, 1))], 1)    # (W,DIN)
        W_eff = pr["Wxh"] + self.Wf if self.plastic else pr["Wxh"]
        pre = np.einsum("wij,wj->wi", W_eff, x) + np.einsum("wij,wj->wi", pr["Whh"], self.h) + pr["bh"]
        self.h = np.tanh(pre)
        self._x = x
        return np.concatenate([self.h, np.ones((self.W, 1))], 1)      # h_aug

    def _move(self, pr, h_aug, d_all):
        n, W = self.n, self.W
        m = np.einsum("wcd,wd->wc", pr["Wm"], h_aug)                  # preference vector in candidate space
        temp = np.exp(np.clip(pr["misc"][:, 0], -2, 2))[:, None]
        # neighbour candidates
        nb = self.G.nbr[self.pos]; rel = self.G.nbr_rel[self.pos]     # (W,NSLOT)
        valid = nb >= 0
        cd = d_all[np.maximum(nb, 0)]                                 # (W,NSLOT,DD)
        oh = np.zeros((W, NSLOT, 3), np.float32)
        for r in range(3):
            oh[:, :, r] = (rel == r)
        c = np.concatenate([cd, oh], 2)
        sc = np.einsum("wkc,wc->wk", c, m) / temp
        sc = np.where(valid, sc, -1e9)
        # jump candidates
        jc = self.rng.integers(0, n, (W, self.M))
        if self.informed:
            trust = sigmoid(pr["misc"][:, 3])[:, None]                     # share of candidates drawn from the risk field
            biased = np.minimum(np.searchsorted(self._cdf, self.rng.random((W, self.M))), n - 1)
            jc = np.where(self.rng.random((W, self.M)) < trust, biased, jc)
        cj = np.concatenate([d_all[jc], np.zeros((W, self.M, 3), np.float32)], 2)
        sj = np.einsum("wkc,wc->wk", cj, m) / temp
        p_jump = sigmoid(np.einsum("wd,wd->w", pr["wj"], h_aug))
        do_jump = (self.rng.random(W) < p_jump) | (~valid.any(1))
        K = max(NSLOT, self.M)

        def padw(a, fill):
            out = np.full((W, K), fill, dtype=a.dtype)
            out[:, :a.shape[1]] = a
            return out
        s_all = np.where(do_jump[:, None], padw(sj, -1e9), padw(sc, -1e9))
        ids = np.where(do_jump[:, None], padw(jc, 0), padw(np.maximum(nb, 0), 0))
        vmask = np.where(do_jump[:, None], padw(np.ones((W, self.M), bool), False), padw(valid, False))
        s_all = np.where(vmask, s_all, -1e9)
        s_all = s_all - s_all.max(1, keepdims=True)
        pk = np.where(vmask, np.exp(s_all), 0.0)
        pk /= np.maximum(pk.sum(1, keepdims=True), 1e-12)
        u = self.rng.random(W)[:, None]
        k = (pk.cumsum(1) < u).sum(1).clip(0, ids.shape[1] - 1)
        self.pos = ids[np.arange(W), k]
        return do_jump.mean()

    # ------------------------------------------------------------------ round
    def pick(self, env, B):
        pr = unpack(self.genomes, self.V)
        d_all, z = self._descriptors(env)
        self._field_cdf(z, env)
        rf = env.round / max(env.rounds - 1, 1)
        jumps = []
        traj = [self.pos.copy()] if self.record else None
        for _ in range(self.ticks):
            h_aug = self._think(pr, d_all, rf)
            jumps.append(self._move(pr, h_aug, d_all))
            self.energy -= 0.002
            if self.record:
                traj.append(self.pos.copy())
        h_aug = self._think(pr, d_all, rf)
        score = np.einsum("wd,wd->w", pr["wi"], h_aug)
        score = score + np.einsum("wd,wd->w", pr["wd"], d_all[self.pos])
        ok = env.verified[self.pos] == 0
        score = np.where(ok, score, -1e9)
        order = np.lexsort((-score, self.pos))                       # best worker per node
        first = np.ones(len(order), bool); first[1:] = self.pos[order][1:] != self.pos[order][:-1]
        cand_w = order[first]
        cand_w = cand_w[score[cand_w] > -1e8]
        if len(cand_w) > B:
            cand_w = cand_w[np.argsort(-score[cand_w])[:B]]
        self._sel = cand_w
        self._ids = self.pos[cand_w]
        self._h_final = h_aug[:, :HD].copy()
        self.log["jump_rate"].append(float(np.mean(jumps)))
        self.log["unused"].append(int(B - len(cand_w)))
        self._traj, self._d_all, self._z = traj, d_all, z
        return self._ids

    def observe(self, env, ids, labels):
        W = self.W
        r = np.zeros(W)
        r[self._sel] = np.where(labels == 1, self.r_hit, self.r_miss)
        self.last_r = r
        self.energy = 0.9 * self.energy + r
        pr = unpack(self.genomes, self.V)
        if self.plastic:
            eta = np.log1p(np.exp(pr["misc"][:, 1]))[:, None, None] * 0.5
            lam = sigmoid(pr["misc"][:, 2])[:, None, None]
            dW = r[:, None, None] * np.einsum("wi,wj->wij", self._h_final, self._x)
            self.Wf = np.clip(lam * self.Wf + eta * dW, -2, 2)
        if self.use_scent:
            if len(ids):
                pos_ids = ids[labels == 1]
                self.hit[pos_ids] += 1.0
                self.vis[ids] += 1.0
            self.hit = (0.85 * (0.5 * self.hit + 0.5 * (self.G.Aun @ self.hit))).astype(np.float32)   # diffuse + fade (never amplifies)
            self.vis = (0.97 * self.vis).astype(np.float32)
        self._cache = None
        if self.use_cells and self.P is not None:
            self._cache = N.forward(self.P, N.build_inp(self.Xn, self.base, env.rev_pos, env.rev_neg, self.miss), self.tabs)[1]
        if self.cell_adapt and self.use_cells and self.P is not None:
            self._adapt_cells(env)
        births = deaths = 0
        if self.evolve_on:
            births = deaths = self._evolve()
        self.log["births"].append(births); self.log["deaths"].append(deaths)
        self.log["energy"].append(float(self.energy.mean()))
        if self.record:
            self.rec.append({"traj": [t.tolist() for t in self._traj], "ids": ids.tolist(), "labels": labels.tolist(),
                             "workers": self._sel.tolist(),
                             "hit": np.round(self.hit, 3).tolist() if self.n <= 1500 else []})

    def _adapt_cells(self, env):
        rev = np.where(env.verified > 0)[0]
        if len(rev) < 60 or env.verified[rev].min() == env.verified[rev].max():
            return
        s = self._cache
        d0 = s @ self.P["wo"] + self.P["bo"]
        w = fit_offset_logreg(s[rev], (env.verified[rev] == 2).astype(float), (self.base + d0)[rev], l2=self.adapt_l2)
        self.readout = (self.P["wo"] + w[:-1].astype(np.float32), self.P["bo"] + np.float32(w[-1]))

    def _evolve(self):
        W = self.W
        k = max(1, int(self.repl * W))
        order = np.argsort(self.energy)
        dead = order[:k]
        top = order[-max(k * 2, 2):]
        w = np.exp((self.energy[top] - self.energy[top].max()) * 3.0)
        parents = self.rng.choice(top, k, p=w / w.sum())
        self.genomes[dead] = self.genomes[parents] + self.rng.normal(0, self.mut, (k, self.PT))
        near = self.rng.random(k) < 0.5
        self.pos[dead] = np.where(near, self.pos[parents], self.rng.integers(0, self.n, k))
        self.h[dead] = 0; self.Wf[dead] = 0; self.energy[dead] = self.energy.mean(); self.last_r[dead] = 0
        return int(k)
