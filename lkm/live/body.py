"""The organism's body for a long-lived life.

Differences from the game version (`lkm.organism.Organism`):
  * workers live on entity IDs, not array positions, so the world may change between cycles (entities appear and vanish);
  * everything that makes the individual (genomes, brains, fast weights, energy, scent, readout, rng) can be exported and restored;
  * rewards arrive whenever an answer arrives, not at the end of a round: each pick carries the context (brain state, input)
    it was made with, and credit goes to the worker only if that same individual is still alive (unique ids);
  * energy leaks and scent fades once per cycle (the passing of time), not once per game round.
"""
import json
import numpy as np
from .. import nca as N
from ..baselines import fit_offset_logreg
from ..organism import Organism, unpack, sigmoid, HD


def _softplus(x):
    return np.log1p(np.exp(x))


class LiveOrganism(Organism):
    def __init__(self, theta, P, vocab=0, seed=0, **kw):
        super().__init__(theta, P=P, base=None, vocab=vocab, seed=seed, **kw)
        self.uid = None
        self.next_uid = 0
        self.ids = None
        self.pos_ids = None
        self.hit_d, self.vis_d = {}, {}
        self.generation_births = 0

    # ------------------------------------------------------------------ birth and attachment to the current world
    def birth(self, env, ids, base):
        self.reset(env)                                   # genomes from the ancestral genome, random positions, empty memory
        self.uid = np.arange(self.W)
        self.next_uid = int(self.W)
        self.attach(env, ids, base, fresh=True)

    def attach(self, env, ids, base, fresh=False):
        """(Re)bind to the current snapshot of the world. Positions and scent are carried over by entity id."""
        old_ids = self.ids
        self.n, self.G, self.ids = env.n, env.G, np.asarray(ids)
        self.Xn = np.nan_to_num(env.ds.X)
        self.tabs = N.tables(self.G) if (self.use_cells and self.P is not None) else None
        age = env.ds.X[:, env.ds.age_col]
        self.age_n = np.clip(np.nan_to_num(age), -3, 3).astype(np.float32) / 3.0
        self.deg_n = (np.log1p(self.G.deg.sum(1)) / 3.0).astype(np.float32)
        self.base = np.asarray(base, np.float32)
        self.pb = (np.clip(self.base, -6, 6) / 3.0).astype(np.float32)
        self._cache = None
        index = {e: i for i, e in enumerate(self.ids.tolist())}
        if fresh:
            self.pos_ids = self.ids[self.pos].copy()
        else:                                             # carry the individuals to the same places in the new snapshot
            pos = np.array([index.get(e, -1) for e in self.pos_ids.tolist()])
            lost = pos < 0
            if lost.any():
                pos[lost] = self.rng.integers(0, self.n, int(lost.sum()))
            self.pos = pos
        self.hit = np.zeros(self.n, np.float32); self.vis = np.zeros(self.n, np.float32)
        for d, arr in ((self.hit_d, self.hit), (self.vis_d, self.vis)):
            for e, v in d.items():
                i = index.get(e)
                if i is not None:
                    arr[i] = v

    def sync_dicts(self):
        """Move the scent arrays back into id-keyed dicts (sparse) and the positions into ids."""
        if self.ids is None or not hasattr(self, "hit"):      # restored but not yet attached to a world: dicts are already current
            return
        nz = np.where(self.hit > 1e-4)[0]
        self.hit_d = {self.ids[i]: float(self.hit[i]) for i in nz}
        nz = np.where(self.vis > 1e-4)[0]
        self.vis_d = {self.ids[i]: float(self.vis[i]) for i in nz}
        self.pos_ids = self.ids[self.pos].copy()

    # ------------------------------------------------------------------ the passing of time (once per cycle)
    def tick_world(self):
        self.energy *= 0.9
        if self.use_scent:
            self.hit = (0.85 * (0.5 * self.hit + 0.5 * (self.G.Aun @ self.hit))).astype(np.float32)   # diffuse and fade, never amplifies
            self.vis = (0.97 * self.vis).astype(np.float32)

    # ------------------------------------------------------------------ rewards that arrive whenever answers arrive
    def credit(self, items):
        """items: dicts with worker, uid, entity_idx (or None), label, h_final, x. Returns the number of credited workers."""
        pr = unpack(self.genomes, self.V)
        n_ok = 0
        for it in items:
            w, lab = it.get("worker"), int(it["label"])
            if w is not None and 0 <= w < self.W and self.uid[w] == it.get("uid") and it.get("x") is not None:
                r = self.r_hit if lab == 1 else self.r_miss
                self.energy[w] += r
                self.last_r[w] = r
                if self.plastic:
                    eta = _softplus(pr["misc"][w, 1]) * 0.5
                    lam = sigmoid(pr["misc"][w, 2])
                    self.Wf[w] = np.clip(lam * self.Wf[w] + eta * r * np.outer(np.asarray(it["h_final"]), np.asarray(it["x"])), -2, 2)
                n_ok += 1
            i = it.get("entity_idx")
            if self.use_scent and i is not None:
                if lab == 1:
                    self.hit[i] += 1.0
                self.vis[i] += 1.0
        return n_ok

    def refresh_cells(self, env):
        """Relax the cell field with everything verified so far and adapt its readout to the labels."""
        self._cache = None
        if self.use_cells and self.P is not None:
            self._cell_logit(env)
            if self.cell_adapt:
                self._adapt_cells(env)

    def evolve_live(self):
        before = self.genomes.copy()
        k = self._evolve() if self.evolve_on else 0
        changed = np.any(before != self.genomes, axis=1)
        if changed.any():                                 # new individuals get new identities
            m = int(changed.sum())
            self.uid[changed] = np.arange(self.next_uid, self.next_uid + m)
            self.next_uid += m
            self.generation_births += m
        return int(changed.sum())

    # ------------------------------------------------------------------ choosing what to check
    def choose(self, env, B):
        """One round of foraging; returns a list of picks with the context each one was made with."""
        idx = self.pick(env, B)
        picks = []
        for w, i in zip(self._sel.tolist(), np.asarray(idx).tolist()):
            picks.append({"worker": int(w), "uid": int(self.uid[w]), "entity_idx": int(i),
                          "risk": float(sigmoid(self._z[i])),
                          "ctx": json.dumps({"h_final": np.round(self._h_final[w], 6).tolist(), "x": np.round(self._x[w], 6).tolist()})})
        self.sync_dicts()
        return picks

    # ------------------------------------------------------------------ persistence
    def export(self):
        self.sync_dicts()
        ro = self.readout
        out = {
            "genomes": self.genomes, "h": self.h, "Wf": self.Wf, "energy": self.energy, "last_r": self.last_r, "uid": self.uid,
            "theta": self.theta, "next_uid": np.array([self.next_uid]), "V": np.array([self.V]), "W": np.array([self.W]),
            "pos_ids": np.array(json.dumps(self.pos_ids.tolist())),
            "hit_ids": np.array(json.dumps(list(self.hit_d.keys()))), "hit_vals": np.array(list(self.hit_d.values()), np.float64),
            "vis_ids": np.array(json.dumps(list(self.vis_d.keys()))), "vis_vals": np.array(list(self.vis_d.values()), np.float64),
            "rng": np.array(json.dumps(self.rng.bit_generator.state, default=int)),
            "log": np.array(json.dumps({k: v[-500:] for k, v in self.log.items()})), "births": np.array([self.generation_births]),
        }
        if ro is not None:
            out["readout_w"], out["readout_b"] = np.asarray(ro[0]), np.asarray(ro[1])
        return out

    def restore(self, z):
        self.genomes, self.h, self.Wf = z["genomes"].copy(), z["h"].copy(), z["Wf"].copy()
        self.energy, self.last_r, self.uid = z["energy"].copy(), z["last_r"].copy(), z["uid"].copy()
        self.theta = z["theta"].copy()
        self.next_uid = int(z["next_uid"][0]); self.W = int(z["W"][0]); self.V = int(z["V"][0])
        self.pos_ids = np.array(json.loads(str(z["pos_ids"])))
        self.hit_d = dict(zip(json.loads(str(z["hit_ids"])), z["hit_vals"].tolist()))
        self.vis_d = dict(zip(json.loads(str(z["vis_ids"])), z["vis_vals"].tolist()))
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = json.loads(str(z["rng"]))
        self.log = json.loads(str(z["log"]))
        self.generation_births = int(z["births"][0])
        self.readout = (z["readout_w"].copy(), z["readout_b"].copy()) if "readout_w" in z else None
