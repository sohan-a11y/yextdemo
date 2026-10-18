"""Gradient training (JAX, back-propagation through the T update steps) of the graph NCA."""
import time
import numpy as np
import jax, jax.numpy as jnp
from . import nca as N

jax.config.update("jax_enable_x64", False)


def _forward_jax(P, inp, tabs, key, T, train):
    s = jnp.tanh(inp @ P["We"] + P["be"])
    keys = jax.random.split(key, T)

    def step(s, k):
        z = jnp.concatenate([s] + [
            (s[i] * m[..., None]).sum(1) / jnp.maximum(m.sum(1, keepdims=True), 1.0) for i, m in tabs], 1)
        h = jnp.tanh(z @ P["W1"] + P["b1"])
        ds = N.ALPHA * (h @ P["W2"] + P["b2"])
        if train:
            upd = (jax.random.uniform(k, (s.shape[0], 1)) < N.KEEP).astype(jnp.float32)
        else:
            upd = N.KEEP
        return s + upd * ds, None

    s, _ = jax.lax.scan(step, s, keys)
    return s @ P["wo"] + P["bo"], s


def _pad(a, n_to, fill=0):
    if len(a) == n_to:
        return a
    pad = [(0, n_to - len(a))] + [(0, 0)] * (a.ndim - 1)
    return np.pad(a, pad, constant_values=fill)


def _prepare(ds, base, bucket=2048):
    n = ds.n
    nn = int(np.ceil(n / bucket) * bucket)
    tabs = N.tables_idx(ds.graph())
    return {"n": n, "nn": nn, "X": _pad(np.nan_to_num(ds.X).astype(np.float32), nn), "base": _pad(base.astype(np.float32), nn),
            "y": _pad(ds.y.astype(np.float32), nn), "valid": _pad(np.ones(n, np.float32), nn),
            "tabs": [(jnp.asarray(_pad(i, nn)), jnp.asarray(_pad(m, nn))) for i, m in tabs]}


def make_loss(T, lam):
    def loss(P, key, inp, base, y, valid, rev_mask, tabs):
        d, _ = _forward_jax(P, inp, tabs, key, T, True)
        z = base + d
        bce = jnp.maximum(z, 0) - z * y + jnp.log1p(jnp.exp(-jnp.abs(z)))
        w = valid * (1.0 - rev_mask)
        return (bce * w).sum() / jnp.maximum(w.sum(), 1.0) + lam * ((d ** 2) * w).sum() / jnp.maximum(w.sum(), 1.0)
    return loss


def evaluate(P, ds, base, T=N.T_DEFAULT):
    inp = N.build_inp(ds.X, base)
    d, _ = N.forward({k: np.asarray(v) for k, v in P.items()}, inp, N.tables(ds.graph()), T)
    z = base + d
    y = ds.y
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def train(train_sets, base_list, val=None, val_base=None, steps=600, lr=3e-3, seed=0, T=N.T_DEFAULT,
          C=N.C_DEFAULT, H=N.H_DEFAULT, lam=1e-3, log=print, eval_every=100, miss_aug=True, clip=1.0):
    rng = np.random.default_rng(seed)
    din = train_sets[0].X.shape[1] + 4
    P = {k: jnp.asarray(v) for k, v in N.init_params(din, C, H, seed).items()}
    prep = [_prepare(d, b) for d, b in zip(train_sets, base_list)]
    loss_fn = make_loss(T, lam)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    m = jax.tree_util.tree_map(jnp.zeros_like, P); v = jax.tree_util.tree_map(jnp.zeros_like, P)
    best, best_P, t0 = 1e9, P, time.time()
    hist = []
    for it in range(1, steps + 1):
        pr = prep[rng.integers(len(prep))]
        n, nn = pr["n"], pr["nn"]
        y = pr["y"]
        if rng.random() < 0.5:        # reveals as they happen in the game: the highest-ranked entities get verified
            k = max(1, int(rng.uniform(0.005, 0.05) * n))
            sel = np.argpartition(-pr["base"][:n], k - 1)[:k]
            top = np.zeros(nn, bool); top[sel] = True
            rp = ((y > 0.5) & top).astype(np.float32); rn = ((y < 0.5) & top).astype(np.float32)
        else:
            qp, qn = rng.uniform(0, 0.4), rng.uniform(0, 0.08)
            rp = ((y > 0.5) & (rng.random(nn) < qp)).astype(np.float32) * pr["valid"]
            rn = ((y < 0.5) & (rng.random(nn) < qn)).astype(np.float32) * pr["valid"]
        X = pr["X"]; miss = np.zeros(nn, np.float32)
        if miss_aug and rng.random() < 0.5:
            p = rng.uniform(0.05, 0.5)
            keep = (rng.random(X.shape) >= p).astype(np.float32)
            X = X * keep; miss = (1.0 - keep).mean(1)
        b = np.clip(pr["base"], -6, 6) / 3.0
        inp = np.concatenate([X, b[:, None], rp[:, None], rn[:, None], miss[:, None]], 1).astype(np.float32)
        key = jax.random.PRNGKey(int(rng.integers(1 << 30)))
        l, g = grad_fn(P, key, jnp.asarray(inp), jnp.asarray(pr["base"]), jnp.asarray(y), jnp.asarray(pr["valid"]),
                       jnp.asarray(np.maximum(rp, rn)), pr["tabs"])
        gn = jnp.sqrt(sum((x ** 2).sum() for x in jax.tree_util.tree_leaves(g)))
        sc = jnp.minimum(1.0, clip / (gn + 1e-9))
        g = jax.tree_util.tree_map(lambda x: x * sc, g)
        m = jax.tree_util.tree_map(lambda a, b_: 0.9 * a + 0.1 * b_, m, g)
        v = jax.tree_util.tree_map(lambda a, b_: 0.999 * a + 0.001 * b_ ** 2, v, g)
        mh = jax.tree_util.tree_map(lambda a: a / (1 - 0.9 ** it), m); vh = jax.tree_util.tree_map(lambda a: a / (1 - 0.999 ** it), v)
        P = jax.tree_util.tree_map(lambda p_, a, b_: p_ - lr * a / (jnp.sqrt(b_) + 1e-8), P, mh, vh)
        hist.append(float(l))
        if it % eval_every == 0 or it == steps:
            msg = f"[nca] step {it}/{steps} train-loss {np.mean(hist[-eval_every:]):.4f}"
            if val is not None:
                vl = evaluate(P, val, val_base, T)
                msg += f" val-bce {vl:.4f}"
                if vl < best:
                    best, best_P = vl, P
            else:
                best_P = P
            log(msg + f" ({time.time()-t0:.0f}s)")
    out = {k: np.asarray(x) for k, x in best_P.items()}
    return out, {"val_bce": best if val is not None else None, "train_loss": hist[-1], "steps": steps,
                 "seconds": time.time() - t0}
