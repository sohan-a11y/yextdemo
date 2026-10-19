"""Graph neural cellular automaton (NCA) on the mesh: NumPy inference (training lives in nca_train.py).

Every node holds a small state vector s (C channels). All nodes share ONE tiny update rule (the "genome" of
the field). For T steps each cell reads its own state and the mean state of its neighbours along each relation
(space / brand / address) and updates itself (asynchronous: each cell updates with probability `keep`).
The final state is read out as a residual on top of a baseline logit:

    logit_i = base_i + s_i . w_out + b_out        (w_out = 0 at initialisation => starts as the baseline)

Inputs per cell: node features, the baseline logit, 'revealed positive/negative' flags (what verification has
uncovered so far) and the fraction of features that are missing.
"""
import numpy as np
import scipy.sparse as sp
from .graph import REL, KMAX

C_DEFAULT, H_DEFAULT, T_DEFAULT = 16, 32, 8
ALPHA, KEEP = 0.5, 0.75


def tables_idx(G):
    """Per-relation padded neighbour tables (idx, mask): used by the JAX trainer."""
    out, o = [], 0
    for r in REL:
        k = KMAX[r]
        idx = G.nbr[:, o:o + k]
        out.append((np.maximum(idx, 0), (idx >= 0).astype(np.float32)))
        o += k
    return out


def tables(G):
    """Same aggregation as sparse row-normalised matrices (fast NumPy inference). Mean over the valid slots."""
    out = []
    for idx, mask in tables_idx(G):
        n, k = idx.shape
        cnt = np.maximum(mask.sum(1, keepdims=True), 1.0)
        rows = np.repeat(np.arange(n), k)
        w = (mask / cnt).ravel()
        keep = mask.ravel() > 0
        A = sp.coo_matrix((w[keep], (rows[keep], idx.ravel()[keep])), shape=(n, n)).tocsr().astype(np.float32)
        out.append(A)
    return out


def build_inp(X, base, rev_pos=None, rev_neg=None, miss=None):
    n = len(X)
    z = np.zeros(n, np.float32)
    rp = z if rev_pos is None else rev_pos.astype(np.float32)
    rn = z if rev_neg is None else rev_neg.astype(np.float32)
    ms = z if miss is None else miss.astype(np.float32)
    b = np.clip(base, -6, 6).astype(np.float32) / 3.0
    return np.concatenate([np.nan_to_num(X).astype(np.float32), b[:, None], rp[:, None], rn[:, None], ms[:, None]], 1)


def init_params(din, C=C_DEFAULT, H=H_DEFAULT, seed=0, V=0, comm="cont"):
    """V > 0 gives the cells a vocabulary of V words. comm: 'cont' neighbours hear the continuous state,
    'lex' neighbours hear only words (a discrete language), 'both' state and words."""
    r = np.random.default_rng(seed)
    comm = comm if V else "cont"
    nin = {"cont": 4 * C, "lex": C + 3 * V, "both": 4 * C + 3 * V}[comm]
    P = {"We": (r.normal(0, 1, (din, C)) / np.sqrt(din)).astype(np.float32), "be": np.zeros(C, np.float32),
         "W1": (r.normal(0, 1, (nin, H)) / np.sqrt(nin)).astype(np.float32), "b1": np.zeros(H, np.float32),
         "W2": (r.normal(0, 0.1, (H, C)) / np.sqrt(H)).astype(np.float32), "b2": np.zeros(C, np.float32),
         "wo": np.zeros(C, np.float32), "bo": np.zeros((), np.float32)}
    if V and comm != "cont":
        P["Wt"] = (r.normal(0, 1, (C, V)) / np.sqrt(C)).astype(np.float32)
        P["bt"] = np.zeros(V, np.float32)
    return P


def comm_mode(P):
    """Infer the communication mode of a parameter set from its shapes."""
    if "Wt" not in P:
        return "cont"
    C, V = P["Wt"].shape
    return "lex" if P["W1"].shape[0] == C + 3 * V else "both"


def _agg(s, A):
    return A @ s


def words(P, s):
    """The word (0..V-1) each cell is 'saying' given its state s."""
    return (s @ P["Wt"] + P["bt"]).argmax(1)


def forward(P, inp, tabs, T=T_DEFAULT, readout=None):
    """Expected-update (deterministic) forward pass. Returns (delta_logit, final_state)."""
    s = np.tanh(inp @ P["We"] + P["be"])
    mode = comm_mode(P)
    for _ in range(T):
        parts = [s]
        if mode in ("cont", "both"):
            parts += [_agg(s, A) for A in tabs]
        if mode in ("lex", "both"):
            V = P["Wt"].shape[1]
            hot = np.eye(V, dtype=np.float32)[words(P, s)]
            parts += [_agg(hot, A) for A in tabs]
        z = np.concatenate(parts, 1)
        h = np.tanh(z @ P["W1"] + P["b1"])
        s = s + KEEP * ALPHA * (h @ P["W2"] + P["b2"])
    wo, bo = (P["wo"], P["bo"]) if readout is None else readout
    return s @ wo + bo, s


def save(P, path):
    np.savez(path, **P)


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}
