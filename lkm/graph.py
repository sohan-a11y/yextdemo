"""Graph utilities shared by synthetic worlds, real data, baselines and the organism.

A mesh has three relation types between entities ("nodes"):
  sp  spatial neighbours (k nearest)
  br  same brand / chain
  ad  same address / building
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

REL = ("sp", "br", "ad")
KMAX = {"sp": 6, "br": 3, "ad": 3}      # neighbour slots per relation for walkers
NSLOT = sum(KMAX.values())


def sym_csr(src, dst, n):
    """Symmetric, unweighted, loop-free CSR adjacency from an edge list."""
    src = np.asarray(src, dtype=np.int64); dst = np.asarray(dst, dtype=np.int64)
    if len(src) == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    A = sp.coo_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(n, n)).tocsr()
    A = A + A.T
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    return A.tocsr()


def knn_edges(xy, k):
    """Directed kNN edge list (src -> its k nearest neighbours)."""
    n = len(xy)
    k = min(k, max(1, n - 1))
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].ravel()
    return src, dst


def group_edges(groups, links, rng):
    """Connect members that share a group id (>=0) in a random ring with `links` links each side."""
    groups = np.asarray(groups)
    members = np.where(groups >= 0)[0]
    if len(members) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    order = members[np.argsort(groups[members], kind="stable")]
    g_sorted = groups[order]
    cuts = np.flatnonzero(np.diff(g_sorted)) + 1
    src, dst = [], []
    for blk in np.split(order, cuts):
        m = len(blk)
        if m < 2:
            continue
        perm = blk[rng.permutation(m)]
        for off in range(1, min(links, m - 1) + 1):
            src.append(perm); dst.append(np.roll(perm, -off))
    if not src:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(src), np.concatenate(dst)


def row_normalize(A):
    d = np.asarray(A.sum(1)).ravel()
    inv = np.where(d > 0, 1.0 / np.maximum(d, 1e-9), 0.0).astype(np.float32)
    return sp.diags(inv) @ A


def _slots(A, k, salt=0):
    """k neighbours per row of CSR A as an (n,k) int array padded with -1. Neighbours are taken in a
    deterministic pseudo-random order so truncation of high-degree nodes is not biased by node id."""
    n = A.shape[0]
    deg = np.diff(A.indptr)
    out = np.full((n, k), -1, np.int64)
    if A.nnz == 0:
        return out
    rows = np.repeat(np.arange(n), deg)
    key = (rows.astype(np.uint64) * np.uint64(2654435761) + A.indices.astype(np.uint64) * np.uint64(40503)
           + np.uint64(salt)) & np.uint64(0xFFFFFFFF)
    order = np.lexsort((key, rows))
    ind = A.indices[order]
    j = np.arange(k)[None, :]
    pos = A.indptr[:-1, None] + j
    valid = j < deg[:, None]
    out = np.where(valid, ind[np.minimum(pos, A.nnz - 1)], -1)
    return out.astype(np.int64)


class Graph:
    """A mesh: relation adjacencies + derived views. Build with Graph.from_edges."""

    def __init__(self, n, A):
        self.n = n
        self.A = A                                   # rel -> binary CSR (symmetric)
        self.An = {r: row_normalize(A[r]).astype(np.float32).tocsr() for r in REL}
        self.deg = np.stack([np.diff(A[r].indptr) for r in REL], 1).astype(np.float32)
        slots, rels = [], []
        for ri, r in enumerate(REL):
            s = _slots(A[r], KMAX[r])
            slots.append(s); rels.append(np.where(s >= 0, ri, -1))
        self.nbr = np.concatenate(slots, 1)          # (n, NSLOT) neighbour ids or -1
        self.nbr_rel = np.concatenate(rels, 1)       # (n, NSLOT) relation id or -1
        U = A["sp"] + A["br"] + A["ad"]
        U.data[:] = 1.0
        self.Au = U.tocsr()
        self.Aun = row_normalize(self.Au).astype(np.float32).tocsr()

    @staticmethod
    def from_edges(n, edges):
        """edges: rel -> (src, dst) arrays over local ids 0..n-1."""
        return Graph(n, {r: sym_csr(*edges.get(r, (np.zeros(0), np.zeros(0))), n) for r in REL})

    @staticmethod
    def induced(n_global, edges, idx):
        """Induced subgraph on node subset idx (global ids). Local id = position in idx."""
        idx = np.asarray(idx)
        loc = np.full(n_global, -1, np.int64)
        loc[idx] = np.arange(len(idx))
        sub = {}
        for r in REL:
            s, d = edges[r]
            ls, ld = loc[s], loc[d]
            ok = (ls >= 0) & (ld >= 0)
            sub[r] = (ls[ok], ld[ok])
        return Graph.from_edges(len(idx), sub)

    def edge_count(self):
        return {r: int(self.A[r].nnz // 2) for r in REL}


def graph_features(G, X, cols):
    """Neighbour aggregates for GBDT baselines: per relation 1-hop mean of selected columns,
    union 2-hop mean, and degrees. cols: indices of columns to aggregate."""
    Xc = X[:, cols].astype(np.float32)
    out = [G.deg / (1.0 + G.deg)]
    for r in REL:
        out.append(G.An[r] @ Xc)
    m1 = G.Aun @ Xc
    out.append(m1)
    out.append(G.Aun @ m1)
    return np.concatenate(out, 1).astype(np.float32)


def reveal_features(G, rev_pos, rev_neg):
    """Features describing revealed (verified) neighbours: hand-crafted 'scent' for baselines."""
    P = rev_pos.astype(np.float32)[:, None]; N_ = rev_neg.astype(np.float32)[:, None]
    out = []
    for r in REL:
        out.append(G.An[r] @ P)
        out.append(G.An[r] @ N_)
        out.append(G.A[r] @ P)
    m1 = G.Aun @ P
    out.append(m1); out.append(G.Aun @ m1)
    out.append(G.Aun @ N_)
    return np.concatenate(out, 1).astype(np.float32)
