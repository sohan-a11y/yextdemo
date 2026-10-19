"""Language for the organism: a pretrained sentence encoder (all-MiniLM-L6-v2, ONNX, runs on CPU) that lets
cells and workers *read* place names, categories and addresses, plus language-derived relations between places.

What the organism reads from language (August table only, no labels, no September data):
  * a 16-d semantic code of "name | category" (MiniLM embedding, PCA fitted on the training places)
  * how name-like each place is to its closest spatial neighbours (near-duplicates, chain siblings)
  * lexical flags (legal suffixes, digits, shouting, generic words)
These enter the cell field and the baselines as ordinary input channels, so every method sees the same text.
"""
import os, re, time
import numpy as np

MODEL_DIR = os.path.join("lkm_data", "minilm", "onnx")
MODEL_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"


def ensure_model(model_dir=MODEL_DIR):
    if os.path.exists(os.path.join(model_dir, "model.onnx")):
        return model_dir
    import tarfile, urllib.request
    base = os.path.dirname(model_dir)
    os.makedirs(base, exist_ok=True)
    tgz = os.path.join(base, "onnx.tar.gz")
    urllib.request.urlretrieve(MODEL_URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(base)
    return model_dir


class TextEncoder:
    """Mean-pooled, L2-normalised MiniLM sentence embeddings (384-d)."""
    dim = 384

    def __init__(self, model_dir=MODEL_DIR, max_len=32, threads=4):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        model_dir = ensure_model(model_dir)
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(os.path.join(model_dir, "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(length=None)
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts, batch=256, log=None):
        out = np.zeros((len(texts), self.dim), np.float32)
        # sort by length so batches pad little
        order = np.argsort([len(t) for t in texts])
        t0 = time.time()
        for s in range(0, len(texts), batch):
            ix = order[s:s + batch]
            enc = self.tok.encode_batch([texts[i] for i in ix])
            ids = np.array([e.ids for e in enc], np.int64); mask = np.array([e.attention_mask for e in enc], np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(1) / np.maximum(m.sum(1), 1.0)
            out[ix] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            if log and (s // batch) % 200 == 0 and s:
                log(f"[lang] encoded {s}/{len(texts)} ({time.time()-t0:.0f}s)")
        return out


# ------------------------------------------------------------------------- lexical flags
_LEGAL = re.compile(r"\b(llc|inc|corp|co|ltd|lp|pllc|pc|dds|md|md pc)\b\.?", re.I)
_GENERIC = re.compile(r"\b(the|of|and|new york|nyc|brooklyn|queens|bronx|manhattan|staten island)\b", re.I)


def lexical_flags(names):
    """Cheap, interpretable text flags for each name."""
    rows = []
    for s in names:
        s = "" if s is None else str(s)
        letters = [c for c in s if c.isalpha()]
        up = sum(c.isupper() for c in letters) / max(len(letters), 1)
        rows.append([
            1.0 if _LEGAL.search(s) else 0.0,
            1.0 if re.search(r"\d", s) else 0.0,
            1.0 if (len(letters) > 3 and up > 0.9) else 0.0,
            1.0 if _GENERIC.search(s) else 0.0,
            min(len(s.split()), 8) / 8.0,
            1.0 if re.search(r"[&@#/\\|]", s) else 0.0,
            1.0 if s.strip() == "" else 0.0,
        ])
    return np.asarray(rows, np.float32)


LEX_NAMES = ["lx_legal", "lx_digit", "lx_shout", "lx_generic", "lx_words", "lx_symbol", "lx_empty"]


# ------------------------------------------------------------------------- language-derived relations
def name_neighbour_features(emb, xy, k=8, radius_deg=0.0008, chunk=20000):
    """For each place: cosine similarity to its most similar spatial neighbours (within ~90 m) and counts of
    near-duplicates. This is where language shows the organism entities that are probably the same thing."""
    from scipy.spatial import cKDTree
    n = len(emb)
    tree = cKDTree(xy)
    d, idx = tree.query(xy, k=min(k + 1, n))
    idx = idx[:, 1:]; d = d[:, 1:]
    sims = np.empty(idx.shape, np.float32)
    for s in range(0, n, chunk):
        e = emb[s:s + chunk].astype(np.float32)
        sims[s:s + chunk] = np.einsum("nd,nkd->nk", e, emb[idx[s:s + chunk]].astype(np.float32))
    near = (d <= radius_deg)
    sims = np.where(near, sims, -1.0)
    top1 = sims.max(1)
    f = np.column_stack([
        np.clip(top1, 0, 1),                                  # best name match among close neighbours
        (sims > 0.90).sum(1) / 3.0,                           # near-duplicate count
        (sims > 0.75).sum(1) / 4.0,                           # same-family count
        near.sum(1) / float(idx.shape[1]),                    # local density
    ]).astype(np.float32)
    return f, ["nn_best", "nn_dup", "nn_family", "nn_density"]


def semantic_edges(emb, train_mask, k=3, thr=0.92, seed=0, n_probe=5):
    """Language edges: each place is linked to its most similar *names* anywhere in the city (cosine >= thr).
    Recovers chains and same-name families that the brand field misses. Candidates come from a 32-d PCA
    index, edges are kept on the exact 384-d cosine."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    tr = np.where(train_mask)[0]
    sub = tr[rng.choice(len(tr), min(len(tr), 60000), replace=False)]
    pca = Pca().fit(emb[sub].astype(np.float32), k=32)
    Z = pca.transform(emb.astype(np.float32))
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    Z = Z + rng.normal(0, 1e-4, Z.shape).astype(np.float32)        # random tie-breaking among identical names
    try:                                   # exact inner-product search (BLAS); a KD-tree is hopeless in 32-d at 400k points
        import faiss
        index = faiss.IndexFlatIP(Z.shape[1]); index.add(np.ascontiguousarray(Z, dtype=np.float32))
        _, idx = index.search(np.ascontiguousarray(Z, dtype=np.float32), min(n_probe + 1, len(Z)))
    except ImportError:
        _, idx = cKDTree(Z).query(Z, k=min(n_probe + 1, len(Z)), workers=4)
    src, dst = [], []
    for j in range(1, idx.shape[1]):
        i = np.arange(len(emb)); t = idx[:, j]
        ok = (t >= 0) & (t != i)
        i, t = i[ok], t[ok]
        c = np.einsum("nd,nd->n", emb[i].astype(np.float32), emb[t].astype(np.float32))
        keep = c >= thr
        src.append(i[keep]); dst.append(t[keep])
    src = np.concatenate(src); dst = np.concatenate(dst)
    # cap per source at k
    order = np.lexsort((np.arange(len(src)), src))
    src, dst = src[order], dst[order]
    first = np.r_[0, np.flatnonzero(np.diff(src)) + 1]
    rank = np.arange(len(src)) - np.repeat(first, np.diff(np.r_[first, len(src)]))
    keep = rank < k
    return src[keep], dst[keep]


class Pca:
    """Plain PCA fitted on a subset (the training places), applied everywhere."""

    def fit(self, X, k=16):
        self.mu = X.mean(0)
        u, s, vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.W = vt[:k].T
        self.scale = (s[:k] / np.sqrt(len(X))) + 1e-6
        return self

    def transform(self, X):
        return (((X - self.mu) @ self.W) / self.scale).astype(np.float32)


TOLL = ("800", "833", "844", "855", "866", "877", "888")
NYC_AREA = ("212", "718", "917", "646", "347", "929", "332")
_SOCIAL = re.compile(r"(facebook|instagram|twitter|x\.com|linkedin|tiktok|youtube|yelp|linktr\.ee|linktree|tripadvisor|foursquare|google\.|maps\.)", re.I)
_BUILDER = re.compile(r"(wixsite|weebly|squarespace|godaddysites|business\.site|square\.site|wordpress\.com|blogspot|sites\.google|webflow\.io|myshopify|carrd\.co)", re.I)
_UNIT = re.compile(r"(\b(ste|suite|apt|unit|fl|floor|rm|room|bsmt)\b|#\s?\d)", re.I)
STR_NAMES = ["w_social", "w_builder", "w_own", "w_https", "w_depth", "w_namematch", "p_toll", "p_nyc", "p_other", "p_two",
             "a_unit", "a_digit", "a_zip4", "loc_manh", "loc_bk", "loc_q", "loc_bx", "loc_si", "b_namematch",
             "s_baseconf", "s_sigconf", "s_confage"]


def _nz(x):
    return x is None or (isinstance(x, float) and x != x)


def _tok(s):
    return set(t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if len(t) >= 3)


def string_features(a, ref=None):
    """Website / phone / address / provenance strings of the August table as numeric features."""
    import pandas as pd
    ref = ref or pd.Timestamp("2026-08-19", tz="UTC")
    n = len(a)
    F = np.zeros((n, len(STR_NAMES)), np.float32)
    names = [None if _nz(d) else d.get("primary") for d in a["names"]]
    brands = [None if _nz(d) or _nz(d.get("names")) else d["names"].get("primary") for d in a["brand"]]
    webs, phones, addrs, srcs = a["websites"].values, a["phones"].values, a["addresses"].values, a["sources"].values
    for i in range(n):
        w = None if _nz(webs[i]) or len(webs[i]) == 0 else str(webs[i][0])
        if w:
            dom = re.sub(r"^https?://", "", w.lower()).split("/")[0].replace("www.", "")
            soc, bld = bool(_SOCIAL.search(w)), bool(_BUILDER.search(w))
            F[i, 0], F[i, 1], F[i, 2] = soc, bld, not (soc or bld)
            F[i, 3] = w.lower().startswith("https")
            F[i, 4] = min(max(w.count("/") - 2, 0), 4) / 4.0
            nt = _tok(names[i])
            F[i, 5] = (sum(t in dom for t in nt) / len(nt)) if nt else 0.0
        ph = None if _nz(phones[i]) or len(phones[i]) == 0 else str(phones[i][0])
        if ph:
            d = re.sub(r"\D", "", ph)
            ac = d[1:4] if d.startswith("1") and len(d) >= 4 else d[:3]
            F[i, 6], F[i, 7] = ac in TOLL, ac in NYC_AREA
            F[i, 8] = not (ac in TOLL or ac in NYC_AREA)
            F[i, 9] = len(phones[i]) > 1
        ad = None if _nz(addrs[i]) or len(addrs[i]) == 0 else addrs[i][0]
        if ad is not None:
            fr = ad.get("freeform") or ""
            F[i, 10] = bool(_UNIT.search(fr)); F[i, 11] = fr[:1].isdigit()
            F[i, 12] = "-" in (ad.get("postcode") or "")
            loc = (ad.get("locality") or "").lower()
            for j, key in enumerate(("new york", "brooklyn", "queens", "bronx", "staten")):
                if key in loc:
                    F[i, 13 + j] = 1.0
        bt, nt = _tok(brands[i]), _tok(names[i])
        F[i, 18] = (len(bt & nt) / len(bt)) if bt else 0.0
        base_c, sig_c, conf_age = 0.5, 0.5, 0.0
        if not _nz(srcs[i]):
            for sc in srcs[i]:
                if sc["property"] == "" and sc["confidence"] is not None:
                    base_c = float(sc["confidence"])
                elif sc["dataset"] == "Overture-signals" and sc["confidence"] is not None:
                    sig_c = float(sc["confidence"])
                elif sc["dataset"] == "Overture" and sc["update_time"]:
                    t = pd.Timestamp(sc["update_time"])
                    conf_age = max((ref - (t.tz_localize("UTC") if t.tzinfo is None else t)).total_seconds() / 86400.0, 0.0)
        F[i, 19], F[i, 20], F[i, 21] = base_c, sig_c, min(np.log1p(conf_age) / 5.0, 2.0)
    return F, list(STR_NAMES)


def build_layer(a, E, xy, train_mask, edges_k=3, edges_thr=0.92, seed=0, log=print):
    """Everything the organism reads from language, for the rows of the August table `a` (aligned with E and xy).
    Returns dict(F, names, src, dst). No label and no September data is used; PCA is fitted on train places only."""
    t0 = time.time()
    E = E.astype(np.float32)
    S, s_names = string_features(a)
    log(f"[lang] string features {S.shape} ({time.time()-t0:.0f}s)")
    names = [None if _nz(d) else d.get("primary") for d in a["names"]]
    Lx = lexical_flags(names)
    nn, nn_names = name_neighbour_features(E, xy)
    log(f"[lang] name-neighbour features ({time.time()-t0:.0f}s)")
    rng = np.random.default_rng(seed)
    tr = np.where(train_mask)[0]
    sub = tr[rng.choice(len(tr), min(len(tr), 60000), replace=False)]
    pca = Pca().fit(E[sub], k=16)
    T = np.clip(pca.transform(E), -4, 4) / 2.0
    src, dst = semantic_edges(E, train_mask, k=edges_k, thr=edges_thr, seed=seed)
    log(f"[lang] {len(src)} language edges ({time.time()-t0:.0f}s)")
    F = np.concatenate([S, Lx, nn, T], 1).astype(np.float32)
    text = np.array(place_text(names, a["basic_category"].fillna("unknown").tolist()))
    return {"F": F, "names": s_names + LEX_NAMES + nn_names + [f"tx{i}" for i in range(T.shape[1])], "src": src, "dst": dst, "text": text}


def place_text(names, cats):
    return [f"{(n or '').strip()} | {(c or '').replace('_', ' ')}" for n, c in zip(names, cats)]


def cache_embeddings(a, path, log=print):
    """Embed 'name | category' for every row of the August table (row order) and cache as float16."""
    if os.path.exists(path):
        return np.load(path)
    names = [None if d is None or (isinstance(d, float)) else d.get("primary") for d in a["names"]]
    tx = place_text(names, a["basic_category"].fillna("unknown").tolist())
    E = TextEncoder().encode(tx, log=log).astype(np.float16)
    np.save(path, E)
    return E


if __name__ == "__main__":
    import pyarrow.parquet as pq
    a = pq.read_table(os.path.join("lkm_data", "overture_2026-08-19.0_nyc.parquet"), columns=["id", "names", "basic_category"]).to_pandas()
    t0 = time.time()
    E = cache_embeddings(a, os.path.join("lkm_data", "emb_aug.npy"))
    print("done", E.shape, f"{time.time()-t0:.0f}s", flush=True)
