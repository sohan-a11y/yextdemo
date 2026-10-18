"""Real data: Overture Maps Places, NYC, two monthly releases (Aug 19 and Sep 23, 2026).

Task (mirrors "which facts in the knowledge graph will be stale next month?"):
  Given ONLY the August record of a place, predict whether by September
    - the place was removed (and not simply re-created under a new id), or
    - it was marked permanently_closed, or
    - its name, phone, website or address changed.
Features use the August table only (see `aug_features`; a unit test checks it never touches September).
"""
import os, json, time
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from .graph import knn_edges, group_edges
from .worlds import Dataset

AUG, SEP = "2026-08-19.0", "2026-09-23.0"
DATASETS = ["meta", "Foursquare", "Microsoft", "BrightQuery", "AllThePlaces", "PinMeTo", "DAC", "Overture-signals"]
TOP_CATS = 24


def _isn(x):
    return x is None or (isinstance(x, float) and x != x)


def _first(x, key=None):
    if _isn(x) or len(x) == 0:
        return None
    v = x[0]
    if key is None:
        return v
    return None if _isn(v) else v.get(key)


def _name(d):
    return None if _isn(d) else d.get("primary")


def _norm(s):
    return "" if s is None else "".join(ch for ch in str(s).lower() if ch.isalnum())


def load_tables(data_dir="lkm_data"):
    a = pq.read_table(os.path.join(data_dir, f"overture_{AUG}_nyc.parquet")).to_pandas()
    b = pq.read_table(os.path.join(data_dir, f"overture_{SEP}_nyc.parquet")).to_pandas()
    return a, b


def aug_features(a, ref_date=pd.Timestamp("2026-08-19", tz="UTC"), top_cats=None):
    """Feature matrix from the AUGUST table only. Returns (X, names, top_cats, age_col)."""
    n = len(a)
    ds_flag = np.zeros((n, len(DATASETS)), np.float32)
    newest, oldest, sig = [None] * n, [None] * n, [None] * n
    nsrc = np.zeros(n, np.float32)
    for i, srcs in enumerate(a["sources"].values):
        nsrc[i] = 0 if _isn(srcs) else len(srcs)
        if _isn(srcs):
            continue
        ts = []
        for s in srcs:
            d = s["dataset"]
            if d in DATASETS:
                ds_flag[i, DATASETS.index(d)] = 1
            t = s["update_time"]
            if d == "Overture-signals":
                sig[i] = t
            elif d != "Overture" and t is not None:
                ts.append(t)
        if ts:
            newest[i], oldest[i] = max(ts), min(ts)

    def age(col):
        t = pd.to_datetime(pd.Series(col), utc=True, errors="coerce")
        d = (ref_date - t).dt.total_seconds() / 86400.0
        return d.clip(lower=0).values

    a_new, a_old, a_sig = age(newest), age(oldest), age(sig)
    miss_new = np.isnan(a_new); miss_sig = np.isnan(a_sig)
    a_new = np.log1p(np.where(miss_new, np.nanmedian(a_new), a_new))
    a_old = np.log1p(np.where(np.isnan(a_old), np.nanmedian(a_old), a_old))
    a_sig = np.log1p(np.where(miss_sig, np.nanmedian(a_sig), a_sig))
    status = a["operating_status"].values
    web_n = a["websites"].map(lambda x: 0 if _isn(x) else len(x)).values.astype(np.float32)
    ph_n = a["phones"].map(lambda x: 0 if _isn(x) else len(x)).values.astype(np.float32)
    addr = a["addresses"].map(lambda x: _first(x, "freeform"))
    post = a["addresses"].map(lambda x: _first(x, "postcode"))
    nm = a["names"].map(_name)
    brand = a["brand"].map(lambda d: 0.0 if _isn(d) or (d.get("names") is None and d.get("wikidata") is None) else 1.0)
    cat = a["basic_category"].fillna("unknown")
    if top_cats is None:
        top_cats = list(cat.value_counts().head(TOP_CATS).index)
    cat_oh = np.stack([(cat.values == c).astype(np.float32) for c in top_cats] +
                      [(~cat.isin(top_cats)).values.astype(np.float32)], 1)
    cols = [
        ("age", a_new), ("age_oldest", a_old), ("age_signal", a_sig),
        ("miss_age", miss_new.astype(np.float32)), ("miss_signal", miss_sig.astype(np.float32)),
        ("conf", a["confidence"].fillna(0.5).values.astype(np.float32)), ("nsrc", nsrc),
        ("status_open", (status == "open").astype(np.float32)),
        ("has_web", (web_n > 0).astype(np.float32)), ("n_web", web_n),
        ("has_phone", (ph_n > 0).astype(np.float32)), ("n_phone", ph_n),
        ("has_addr", addr.notna().values.astype(np.float32)), ("has_post", post.notna().values.astype(np.float32)),
        ("name_len", nm.fillna("").str.len().values.astype(np.float32)),
        ("name_digit", nm.fillna("").str.contains(r"\d").values.astype(np.float32)),
        ("brand", brand.values.astype(np.float32)),
    ]
    names = [c[0] for c in cols] + [f"ds_{d}" for d in DATASETS] + [f"cat_{c}" for c in top_cats] + ["cat_other"]
    X = np.column_stack([c[1] for c in cols] + [ds_flag, cat_oh]).astype(np.float32)
    cont = [names.index(k) for k in ("age", "age_oldest", "age_signal", "conf", "nsrc", "n_web", "n_phone", "name_len")]
    mu, sd = X[:, cont].mean(0), X[:, cont].std(0) + 1e-6
    X[:, cont] = (X[:, cont] - mu) / sd
    return X, names, top_cats, names.index("age")


def make_labels(a, b):
    """Labels from the SEPTEMBER table, defined for places not already closed in August.
    Returns dict of boolean arrays aligned with `a` plus diagnostics."""
    bi = b.set_index("id")
    inb = a["id"].isin(bi.index).values
    sb = bi.reindex(a["id"].values)
    closed = sb["operating_status"].values == "permanently_closed"
    # removal is only an event if the place was not simply re-created under a new id nearby
    removed_raw = ~inb
    sxy = np.column_stack([b["bbox"].map(lambda d: d["xmin"]).values, b["bbox"].map(lambda d: d["ymin"]).values])
    axy = np.column_stack([a["bbox"].map(lambda d: d["xmin"]).values, a["bbox"].map(lambda d: d["ymin"]).values])
    snames = b["names"].map(lambda d: _norm(_name(d))).values
    anames = a["names"].map(lambda d: _norm(_name(d))).values
    tree = cKDTree(sxy)
    rematched = np.zeros(len(a), bool)
    cand = np.where(removed_raw)[0]
    hits = tree.query_ball_point(axy[cand], r=0.0004)    # ~35 m
    for i, h in zip(cand, hits):
        if any(snames[j] == anames[i] and anames[i] != "" for j in h):
            rematched[i] = True
    removed = removed_raw & ~rematched

    def col(df, f):
        return f(df)
    ca = pd.DataFrame({"name": a["names"].map(_name), "phone": a["phones"].map(_first), "web": a["websites"].map(_first),
                       "addr": a["addresses"].map(lambda x: _first(x, "freeform"))})
    cb = pd.DataFrame({"name": sb["names"].map(_name).values, "phone": sb["phones"].map(_first).values,
                       "web": sb["websites"].map(_first).values,
                       "addr": sb["addresses"].map(lambda x: _first(x, "freeform")).values})
    changed = np.zeros(len(a), bool)
    for c in ("name", "phone", "web", "addr"):
        changed |= (ca[c].fillna("§").values != cb[c].fillna("§").values) & inb
    gone = removed | closed
    return {"y": gone | changed, "y_gone": gone, "removed_raw": removed_raw, "rematched": rematched,
            "removed": removed, "closed": closed, "changed": changed}


def _chunks(block_ids, nodes_per_block, target):
    order = sorted(block_ids)
    out, cur, cnt = [], [], 0
    for b in order:
        cur.append(b); cnt += nodes_per_block[b]
        if cnt >= target:
            out.append(cur); cur, cnt = [], 0
    if cur:
        if out and cnt < target // 3:
            out[-1].extend(cur)
        else:
            out.append(cur)
    return out


def build_real(data_dir="lkm_data", seed=0, chunk_target=8000, cache=True, log=print):
    cpath = os.path.join(data_dir, "real_built.npz")
    if cache and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        full = _from_npz(z)
        log(f"[real] loaded cached dataset: {full.n} places")
    else:
        t0 = time.time()
        a, b = load_tables(data_dir)
        a = a[a["operating_status"] != "permanently_closed"].reset_index(drop=True)
        lab = make_labels(a, b)
        X, names, top_cats, age_col = aug_features(a)
        lon = a["bbox"].map(lambda d: d["xmin"]).values; lat = a["bbox"].map(lambda d: d["ymin"]).values
        xy = np.column_stack([lon * np.cos(np.deg2rad(40.7)), lat]).astype(np.float32)
        rng = np.random.default_rng(seed)
        bname = a["brand"].map(lambda d: "" if _isn(d) else (d.get("wikidata") or _norm(_name(d.get("names")) if d.get("names") is not None else None) or ""))
        gid = pd.factorize(bname.replace("", np.nan))[0]
        addr = a["addresses"].map(lambda x: _first(x, "freeform")); post = a["addresses"].map(lambda x: _first(x, "postcode"))
        akey = (addr.fillna("").str.lower().str.replace(r"[^a-z0-9]", "", regex=True) + "|" + post.fillna("").str[:5])
        akey = akey.where(addr.notna(), np.nan)
        aid = pd.factorize(akey)[0]
        edges = {"sp": knn_edges(xy, 6), "br": group_edges(gid, 2, rng), "ad": group_edges(aid, 2, rng)}
        full = Dataset("overture-nyc", X, lab["y"].astype(np.int8), edges, xy, names, [names.index(k) for k in ("age", "conf", "age_signal", "status_open")],
                       age_col, {"top_cats": top_cats}, lab["y_gone"].astype(np.int8))
        diag = {"n_pool": int(len(a)), "rate_union": float(lab["y"].mean()), "rate_gone": float(lab["y_gone"].mean()),
                "removed_raw": int(lab["removed_raw"].sum()), "rematched": int(lab["rematched"].sum()),
                "removed_true": int(lab["removed"].sum()), "closed": int(lab["closed"].sum()),
                "changed": int(lab["changed"].sum()), "t_build_s": round(time.time() - t0, 1)}
        full.meta["diag"] = diag
        if cache:
            _to_npz(full, cpath)
        log(f"[real] built dataset in {diag['t_build_s']}s: {diag}")
    # ---- spatial block split (blocks of 0.015 degrees), no leakage across splits
    bx = np.floor((full.xy[:, 0] / np.cos(np.deg2rad(40.7)) + 74.05) / 0.015).astype(int)
    by = np.floor((full.xy[:, 1] - 40.54) / 0.015).astype(int)
    blk = bx * 1000 + by
    ub, counts = np.unique(blk, return_counts=True)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(len(ub))
    n_te, n_va = int(0.20 * len(ub)), int(0.15 * len(ub))
    te_b, va_b, tr_b = set(ub[perm[:n_te]]), set(ub[perm[n_te:n_te + n_va]]), list(ub[perm[n_te + n_va:]])
    split = np.where(np.isin(blk, list(te_b)), 2, np.where(np.isin(blk, list(va_b)), 1, 0))
    nb = dict(zip(ub, counts))
    chunk_list = _chunks(tr_b, nb, chunk_target)
    train = [full.subset(np.where(np.isin(blk, ch))[0], name=f"train-chunk{i}") for i, ch in enumerate(chunk_list)]
    val = full.subset(np.where(split == 1)[0], name="val")
    test = full.subset(np.where(split == 2)[0], name="test")
    va_chunks = _chunks(sorted(va_b), nb, 10000)
    te_chunks = _chunks(sorted(te_b), nb, 7500)
    val_chunks = [full.subset(np.where(np.isin(blk, ch))[0], name=f"val-chunk{i}") for i, ch in enumerate(va_chunks)]
    test_pools = [full.subset(np.where(np.isin(blk, ch))[0], name=f"test-pool{i}") for i, ch in enumerate(te_chunks)]
    info = {"n_train": int(sum(d.n for d in train)), "n_chunks": len(train), "n_val": val.n, "n_test": test.n, "n_val_chunks": len(val_chunks),
            "n_test_pools": len(test_pools), "test_pool_sizes": [d.n for d in test_pools],
            "base_rate_train": float(np.concatenate([d.y for d in train]).mean()), "base_rate_test": float(test.y.mean()),
            "base_rate_gone_test": float(test.y_alt.mean())}
    log(f"[real] split: {info}")
    return {"full": full, "train": train, "val": val, "test": test, "val_chunks": val_chunks, "test_pools": test_pools, "info": info, "diag": full.meta.get("diag", {})}


def _to_npz(ds, path):
    arrs = {"X": ds.X, "y": ds.y, "y_alt": ds.y_alt, "xy": ds.xy, "feat_names": np.array(ds.feat_names), "agg_cols": np.array(ds.agg_cols),
            "age_col": np.array(ds.age_col), "meta": np.array(json.dumps({k: v for k, v in ds.meta.items()}))}
    for r in ("sp", "br", "ad"):
        arrs[f"e_{r}_s"], arrs[f"e_{r}_d"] = ds.edges[r]
    np.savez_compressed(path, **arrs)


def _from_npz(z):
    edges = {r: (z[f"e_{r}_s"], z[f"e_{r}_d"]) for r in ("sp", "br", "ad")}
    return Dataset("overture-nyc", z["X"], z["y"], edges, z["xy"], list(z["feat_names"]), list(z["agg_cols"]), int(z["age_col"]),
                   json.loads(str(z["meta"])), z["y_alt"])
