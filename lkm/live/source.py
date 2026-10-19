"""Where the organism senses the world. A source returns the CURRENT state of the world each time it is asked.

A source must provide:
    version()          cheap string that changes when the world changed (the organism re-senses only then)
    snapshot()         -> (Dataset, ids)   ids are stable string keys of the entities, aligned with the Dataset rows
    url_of(entity_id)  optional, used by the web-check verifier
The Dataset's label field `y` is NEVER read by the organism; sources hand it zeros.
"""
import hashlib, json, os
from dataclasses import replace
import numpy as np


class DatasetSource:
    """In-memory world (used by tests and by wrappers that build their own Dataset)."""

    def __init__(self, ds, ids, version="v1", urls=None):
        self.ds = replace(ds, y=np.zeros(ds.n, np.int8), y_alt=None, _g=None)
        self.ids = np.asarray([str(i) for i in ids])
        self._version, self.urls = version, urls or {}

    def set_version(self, v):
        self._version = v

    def version(self):
        return self._version

    def snapshot(self):
        return self.ds, self.ids

    def url_of(self, eid):
        return self.urls.get(eid)


class CsvSource:
    """Generic world from a CSV file that your system refreshes (export it nightly, or write it from your database).

    One row per entity. Columns: an id column, numeric feature columns, optional lat/lon (spatial neighbours), optional
    group columns that link entities (same chain, same address, same owner ...) and an age column ("days since last
    confirmed", any monotone scale). The spec (feature list, scaler, age column) is fixed at bootstrap so that later
    snapshots are encoded exactly like the history the models were trained on.
    """

    def __init__(self, path, spec):
        self.path, self.spec = path, spec

    @staticmethod
    def fit_spec(df, id_col, age_col, lat_col=None, lon_col=None, group_cols=None, text_col=None, url_col=None,
                 label_col=None, feature_cols=None, agg_cols=None):
        special = {id_col, lat_col, lon_col, text_col, url_col, label_col, *(group_cols or {}).values()}
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in special and np.issubdtype(df[c].dtype, np.number)]
        scaler = {}
        for c in feature_cols:
            v = df[c].astype(float)
            binary = set(np.unique(v.dropna())) <= {0.0, 1.0}
            scaler[c] = {"binary": bool(binary), "mean": 0.0 if binary else float(v.mean()), "std": 1.0 if binary else float(v.std() + 1e-9)}
        if age_col not in feature_cols:
            raise ValueError(f"age column {age_col!r} must be one of the numeric feature columns")
        agg = agg_cols or [c for c in (age_col,) if c in feature_cols] + [c for c in feature_cols if c != age_col][:3]
        return {"id_col": id_col, "age_col": age_col, "lat_col": lat_col, "lon_col": lon_col, "group_cols": group_cols or {},
                "text_col": text_col, "url_col": url_col, "label_col": label_col, "feature_cols": feature_cols, "scaler": scaler, "agg_cols": agg}

    def version(self):
        st = os.stat(self.path)
        return f"{int(st.st_mtime)}-{st.st_size}"

    def _read(self):
        import pandas as pd
        return pd.read_csv(self.path)

    def build(self, df, labels=False):
        from ..graph import knn_edges, group_edges
        from ..worlds import Dataset
        s = self.spec
        ids = df[s["id_col"]].astype(str).values
        X = np.column_stack([((df[c].astype(float).values if c in df else np.full(len(df), np.nan)) - s["scaler"][c]["mean"]) / s["scaler"][c]["std"]
                             for c in s["feature_cols"]]).astype(np.float32)
        rng = np.random.default_rng(0)
        empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        edges = {"sp": empty, "br": empty, "ad": empty}
        xy = np.zeros((len(df), 2), np.float32)
        if s["lat_col"] and s["lon_col"]:
            xy = np.column_stack([df[s["lon_col"]].values * np.cos(np.deg2rad(df[s["lat_col"]].values.mean())), df[s["lat_col"]].values]).astype(np.float32)
            edges["sp"] = knn_edges(xy, 6)
        for rel, col in s["group_cols"].items():
            import pandas as pd
            gid = pd.factorize(df[col].where(df[col].notna()))[0]
            edges[rel] = group_edges(gid, 2, rng)
        y = df[s["label_col"]].astype(np.int8).values if (labels and s.get("label_col")) else np.zeros(len(df), np.int8)
        feats = s["feature_cols"]
        ds = Dataset("csv", X, y, edges, xy, feats, [feats.index(c) for c in s["agg_cols"]], feats.index(s["age_col"]), {})
        if s["text_col"] and s["text_col"] in df:
            ds.text = df[s["text_col"]].astype(str).values
        return ds, ids

    def snapshot(self):
        return self.build(self._read())

    def url_of(self, eid):
        col = self.spec.get("url_col")
        if not col:
            return None
        df = self._read()
        r = df[df[self.spec["id_col"]].astype(str) == eid]
        return None if r.empty or r.iloc[0][col] != r.iloc[0][col] else str(r.iloc[0][col])

    def save_spec(self, home):
        json.dump(self.spec, open(os.path.join(home, "source_spec.json"), "w"), indent=1)


def load_source(home, cfg):
    """Build the source named in the habitat config."""
    kind = cfg["source"]["type"]
    if kind == "csv":
        spec = json.load(open(os.path.join(home, "source_spec.json")))
        return CsvSource(cfg["source"]["path"], spec)
    if kind == "overture":
        from .overture import OvertureSource
        return OvertureSource(pool=cfg["source"].get("pool", 0), data_dir=cfg["source"].get("data_dir", "lkm_data"))
    if kind == "custom":                                   # "package.module:factory" returning an object with the source interface
        import importlib
        mod, fn = cfg["source"]["factory"].split(":")
        return getattr(importlib.import_module(mod), fn)(**cfg["source"].get("args", {}))
    raise ValueError(f"unknown source type {kind}")
