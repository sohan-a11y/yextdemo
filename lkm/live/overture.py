"""A real public world for running the organism: Overture Maps places in New York City.

The Dataset comes from the same reader pipeline the study used (strings, name codes, graph). The organism receives the
entities' features and graph, never their labels. `truth()` exposes what really happened between the two public releases
so that answers can be replayed or the organism's picks graded afterwards.
"""
from dataclasses import replace
import numpy as np
from .source import DatasetSource


class OvertureSource(DatasetSource):
    def __init__(self, pool=0, data_dir="lkm_data"):
        from ..realdata import build_real, load_tables
        R = build_real(data_dir=data_dir, language=True, lang_edges=False, log=lambda *a: None)
        a, _ = load_tables(data_dir)
        ids_full = a.loc[(a["operating_status"] != "permanently_closed").values, "id"].astype(str).values
        ds = R["test_pools"][pool]
        ids = ids_full[ds.meta["idx"]]
        self._truth = {i: int(y) for i, y in zip(ids.tolist(), ds.y.tolist())}
        urls = {}
        web = a.set_index("id")["websites"]
        for i in ids.tolist():
            w = web.get(i)
            if w is not None and len(w):
                urls[i] = str(w[0])
        super().__init__(ds, ids, version="overture-2026-08-19.0", urls=urls)

    def truth(self):
        return dict(self._truth)
