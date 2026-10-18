"""Speech for the organism: turns what cells and workers know into plain sentences.

Two layers, both grounded in the organism's actual state (nothing is invented):
  * a DICTIONARY of the cells' own vocabulary: for every word, who says it, what they have in common, how often
    those places change;
  * EXPLANATIONS and WORKER LOGS: for a place or a verification, which signals stood out, which word the cell
    field was saying there, what neighbouring verifications showed.
The sentences come from templates filled with measured numbers; this is not a language model.
"""
import numpy as np
from . import nca as N

# feature -> (phrase when unusually high, phrase when unusually low)
PHRASES = {
    "age": ("its source record has not been refreshed for a long time", "its source record is fresh"),
    "age_oldest": ("its oldest source is very old", "all its sources are recent"),
    "age_signal": ("the open/closed signal is old", "the open/closed signal is recent"),
    "conf": ("source confidence is high", "source confidence is low"),
    "nsrc": ("many sources back it", "it rests on a single source"),
    "has_web": ("it lists a website", "it has no website"),
    "has_phone": ("it lists a phone", "it has no phone"),
    "w_social": ("its 'website' is only a social-media page", ""),
    "w_builder": ("its website is on a site-builder platform", ""),
    "w_own": ("it has its own web domain", "it has no own web domain"),
    "w_https": ("its website uses https", "its website is plain http"),
    "w_namematch": ("its web domain matches its name", "its web domain does not match its name"),
    "p_toll": ("its phone is toll-free", ""), "p_nyc": ("its phone has a New York area code", ""),
    "p_other": ("its phone has an out-of-area code", ""),
    "a_unit": ("its address includes a suite/unit", "its address has no suite/unit"),
    "a_zip4": ("its postcode has the +4 extension", ""),
    "lx_legal": ("its name carries a legal suffix (LLC, Inc ...)", ""),
    "lx_shout": ("its name is in capitals", ""), "lx_digit": ("its name contains digits", ""),
    "lx_generic": ("its name is generic", ""),
    "nn_best": ("a near-identical name sits right next door", "no similar name nearby"),
    "nn_dup": ("it has a near-duplicate within about 90 m", "no near-duplicate nearby"),
    "nn_family": ("it belongs to a family of similar names nearby", ""),
    "nn_density": ("it sits on a crowded block", "it is isolated"),
    "brand": ("it belongs to a chain", "it is independent"),
    "name_len": ("its name is long", "its name is very short"),
    "status_open": ("it is marked open", "it is not marked open"),
    "b_namematch": ("its name matches its brand", ""),
    "s_confage": ("its confidence score is old", "its confidence score is recent"),
}


def phrase(name, high=True):
    if name in PHRASES:
        return PHRASES[name][0 if high else 1]
    if name.startswith("cat_"):
        c = name[4:].replace("_", " ")
        return f"its category is {c}" if high else f"its category is not {c}"
    if name.startswith("ds_"):
        return f"it is sourced from {name[3:]}" if high else f"it is not sourced from {name[3:]}"
    return ""


def _explainable(name):
    return bool(phrase(name, True) or phrase(name, False)) and not name.startswith("tx")


class Narrator:
    def __init__(self, ds, base, P=None, seed=0):
        self.ds, self.base, self.P = ds, np.asarray(base, np.float64), P
        X = np.nan_to_num(ds.X).astype(np.float64)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.Z = (X - self.mu) / self.sd
        self.names = list(ds.feat_names)
        self.ok = np.array([_explainable(n) for n in self.names])
        self.seed = seed
        self.words = None
        self.V = 0
        if P is not None and "Wt" in P:
            tabs = N.tables(ds.graph())
            _, s = N.forward(P, N.build_inp(ds.X, self.base), tabs)
            self.words = N.words(P, s)
            self.V = P["Wt"].shape[1]
        self.overall = float(ds.y.mean())
        self.text = ds.text
        self._dic = None

    # ------------------------------------------------------------------ vocabulary
    def dictionary(self, min_share=0.002):
        rng = np.random.default_rng(self.seed)
        out = []
        if self.words is None:
            return out
        for w in range(self.V):
            m = self.words == w
            n = int(m.sum())
            if n == 0:
                out.append({"word": w, "n": 0, "share": 0.0})
                continue
            d = (self.Z[m].mean(0)) * np.sqrt(1.0)           # mean z-score = effect size vs the whole city
            score = np.where(self.ok, np.abs(d), 0.0)
            top = np.argsort(-score)[:3]
            traits = [phrase(self.names[j], d[j] > 0) for j in top if score[j] >= 0.25 and phrase(self.names[j], d[j] > 0)]
            ex = []
            if self.text is not None:
                pick = rng.choice(np.where(m)[0], min(3, n), replace=False)
                ex = [str(self.text[i]) for i in pick]
            out.append({"word": w, "n": n, "share": n / len(m), "change_rate": float(self.ds.y[m].mean()),
                        "lift": float(self.ds.y[m].mean() / max(self.overall, 1e-9)), "mean_risk": float(1 / (1 + np.exp(-self.base[m].mean()))),
                        "traits": traits, "examples": ex})
        return out

    def say_word(self, w, entry=None):
        e = entry or self.dictionary()[w]
        if not e.get("n"):
            return f"word {w}: unused"
        tr = "; ".join(e["traits"]) if e["traits"] else "no single trait stands out"
        return (f"word {w} ({100*e['share']:.1f}% of places, {100*e['change_rate']:.0f}% of them change, "
                f"{e['lift']:.1f}x the city rate): {tr}")

    # ------------------------------------------------------------------ one place
    def reasons(self, i, k=3):
        z = self.Z[i]
        score = np.where(self.ok, np.abs(z), 0.0)
        out = []
        for j in np.argsort(-score)[:k + 3]:
            if score[j] < 1.0:
                break
            p = phrase(self.names[j], z[j] > 0)
            if p:
                out.append(p)
            if len(out) == k:
                break
        return out

    def label(self, i):
        t = None if self.text is None else str(self.text[i])
        if not t:
            return f"place #{i}"
        nm, _, cat = t.partition(" | ")
        return f"'{nm}'" + (f" ({cat})" if cat else "")

    def explain(self, i, risk=None, env=None):
        parts = [f"{self.label(i)} is flagged"]
        if risk is not None:
            parts[0] += f" at {100*risk:.0f}% risk"
        why = self.reasons(i)
        s = parts[0] + (": " + "; ".join(why) if why else "")
        if self.words is not None:
            w = int(self.words[i])
            if self._dic is None:
                self._dic = self.dictionary()
            gist = "; ".join(self._dic[w].get("traits", [])[:2])
            s += f". The cells say word {w}" + (f" ({gist})" if gist else "")
        if env is not None:
            G = self.ds.graph()
            nbrs = G.Au[i].indices
            pos = int((env.verified[nbrs] == 2).sum()); neg = int((env.verified[nbrs] == 1).sum())
            if pos or neg:
                s += f". Among its {len(nbrs)} neighbours, {pos} verified changed and {neg} verified fine"
        return s + "."

    # ------------------------------------------------------------------ workers
    def worker_log(self, rec, env, risk_fn=None, limit=12):
        """Sentences about what individual workers did, from a recorded organism run (Organism(record=True))."""
        lines = []
        for r, entry in enumerate(rec):
            ids, labels = entry["ids"], entry["labels"]
            workers = entry.get("workers", [None] * len(ids))
            for w, i, y in list(zip(workers, ids, labels))[: max(1, limit // max(len(rec), 1))]:
                who = f"worker {w}" if w is not None else "a worker"
                res = "it had changed" if y == 1 else "nothing had changed"
                risk = None if risk_fn is None else risk_fn(i)
                lines.append(f"round {r+1}: {who} checked {self.label(i)}" +
                             (f" (risk {100*risk:.0f}%)" if risk is not None else "") +
                             f"; {res}. " + (("Why it looked suspect: " + "; ".join(self.reasons(i, 2)) + ".") if self.reasons(i, 2) else ""))
        return lines
