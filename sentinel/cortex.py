"""The cortex: knowledge borrowed from a pretrained language model, and neurons that can be made in milliseconds on top of it.

  Cortex          a frozen pretrained sentence encoder (all-MiniLM-L6-v2, 22M parameters, ONNX, CPU). Every event the senses report is
                  written as a short sentence and turned into a 384-number meaning vector. The encoder is never trained here: it is the
                  organism's inherited knowledge, like an animal's visual cortex at birth.
  ConceptNeuron   one new neuron = one direction in that meaning space. It is BORN from a plain-language description (and, if you have them,
                  a few example events) in one encoder call, CALIBRATED on the machine's own recent events so that it fires on at most a
                  chosen fraction of them, and it LEARNS ONE-SHOT from feedback: "this was real" adds the event to its examples, "this was
                  fine" adds it to its counter-examples. No gradient training, no dataset, no waiting.
  NoveltyMemory   neurons that grow by themselves: an adaptive-resonance style memory of what normal events look like on this machine. An
                  event that resembles no existing prototype closely enough makes a NEW prototype neuron; a familiar one strengthens its
                  winner. Its novelty score says "I have never seen anything like this here". Capacity is bounded by merging the two most
                  similar prototypes (consolidation).

Like every neuron in the Sentinel, these only produce numbers; what the organism may do with them is fixed in constitution.py.
"""
import json
import os

import numpy as np

DIM = 384


def unit(v):
    v = np.asarray(v, np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


# ============================================================================================ the borrowed encoder
def default_model_dir():
    """Where the language model is installed (python -m sentinel fetch-model). The running Sentinel never downloads anything."""
    env = os.environ.get("SENTINEL_MODEL_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lkm_data", "minilm", "onnx")


class Cortex:
    PARAMS = 22_713_216          # all-MiniLM-L6-v2: inherited, frozen

    def __init__(self, model_dir=None, max_len=64, threads=2):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        model_dir = model_dir or default_model_dir()
        if not os.path.exists(os.path.join(model_dir, "model.onnx")):
            raise FileNotFoundError(f"language model not installed in {model_dir}: run  python -m sentinel fetch-model")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(os.path.join(model_dir, "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(length=None)
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.outputs = [o.name for o in self.sess.get_outputs()]

    def run(self, texts, names=None):
        """Raw model outputs for a batch, plus the attention mask (used by research code to read inner layers)."""
        enc = self.tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc], np.int64)
        mask = np.array([e.attention_mask for e in enc], np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        return self.sess.run(names, feed), mask

    def encode(self, texts, batch=128):
        """Mean-pooled, unit-length sentence vectors (n, 384)."""
        texts = list(texts)
        out = np.zeros((len(texts), DIM), np.float32)
        order = np.argsort([len(t) for t in texts])
        for s in range(0, len(texts), batch):
            ix = order[s:s + batch]
            (h,), mask = self.run([texts[i] for i in ix], [self.outputs[0]])
            m = mask[..., None].astype(np.float32)
            out[ix] = unit((h * m).sum(1) / np.maximum(m.sum(1), 1.0))
        return out


# ============================================================================================ events as sentences
def event_sentence(ev):
    """What the cortex reads for one sensor event. Only already-redacted fields are used; None = not worth reading."""
    k = ev.get("kind")
    if k == "process_new":
        cmd = ev.get("cmd") or ev.get("name") or ""
        return f"{ev.get('name', '')} started by {ev.get('parent') or 'unknown'} from the {ev.get('exe_class', 'other')} folder: {cmd}"
    if k == "conn_new":
        where = "the local network" if ev.get("private") else "the internet"
        return f"{ev.get('prog', 'a program')} ({ev.get('exe_class', 'other')} folder) connected to {where} on port {ev.get('remote_port')}"
    if k == "listen_new":
        scope = "all network interfaces" if ev.get("public_bind") else "this computer only"
        return f"{ev.get('prog', 'a program')} ({ev.get('exe_class', 'other')} folder) opened port {ev.get('port')} for {scope}"
    if k == "file":
        if ev.get("sensitive"):
            return None
        name = str(ev.get("path", "")).replace("\\", "/").rsplit("/", 1)[-1]
        return f"file {ev.get('op')} in the {ev.get('loc', 'other')} folder: {name}"
    if k == "dns":
        return f"name lookup {ev.get('name', '')}" + (" (failed)" if ev.get("nx") else "")
    if k == "lan":
        return f"local network change: {ev.get('new_devices', 0)} new devices, router changed {bool(ev.get('gw_changed'))}"
    if k == "wifi":
        return f"wifi {ev.get('ssid', '')} security {ev.get('auth', '')} {ev.get('cipher', '')}"
    return None


# ============================================================================================ instant neurons
class ConceptNeuron:
    """A neuron made from words. Score = similarity to what it should fire on, minus similarity to the closest thing it was told is fine."""

    def __init__(self, name, description, desc_vec, examples=None, counter=None, desc_weight=1.0, alpha=0.001, max_counter=500):
        self.name, self.description = name, description
        self.desc_vec = unit(desc_vec)
        self.examples = [unit(v) for v in (examples if examples is not None else [])]
        self.counter = [unit(v) for v in (counter if counter is not None else [])]
        self.desc_weight, self.alpha, self.max_counter = desc_weight, alpha, max_counter
        self.thr = None
        self.n_feedback = 0
        self._update()

    def _update(self):
        parts = [self.desc_vec * self.desc_weight] + self.examples
        self.proto = unit(np.sum(parts, axis=0))
        self.C = np.array(self.counter, np.float32) if self.counter else None

    def score_many(self, X):
        X = np.asarray(X, np.float32)
        s = X @ self.proto
        if self.C is not None:
            s = s - np.clip(X @ self.C.T, 0, None).max(1) * 0.5
        return s

    def score(self, x):
        return float(self.score_many(np.asarray(x)[None])[0])

    def calibrate(self, normal_vectors, alpha=None):
        """Alarm line: fire on at most `alpha` of this machine's recent normal events."""
        self.alpha = alpha if alpha is not None else self.alpha
        s = self.score_many(normal_vectors)
        self.thr = float(np.quantile(s, 1.0 - self.alpha)) + 1e-6 if len(s) else 0.5
        return self.thr

    def fires(self, x):
        return self.thr is not None and self.score(x) > self.thr

    def learn(self, x, real):
        """One-shot feedback. real=True: this is what you are for. real=False: this was fine, do not fire on things like it."""
        self.n_feedback += 1
        if real:
            self.examples.append(unit(x))
        else:
            self.counter.append(unit(x))
            if len(self.counter) > self.max_counter:
                self.counter = self.counter[-self.max_counter:]
        self._update()

    def to_dict(self):
        return {"name": self.name, "description": self.description, "desc_vec": self.desc_vec.tolist(), "examples": [v.tolist() for v in self.examples],
                "counter": [v.tolist() for v in self.counter], "desc_weight": self.desc_weight, "alpha": self.alpha, "thr": self.thr,
                "n_feedback": self.n_feedback}

    @classmethod
    def from_dict(cls, d):
        o = cls(d["name"], d["description"], np.array(d["desc_vec"]), [np.array(v) for v in d["examples"]], [np.array(v) for v in d["counter"]],
                d.get("desc_weight", 1.0), d.get("alpha", 0.001))
        o.thr, o.n_feedback = d.get("thr"), d.get("n_feedback", 0)
        return o

    @property
    def n_params(self):
        return DIM * (1 + len(self.examples) + len(self.counter)) + 1


def make_neuron(cortex, name, description, example_texts=(), normal_vectors=None, alpha=0.001):
    """Description (+ optional examples) -> calibrated neuron. One encoder call; milliseconds on a laptop CPU."""
    vecs = cortex.encode([description] + list(example_texts))
    n = ConceptNeuron(name, description, vecs[0], list(vecs[1:]), alpha=alpha)
    if normal_vectors is not None and len(normal_vectors):
        n.calibrate(normal_vectors, alpha)
    return n


# ============================================================================================ neurons that grow by themselves
class NoveltyMemory:
    """Adaptive-resonance style prototype memory over event vectors.

    vigilance: how similar (cosine) an event must be to a prototype to count as 'seen before'. Below it, a new prototype neuron is born.
    capacity:  maximum number of prototypes; when full, the two most similar ones are merged (weighted by how often each fired).
    Each prototype remembers its most similar neighbour, so consolidation costs one pass over the memory instead of all pairs.
    """

    def __init__(self, vigilance=0.80, capacity=2000, lr_min=0.02):
        self.vigilance, self.capacity, self.lr_min = vigilance, capacity, lr_min
        self.P = np.zeros((0, DIM), np.float32)
        self.count = np.zeros(0, np.int64)
        self.nn_sim = np.zeros(0, np.float32)
        self.nn_idx = np.zeros(0, np.int64)
        self.born = 0
        self.merged = 0
        self.seen = 0

    def similarity(self, X):
        X = np.atleast_2d(np.asarray(X, np.float32))
        if len(self.P) == 0:
            return np.full(len(X), -1.0, np.float32), np.full(len(X), -1)
        S = X @ self.P.T
        j = S.argmax(1)
        return S[np.arange(len(X)), j], j

    def novelty(self, X):
        """0 = exactly like something seen often; up to 2 = unlike anything seen. Rare prototypes count as less familiar."""
        out = []
        X = np.atleast_2d(np.asarray(X, np.float32))
        for s0 in range(0, len(X), 4096):
            s, j = self.similarity(X[s0:s0 + 4096])
            if len(self.P) == 0:
                out.append(np.ones(len(s), np.float32))
                continue
            rarity = 1.0 / np.sqrt(1.0 + self.count[j])
            out.append((1.0 - s) + 0.1 * rarity)
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def _refresh_nn(self, i):
        s = self.P @ self.P[i]
        s[i] = -2.0
        j = int(s.argmax())
        self.nn_sim[i], self.nn_idx[i] = s[j], j

    def _moved(self, j):
        """Prototype j changed: fix its own nearest neighbour and everyone whose nearest neighbour it is or has become."""
        if len(self.P) < 2:
            self.nn_sim[:] = -2.0; self.nn_idx[:] = -1
            return
        s = self.P @ self.P[j]
        s[j] = -2.0
        k = int(s.argmax())
        self.nn_sim[j], self.nn_idx[j] = s[k], k
        was = self.nn_idx == j
        upd = s > self.nn_sim
        upd[j] = False
        self.nn_sim[upd] = s[upd]; self.nn_idx[upd] = j
        for i in np.where(was & ~upd)[0]:
            if i != j:
                self._refresh_nn(int(i))

    def observe(self, x):
        x = unit(x)
        self.seen += 1
        if len(self.P):
            s_all = self.P @ x
            j = int(s_all.argmax())
            if s_all[j] >= self.vigilance:
                self.count[j] += 1
                lr = max(1.0 / self.count[j], self.lr_min)
                self.P[j] = unit(self.P[j] + lr * (x - self.P[j]))
                self._moved(j)
                return False
        if len(self.P) >= self.capacity:
            self._consolidate()
        self.P = np.vstack([self.P, x[None]])
        self.count = np.append(self.count, 1)
        self.nn_sim = np.append(self.nn_sim, np.float32(-2.0)); self.nn_idx = np.append(self.nn_idx, -1)
        self._moved(len(self.P) - 1)
        self.born += 1
        return True

    def observe_many(self, X):
        return sum(self.observe(x) for x in np.asarray(X, np.float32))

    def _consolidate(self):
        a = int(self.nn_sim.argmax())
        b = int(self.nn_idx[a])
        if b < 0:
            return
        a, b = min(a, b), max(a, b)
        ca, cb = self.count[a], self.count[b]
        self.P[a] = unit((self.P[a] * ca + self.P[b] * cb) / (ca + cb))
        self.count[a] = ca + cb
        keep = np.ones(len(self.P), bool); keep[b] = False
        pointed_b = (self.nn_idx == b) & keep
        self.P, self.count, self.nn_sim, self.nn_idx = self.P[keep], self.count[keep], self.nn_sim[keep], self.nn_idx[keep]
        pointed_b = pointed_b[keep]
        self.nn_idx = np.where(self.nn_idx > b, self.nn_idx - 1, self.nn_idx)
        for i in np.where(pointed_b)[0]:
            self._refresh_nn(int(i))
        self._moved(a)
        self.merged += 1

    def to_arrays(self):
        return {"P": self.P, "count": self.count, "meta": np.frombuffer(json.dumps(
            {"vigilance": self.vigilance, "capacity": self.capacity, "lr_min": self.lr_min, "born": self.born, "merged": self.merged, "seen": self.seen}).encode(), np.uint8)}

    @classmethod
    def from_arrays(cls, a):
        m = json.loads(bytes(a["meta"].astype(np.uint8)).decode())
        o = cls(m["vigilance"], m["capacity"], m["lr_min"])
        o.P, o.count = a["P"].astype(np.float32).copy(), a["count"].astype(np.int64).copy()
        o.nn_sim = np.zeros(len(o.P), np.float32); o.nn_idx = np.zeros(len(o.P), np.int64)
        for i in range(len(o.P)):
            if len(o.P) > 1:
                o._refresh_nn(i)
            else:
                o.nn_sim[i], o.nn_idx[i] = -2.0, -1
        o.born, o.merged, o.seen = m["born"], m["merged"], m["seen"]
        return o
