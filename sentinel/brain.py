"""The Sentinel's brain, in five parts, from inherited to learned:

  cortex     knowledge inherited from a pretrained language model (frozen): reads every event as a sentence
  instincts  rules people wrote
  reflex     learns what normal looks like on this machine, window by window
  memory     grows a new prototype neuron whenever an event resembles nothing seen before (novelty)
  concepts   neurons made in milliseconds from a sentence you write; they learn one-shot from your answers
  forged     slower neurons from the Forge, trained and gated on window statistics

All of them can only produce alerts. Which alerts get through (de-duplication, cool-down, hourly cap, "this was fine" suppression) is decided
here in plain code, not by any neuron.
"""
import time
from collections import deque

from lkm.live.store import pack, unpack_blob

import numpy as np

from .cortex import ConceptNeuron, NoveltyMemory, event_sentence
from .features import to_vector
from .instincts import Instincts, SEV, alert as mk_alert
from .neurons import KitEnsemble, ForgeNeuron

# vigilance 0.7 with room for 8,000 prototypes matched an exact nearest-neighbour memory in the benchmark (research/results) at half the storage
CORTEX_DEFAULTS = {"enabled": True, "model_dir": None, "childhood_events": 2000, "novel_threshold": 0.35, "vigilance": 0.70, "capacity": 8000,
                   "max_per_window": 256, "min_calibration": 1000}

DEFAULT_COOLDOWN = {"info": 3600, "low": 1800, "medium": 600, "high": 300}


class Brain:
    def __init__(self, store, cfg, log=print, clock=time.time, cortex=None):
        self.store, self.cfg, self.log, self.clock = store, cfg, log, clock
        self.instincts = Instincts(cfg.get("instincts"))
        snap = store.load_snapshot()
        self.kit = None
        if snap:
            try:
                self.kit = KitEnsemble.from_arrays(unpack_blob(snap["blob"]))
            except Exception as ex:
                log(f"[sentinel] could not restore the reflex ({ex}); it starts its childhood again")
        if self.kit is None:
            k = cfg["reflex"]
            self.kit = KitEnsemble(fm=k["fm_windows"], ad=k["ad_windows"], cal=k["cal_windows"], seed=cfg.get("seed", 0))
        self.neurons, self._stamp = [], None
        self.flags = deque(maxlen=3)
        self.cool = store.meta_get("cooldowns", {})
        self.suppress = {k: v for k, v in store.meta_get("suppress", {}).items() if v > clock()}
        for k, until in self.suppress.items():
            self.instincts.suppress(k, until)
        self.sent = deque(store.meta_get("sent_times", []))
        self.dropped = 0
        self._init_cortex(cortex)

    # ------------------------------------------------------------------ cortex, memory and concept neurons
    def _init_cortex(self, cortex):
        c = {**CORTEX_DEFAULTS, **(self.cfg.get("cortex") or {})}
        self.ccfg = c
        self.cortex, self.cortex_reason = cortex, ""
        if self.cortex is None and c["enabled"]:
            try:
                from .cortex import Cortex
                self.cortex = Cortex(c["model_dir"])
            except Exception as ex:
                self.cortex_reason = str(ex)[:300]
                self.log(f"[sentinel] cortex not available ({self.cortex_reason}); the other parts of the brain work without it")
        self.memory = None
        blob = self.store.get_blob("novelty_memory")
        if blob:
            try:
                self.memory = NoveltyMemory.from_arrays(unpack_blob(blob))
            except Exception as ex:
                self.log(f"[sentinel] could not restore the novelty memory ({ex}); it starts empty")
        if self.memory is None:
            self.memory = NoveltyMemory(vigilance=c["vigilance"], capacity=c["capacity"])
        self.concepts, self._cstamp = [], None
        self.window_vectors = []

    def reload_concepts(self):
        stamp = self.store.concept_stamp()
        if stamp == self._cstamp and all(n.thr is not None for n in self.concepts):
            return
        out = []
        need_cal = []
        for row in self.store.concepts("active"):
            try:
                n = ConceptNeuron.from_dict(row["data"])
            except Exception as ex:
                self.log(f"[sentinel] concept neuron {row['name']} could not be loaded: {ex}")
                continue
            n.db_id, n.severity = row["id"], row["severity"] or "medium"
            out.append(n)
            if n.thr is None:
                need_cal.append(n)
        if need_cal:
            ids, V, _, _ = self.store.recent_vectors(20000)
            if len(V) >= self.ccfg["min_calibration"]:
                for n in need_cal:
                    n.calibrate(V)
                    self.store.update_concept(n.db_id, n.to_dict())
        self.concepts, self._cstamp = out, stamp

    def _read_events(self, events, raw, now, serious_so_far):
        """The cortex reads the window's events; concept neurons fire; the novelty memory scores and then learns."""
        self.window_vectors = []
        raw["n_novel_events"] = 0.0
        if self.cortex is None:
            return []
        items = []
        for ev in events:
            s = event_sentence(ev)
            if s:
                items.append((ev, s))
            if len(items) >= self.ccfg["max_per_window"]:
                break
        if not items:
            return []
        V = self.cortex.encode([s for _, s in items])
        alerts, fired = [], set()
        grown_up = self.memory.seen >= self.ccfg["childhood_events"]
        nov = self.memory.novelty(V) if grown_up else np.zeros(len(V), np.float32)
        raw["n_novel_events"] = float((nov > self.ccfg["novel_threshold"]).sum())
        self.reload_concepts()
        for n in self.concepts:
            if n.thr is None:
                continue
            sc = n.score_many(V)
            for i in np.where(sc > n.thr)[0]:
                ev, sent = items[int(i)]
                t = {k: ev[k] for k in ("pid",) if isinstance(ev.get(k), int)}
                if ev.get("kind") == "conn_new":
                    t = {"kind": "connection", "pid": ev.get("pid"), "ip": ev.get("remote_ip")}
                elif ev.get("kind") == "process_new":
                    t = {"kind": "process", "pid": ev.get("pid")}
                a = mk_alert(f"concept:{n.name}", n.severity, f"{n.name}: {sent[:90]}",
                             f"Your neuron '{n.name}' ({n.description}) recognised this event (match {sc[i]:.2f}, alarm line {n.thr:.2f})."
                             + (f" It is also new to this machine." if nov[i] > self.ccfg["novel_threshold"] else ""),
                             f"{n.name}:{sent[:60]}", [sent], t)
                a.update(source=f"concept:{n.name}", concept_id=n.db_id, _vec_index=int(i), confidence=float(sc[i]))
                alerts.append(a)
                fired.add(int(i))
        if not serious_so_far:
            for i, v in enumerate(V):
                if i not in fired:
                    self.memory.observe(v)
        self.window_vectors = [(ev["kind"], s, V[i]) for i, (ev, s) in enumerate(items)]
        return alerts

    def concept_feedback(self, concept_id, vec, real):
        for n in self.concepts:
            if n.db_id == concept_id:
                n.learn(vec, real)
                self.store.update_concept(concept_id, n.to_dict())
                break
        if not real:
            self.memory.observe(vec)

    def brain_map(self):
        kit_params = 0
        if self.kit.aes:
            kit_params = sum(ae.W.size + ae.b.size + ae.c.size for ae in self.kit.aes + [self.kit.out])
        return {"cortex_inherited_params": (self.cortex.PARAMS if self.cortex is not None else 0), "cortex": "on" if self.cortex is not None else f"off: {self.cortex_reason}",
                "rules": 10, "reflex_params": int(kit_params), "reflex_phase": self.kit.phase,
                "memory_prototypes": int(len(self.memory.P)), "memory_events_seen": int(self.memory.seen), "memory_born": int(self.memory.born),
                "memory_childhood_left": max(0, self.ccfg["childhood_events"] - self.memory.seen),
                "concept_neurons": len(self.concepts), "concept_params": int(sum(n.n_params for n in self.concepts)),
                "concept_feedback_answers": int(sum(n.n_feedback for n in self.concepts)), "forged_neurons": len(self.neurons)}

    # ------------------------------------------------------------------ forged neurons
    def reload_neurons(self):
        stamp = self.store.neuron_stamp()
        if stamp == self._stamp:
            return
        out = []
        for n in self.store.neurons("active"):
            blob = self.store.neuron_weights(n["id"])
            if blob:
                try:
                    out.append(ForgeNeuron.from_blob(blob, n["name"], n["spec"], n["version"]))
                except Exception as ex:
                    self.log(f"[sentinel] neuron {n['name']} v{n['version']} could not be loaded: {ex}")
        self.neurons, self._stamp = out, stamp

    # ------------------------------------------------------------------ events and windows
    def observe(self, ev, now):
        out = self.instincts.observe(ev, now)
        for a in out:
            a["source"] = "instinct"
        return out

    def close_window(self, raw, events, now):
        """Returns (alerts, reflex_score, flagged). `flagged` = this window is NOT to be treated as normal by anyone."""
        self.reload_neurons()
        alerts = self.instincts.end_window(raw, events, now)
        for a in alerts:
            a["source"] = "instinct"
        serious = any(SEV[a["severity"]] >= SEV["medium"] for a in alerts)
        alerts += self._read_events(events, raw, now, serious)
        vec = to_vector(raw)
        score, info = self.kit.process(vec, learn=not serious)
        flag = False
        if score is not None:
            ratio = score / max(info["threshold"], 1e-9)
            flag = ratio > 1.0
            persistent = flag and any(self.flags)
            self.flags.append(flag)
            if persistent or ratio > 3.0:                       # two windows in a row over the line, or one window far over it
                sev = "high" if ratio >= 3.0 else "medium" if ratio >= 1.5 else "low"
                why = self.kit.explain(info, raw)
                alerts.append(dict(mk_alert("reflex_unusual", sev, "This machine is behaving unlike its usual self",
                                            "Compared with what it has learned as normal here, this period is " + f"{ratio:.1f}x past the alarm line. "
                                            + ("Standing out: " + "; ".join(why) + "." if why else ""), "reflex", why), source="reflex"))
        for n in self.neurons:
            p = n.score(vec)
            if p >= n.thr:
                sp = n.spec.get("alert", {})
                why = n.explain(vec, raw)
                alerts.append(dict(mk_alert(f"neuron:{n.name}", sp.get("severity", "medium"), sp.get("title", n.name),
                                            (sp.get("explanation", "") + (" Standing out: " + "; ".join(why) + "." if why else "")).strip(), n.name, why),
                                   source=f"neuron:{n.name}@v{n.version}", confidence=p))
        return alerts, score, serious or flag or any(SEV[a["severity"]] >= SEV["medium"] for a in alerts)

    # ------------------------------------------------------------------ which alerts get through
    def admit(self, a, now):
        key = a["key"]
        if self.suppress.get(key, 0) > now:
            return False
        if now - self.cool.get(key, -1e18) < DEFAULT_COOLDOWN[a["severity"]]:
            return False
        while self.sent and now - self.sent[0] > 3600:
            self.sent.popleft()
        if len(self.sent) >= self.cfg["max_alerts_per_hour"]:
            self.dropped += 1
            return False
        self.cool[key] = now
        self.sent.append(now)
        return True

    # ------------------------------------------------------------------ feedback
    def learn_benign(self, key, vec=None, days=30):
        until = self.clock() + days * 86400
        self.suppress[key] = until
        self.instincts.suppress(key, until)
        if vec is not None:
            self.kit.teach_normal(vec)

    # ------------------------------------------------------------------ persistence
    def save(self, cycle, ts):
        self.store.save_snapshot(cycle, ts, pack(self.kit.to_arrays()))
        self.cool = {k: v for k, v in self.cool.items() if ts - v < 86400}
        self.store.meta_set("cooldowns", self.cool)
        self.store.meta_set("suppress", self.suppress)
        self.store.meta_set("sent_times", list(self.sent))
        self.store.put_blob("novelty_memory", pack(self.memory.to_arrays()), ts)
