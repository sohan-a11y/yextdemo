"""The Sentinel's brain: instincts (rules) + reflex (learns this machine's normal) + forged neurons (made from a purpose, approved by a human).

All three can only produce alerts. Which alerts get through (de-duplication, cool-down, hourly cap, "this was fine" suppression) is decided
here in plain code, not by any neuron.
"""
import time
from collections import deque

from lkm.live.store import pack, unpack_blob

from .features import to_vector
from .instincts import Instincts, SEV, alert as mk_alert
from .neurons import KitEnsemble, ForgeNeuron

DEFAULT_COOLDOWN = {"info": 3600, "low": 1800, "medium": 600, "high": 300}


class Brain:
    def __init__(self, store, cfg, log=print, clock=time.time):
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
        vec = to_vector(raw)
        alerts = self.instincts.end_window(raw, events, now)
        for a in alerts:
            a["source"] = "instinct"
        serious = any(SEV[a["severity"]] >= SEV["medium"] for a in alerts)
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
