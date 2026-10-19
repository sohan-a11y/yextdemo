"""The living Sentinel: boots with the machine, watches, thinks, writes alerts, never acts.

Lifecycle: boot (single-instance lock, memory, reflex restored, sensors restored) -> step every `poll_s` seconds -> every `window_s` seconds
one window is closed, scored by instincts/reflex/forged neurons, and the alerts + the window + the reflex snapshot are saved in ONE
transaction -> shutdown (save, release lock). Kill switch: a file named PAUSE in the home directory, or POST /pause.
"""
import json
import os
import signal
import threading
import time

import numpy as np

from lkm.live.store import unpack_blob

from . import actuators
from .brain import Brain
from .constitution import ALLOWED_ACTIONS, check_action, own_path
from .features import WindowBuilder, to_vector
from .store import SentinelStore

DEFAULTS = {
    "window_s": 30, "poll_s": 5, "host": "127.0.0.1", "port": 8787, "token_env": "SENTINEL_TOKEN",
    "max_alerts_per_hour": 30, "snapshot_every": 10, "retention_days": 30, "seed": 0,
    "reflex": {"fm_windows": 120, "ad_windows": 240, "cal_windows": 240},
    "watch_dirs": None, "dns_sniff": False,
}
CANNOT_SEE = [
    "The content of encrypted traffic (nearly all web traffic): only who talks to whom, how much, and when.",
    "Other people's devices on your Wi-Fi beyond their presence; traffic that does not pass through this computer.",
    "What other programs show on screen or type (screen and keyboard reading are not part of this version).",
    "Anything while it is switched off, paused, or running without the rights to see other users' programs.",
]


def acquire_lock(path):
    f = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


class Sentinel:
    def __init__(self, home, sensors, cfg=None, clock=time.time, sleep=time.sleep, log=print):
        self.home = os.path.abspath(home)
        c = json.loads(json.dumps(DEFAULTS))
        for k, v in (cfg or {}).items():
            c[k] = {**c[k], **v} if isinstance(v, dict) and isinstance(c.get(k), dict) else v
        self.cfg, self.sensors, self.clock, self.sleep, self._print = c, list(sensors), clock, sleep, log
        self.mx = threading.RLock()
        self.stop = threading.Event()
        self._lock = None
        self.window_events = []
        self.pending = []
        self.sensor_errors = {}
        self.cycle = 0

    def log(self, msg):
        """Print it, and keep it in the database too (a background Windows task has no console to print to)."""
        self._print(msg)
        try:
            self.store.event("log", {"msg": msg})
        except Exception:
            pass

    # ------------------------------------------------------------------ lifecycle
    def boot(self):
        os.makedirs(self.home, exist_ok=True)
        self._lock = acquire_lock(own_path(self.home, "lock"))
        if self._lock is None:
            raise RuntimeError(f"another Sentinel is already running in {self.home}")
        self.store = SentinelStore(self.home)
        self.brain = Brain(self.store, self.cfg, self.log, self.clock)
        self.builder = WindowBuilder()
        st = self.store.meta_get("sensor_state", {})
        for s in self.sensors:
            s.load(st.get(s.name, {}))
        self.boots = self.store.meta_get("boots", 0) + 1
        self.store.meta_set("boots", self.boots)
        self.cycle = (self.store.load_snapshot() or {"cycle": 0})["cycle"]
        self.boot_ts = self.clock()
        self.window_start = self.boot_ts
        self.last_step = self.boot_ts
        self.store.event("boot", {"boots": self.boots, "phase": self.brain.kit.phase, "cycle": self.cycle}, ts=self.boot_ts)
        self.log(f"[sentinel] awake (boot #{self.boots}); reflex phase: {self.brain.kit.phase}; {len(self.sensors)} sensors")
        return self

    def shutdown(self):
        try:
            self._save(self.clock())
            self.store.event("shutdown", {"cycle": self.cycle}, ts=self.clock())
        finally:
            for s in self.sensors:
                try:
                    s.close()
                except Exception:
                    pass
            if self._lock:
                self._lock.close()
                self._lock = None
            self.store.close()

    # ------------------------------------------------------------------ pause (kill switch)
    def paused_by(self):
        if os.path.exists(os.path.join(self.home, "PAUSE")):
            return "file"
        return "api" if self.store.meta_get("paused", False) else None

    def set_paused(self, on):
        """The API can pause and resume. A PAUSE file placed by a person always wins and only a person can remove it."""
        self.store.meta_set("paused", bool(on))

    # ------------------------------------------------------------------ one tick
    def step(self, now=None):
        with self.mx:
            now = self.clock() if now is None else now
            if self.paused_by():
                self.window_start = now
                return {"paused": True}
            events = []
            for s in self.sensors:
                try:
                    events += s.poll(now)
                    self.sensor_errors.pop(s.name, None)
                except Exception as ex:
                    if s.name not in self.sensor_errors:
                        self.log(f"[sentinel] sensor {s.name} failed: {ex}")
                    self.sensor_errors[s.name] = str(ex)[:200]
            for ev in events:
                self.builder.add(ev)
                if len(self.window_events) < 5000:
                    self.window_events.append(ev)
                self.pending += self.brain.observe(ev, now)
            if now - self.window_start >= self.cfg["window_s"]:
                return self._close(now)
            return {"events": len(events)}

    def _close(self, now):
        raw = self.builder.close()
        alerts, score, flagged = self.brain.close_window(raw, self.window_events, now)
        alerts = self.pending + alerts
        self.pending, self.window_events, self.window_start = [], [], now
        admitted = []
        for a in alerts:
            check_action("alert")
            if self.brain.admit(a, now):
                admitted.append(a)
        self.cycle += 1
        with self.store.tx():
            wid = self.store.add_window(now, raw, score, flagged)
            for a in admitted:
                a["id"] = self.store.add_alert(a, now, wid)
                prop = actuators.propose(a)
                if prop:
                    check_action("request_action")
                    self.store.add_request(a["id"], prop[0], prop[1], prop[2], now)
            if self.cycle % self.cfg["snapshot_every"] == 0:
                self._save(now)
        if self.cycle % 500 == 0:
            self.store.prune(now, self.cfg["retention_days"])
        return {"window": wid, "alerts": admitted, "score": score, "flagged": flagged, "raw": raw}

    def _save(self, now):
        self.brain.save(self.cycle, now)
        self.store.meta_set("sensor_state", {s.name: s.state() for s in self.sensors})

    # ------------------------------------------------------------------ feedback from the human
    def feedback(self, alert_id, verdict):
        """verdict: 'benign' (it was fine: stop telling me, learn this as normal) or 'real' (it was a true alarm)."""
        if verdict not in ("benign", "real"):
            raise ValueError("verdict must be 'benign' or 'real'")
        with self.mx:
            a = self.store.alert(int(alert_id))
            if a is None:
                return "unknown alert"
            w = self.store.window(a["window_id"]) if a["window_id"] else None
            vec = to_vector(w["raw"]) if w else None
            if verdict == "benign":
                self.brain.learn_benign(a["key"], vec if a["source"] in ("reflex",) or a["source"].startswith("neuron:") else None)
                if w:
                    self.store.set_window_label(w["id"], "benign")
            elif w:
                self.store.set_window_label(w["id"], "real")
            self.store.judge(a["id"], verdict, self.clock())
            self.brain.save(self.cycle, self.clock())
            return "ok"

    # ------------------------------------------------------------------ views
    def status(self):
        with self.mx:
            counts = self.store.alert_counts()
            real = sum(c["n"] for c in counts if c["status"] == "real")
            benign = sum(c["n"] for c in counts if c["status"] == "benign")
            k = self.brain.kit
            return {"home": self.home, "boots": self.boots, "alive_for_s": self.clock() - self.boot_ts, "windows": self.store.n_windows(), "cycle": self.cycle,
                    "reflex": {"phase": k.phase, "windows_seen": k.n, "threshold": k.thr,
                               "childhood_windows": k.fm + k.ad + k.cal},
                    "sensors": [{**s.info(), "error": self.sensor_errors.get(s.name)} for s in self.sensors],
                    "alerts": counts, "judged": {"real": real, "benign": benign, "precision": (real / (real + benign)) if real + benign else None},
                    "dropped_by_cap": self.brain.dropped, "open_requests": len(self.store.requests("open")),
                    "neurons": [{"name": n["name"], "version": n["version"], "state": n["state"]} for n in self.store.neurons()],
                    "paused": self.paused_by(), "can_do": list(ALLOWED_ACTIONS), "cannot_see": CANNOT_SEE}

    # ------------------------------------------------------------------ run
    def run_forever(self, max_steps=None, serve=True):
        for s in (signal.SIGINT, getattr(signal, "SIGTERM", None), getattr(signal, "SIGBREAK", None)):
            if s is not None:
                try:
                    signal.signal(s, lambda *_: self.stop.set())
                except (ValueError, OSError):
                    pass
        srv = None
        if serve:
            from .serve import start_server
            srv = start_server(self)
        n = 0
        try:
            while not self.stop.is_set() and (max_steps is None or n < max_steps):
                t0 = self.clock()
                self.step()
                n += 1
                self.sleep(max(self.cfg["poll_s"] - (self.clock() - t0), 0.05))
        finally:
            if srv:
                srv.shutdown()
                srv.server_close()
            self.shutdown()
