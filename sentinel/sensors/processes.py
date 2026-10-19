"""New programs: name, where they live, who started them, whether the command line looks encoded. Read-only (psutil)."""
import psutil

from . import Sensor
from ..constitution import redact, is_sensitive
from ..paths import classify_path, base_name, SCRIPT_HOSTS, LOLBINS, OFFICE_PARENTS, OBFUSCATION

MAX_SEEN = 5000


def describe(info, parent_name, seen_programs):
    """Pure function: process facts -> event. `info` has name, exe, cmdline, pid, ppid."""
    exe = info.get("exe") or ""
    name = base_name(exe or info.get("name") or "")
    cmd = " ".join(info.get("cmdline") or [])
    cls = classify_path(exe)
    parent = base_name(parent_name or "")
    sens = is_sensitive(exe)
    key = (exe or name).lower()
    ev = {"kind": "process_new", "pid": info.get("pid"), "ppid": info.get("ppid"), "name": name, "parent": parent, "exe_class": cls,
          "exe": "<sensitive>" if sens else redact(exe, 260), "cmd": redact(cmd, 300),
          "script_host": name in SCRIPT_HOSTS, "lolbin": name in LOLBINS, "obfuscated": bool(OBFUSCATION.search(cmd)),
          "office_parent": parent in OFFICE_PARENTS, "novel_program": key not in seen_programs}
    return ev, key


class ProcessSensor(Sensor):
    name = "processes"

    def __init__(self):
        self.known = None            # {(pid, create_time)}
        self.seen_programs = set()

    def poll(self, now=None):
        cur, names, rows, events = set(), {}, [], []
        for p in psutil.process_iter(["pid", "ppid", "name", "exe", "cmdline", "create_time"]):
            try:
                i = p.info
                cur.add((i["pid"], i["create_time"]))
                names[i["pid"]] = i.get("name") or ""
                rows.append(i)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        if self.known is None:                    # first look: this is the machine as it already was, not news
            self.known = cur
            for i in rows:
                self.seen_programs.add((i.get("exe") or base_name(i.get("name") or "")).lower())
            return []
        for i in rows:
            if (i["pid"], i["create_time"]) in self.known:
                continue
            ev, key = describe(i, names.get(i.get("ppid")), self.seen_programs)
            ev["ts"] = now
            events.append(ev)
            self.seen_programs.add(key)
        self.known = cur
        if len(self.seen_programs) > MAX_SEEN:
            self.seen_programs = set(list(self.seen_programs)[-MAX_SEEN // 2:])
        return events

    def state(self):
        return {"seen_programs": sorted(self.seen_programs)[-MAX_SEEN:]}

    def load(self, st):
        self.seen_programs = set(st.get("seen_programs", []))
