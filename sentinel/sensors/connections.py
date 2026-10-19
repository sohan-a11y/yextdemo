"""Who is talking to whom: new outbound connections and new listening ports, per program. Addresses and ports only, never contents."""
import psutil

from . import Sensor
from ..paths import classify_path, base_name, is_private_ip, COMMON_PORTS

MAX_DEST = 20000


class ConnectionSensor(Sensor):
    name = "connections"

    def __init__(self):
        self.active = None           # {(pid, ip, port)}
        self.listening = None        # {(pid, port)}
        self.dest = {}               # program -> set of remote ips it has contacted before
        self._names = {}

    def _prog(self, pid):
        if pid in self._names:
            return self._names[pid]
        try:
            p = psutil.Process(pid)
            v = (base_name(p.name()), classify_path(p.exe() or ""), p.exe() or "")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
            v = ("unknown", "other", "")
        self._names[pid] = v
        return v

    def poll(self, now=None):
        try:
            conns = psutil.net_connections(kind="inet")
        except (psutil.AccessDenied, OSError) as ex:
            self.available, self.reason = False, f"cannot list connections: {ex}"
            return []
        self.available, self.reason = True, ""
        self._names = {}
        act, lis = set(), set()
        for c in conns:
            if c.pid is None:
                continue
            if c.status == psutil.CONN_LISTEN:
                lis.add((c.pid, c.laddr.port, c.laddr.ip))
            elif c.raddr and c.status in (psutil.CONN_ESTABLISHED, psutil.CONN_SYN_SENT):
                act.add((c.pid, c.raddr.ip, c.raddr.port))
        first = self.active is None
        events = []
        if not first:
            for pid, ip, port in sorted(act - self.active):
                prog, cls, exe = self._prog(pid)
                seen = self.dest.setdefault(prog, set())
                events.append({"kind": "conn_new", "pid": pid, "prog": prog, "exe_class": cls, "remote_ip": ip, "remote_port": port,
                               "private": is_private_ip(ip), "rare_port": port not in COMMON_PORTS, "novel": ip not in seen, "ts": now})
            for pid, port, ip in sorted(lis - self.listening):
                prog, cls, exe = self._prog(pid)
                events.append({"kind": "listen_new", "pid": pid, "prog": prog, "exe_class": cls, "port": port,
                               "public_bind": ip in ("0.0.0.0", "::", ""), "ts": now})
        for pid, ip, port in act:
            self.dest.setdefault(self._prog(pid)[0], set()).add(ip)
        self.active = {(p, i, q) for p, i, q in act}
        self.listening = {(p, q, i) for p, q, i in lis}
        return events

    def state(self):
        d = {k: sorted(v)[-200:] for k, v in list(self.dest.items())[-500:]}
        return {"dest": d}

    def load(self, st):
        self.dest = {k: set(v) for k, v in st.get("dest", {}).items()}
