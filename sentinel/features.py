"""From events to what the neurons see: one fixed-length vector per time window (default 30 s)."""
import math
import numpy as np

FEATURES = [
    "n_proc_new", "n_proc_user_dir", "n_script_host", "n_obfuscated", "n_lolbin", "n_office_child",
    "n_conn_new", "n_conn_novel", "n_remote_ips", "n_rare_port", "n_public_conn", "n_listen_new", "n_listen_public",
    "n_file_events", "n_file_created", "n_exec_created", "n_startup_touch", "n_file_renamed", "n_ext_change",
    "dns_queries", "dns_nx", "dns_entropy", "net_bytes_out", "net_bytes_in", "net_pkts",
    "n_lan_new", "gw_changed", "arp_dup", "wifi_weak", "wifi_ap_changed", "n_entropy_anomaly", "n_novel_events",
]
DESCRIPTIONS = {
    "n_proc_new": "new processes started", "n_proc_user_dir": "new processes whose program sits in a user, temp, downloads or appdata folder",
    "n_script_host": "script interpreters started (powershell, cmd, wscript, python ...)", "n_obfuscated": "command lines that look encoded or obfuscated",
    "n_lolbin": "built-in system tools often abused for downloads (certutil, mshta, rundll32 ...)", "n_office_child": "programs started by office or pdf software",
    "n_conn_new": "new outbound connections", "n_conn_novel": "connections to a destination this program has never contacted before",
    "n_remote_ips": "distinct remote addresses contacted", "n_rare_port": "connections to uncommon ports", "n_public_conn": "connections to public (non-local) addresses",
    "n_listen_new": "new listening ports", "n_listen_public": "new ports listening on all network interfaces",
    "n_file_events": "file changes seen in watched folders", "n_file_created": "files created", "n_exec_created": "program or script files created",
    "n_startup_touch": "changes in start-up locations", "n_file_renamed": "files renamed", "n_ext_change": "renames that changed the file extension",
    "dns_queries": "name lookups", "dns_nx": "name lookups that failed", "dns_entropy": "randomness of looked-up names (high looks machine-generated)",
    "net_bytes_out": "bytes sent", "net_bytes_in": "bytes received", "net_pkts": "packets",
    "n_lan_new": "devices newly seen on the local network", "gw_changed": "the router's hardware address changed",
    "n_entropy_anomaly": "text-like files whose content suddenly looks random (encrypted)",
    "n_novel_events": "events unlike anything this machine has done before (judged by the cortex)",
    "arp_dup": "one hardware address claiming several network addresses (a spoofing sign)",
    "wifi_weak": "the Wi-Fi network uses open, WEP or outdated security", "wifi_ap_changed": "the access point behind the same Wi-Fi name changed",
}
# plausible upper bounds per raw feature (used by the Forge to reject absurd values from a language model)
BOUNDS = {f: 5000.0 for f in FEATURES}
BOUNDS.update({"dns_entropy": 6.0, "net_bytes_out": 1e11, "net_bytes_in": 1e11, "net_pkts": 1e8, "gw_changed": 1.0, "wifi_weak": 1.0, "wifi_ap_changed": 1.0,
               "arp_dup": 50.0, "n_lan_new": 200.0})
LOGGED = {"net_bytes_out", "net_bytes_in", "net_pkts"}          # heavy-tailed: log-scaled; counts are log1p'd too
KIND_TO_FEATURES = {}


def shannon(names):
    s = "".join(names)
    if not s:
        return 0.0
    counts = np.bincount(np.frombuffer(s.encode("utf8", "ignore"), np.uint8))
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


class WindowBuilder:
    """Counts events of one window. Events are small dicts produced by the sensors."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.c = {f: 0.0 for f in FEATURES}
        self.remote_ips = set()
        self.dns_names = []

    def add(self, ev):
        k, c = ev["kind"], self.c
        if k == "process_new":
            c["n_proc_new"] += 1
            c["n_proc_user_dir"] += ev["exe_class"] in ("user", "temp", "downloads", "appdata")
            c["n_script_host"] += bool(ev.get("script_host"))
            c["n_obfuscated"] += bool(ev.get("obfuscated"))
            c["n_lolbin"] += bool(ev.get("lolbin"))
            c["n_office_child"] += bool(ev.get("office_parent"))
        elif k == "conn_new":
            c["n_conn_new"] += 1
            c["n_conn_novel"] += bool(ev.get("novel"))
            self.remote_ips.add(ev["remote_ip"])
            c["n_rare_port"] += bool(ev.get("rare_port"))
            c["n_public_conn"] += not ev.get("private", True)
        elif k == "listen_new":
            c["n_listen_new"] += 1
            c["n_listen_public"] += bool(ev.get("public_bind"))
        elif k == "file":
            c["n_file_events"] += 1
            c["n_file_created"] += ev["op"] == "created"
            c["n_exec_created"] += ev["op"] == "created" and bool(ev.get("is_exec"))
            c["n_startup_touch"] += ev["loc"] == "startup"
            c["n_file_renamed"] += ev["op"] == "moved"
            c["n_ext_change"] += bool(ev.get("ext_changed"))
            c["n_entropy_anomaly"] += bool(ev.get("entropy_anomaly"))
        elif k == "dns":
            c["dns_queries"] += 1
            c["dns_nx"] += bool(ev.get("nx"))
            self.dns_names.append(ev.get("name", ""))
        elif k == "lan":
            c["n_lan_new"] += ev.get("new_devices", 0)
            c["gw_changed"] = max(c["gw_changed"], float(bool(ev.get("gw_changed"))))
            c["arp_dup"] = max(c["arp_dup"], float(ev.get("arp_dup", 0)))
        elif k == "wifi":
            c["wifi_weak"] = max(c["wifi_weak"], float(bool(ev.get("weak"))))
            c["wifi_ap_changed"] = max(c["wifi_ap_changed"], float(bool(ev.get("ap_changed"))))
        elif k == "netstat":
            c["net_bytes_out"] += ev.get("bytes_out", 0)
            c["net_bytes_in"] += ev.get("bytes_in", 0)
            c["net_pkts"] += ev.get("pkts", 0)

    def close(self):
        self.c["n_remote_ips"] = float(len(self.remote_ips))
        self.c["dns_entropy"] = shannon(self.dns_names)
        raw = dict(self.c)
        self.reset()
        return raw


def to_vector(raw: dict) -> np.ndarray:
    """Raw counts -> the log-scaled vector the neurons use."""
    return np.array([math.log1p(max(raw.get(f, 0.0), 0.0)) if f != "dns_entropy" else raw.get(f, 0.0) / 4.0 for f in FEATURES], np.float64)
