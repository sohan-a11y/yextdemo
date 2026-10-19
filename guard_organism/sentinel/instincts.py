"""Instincts: rules written by people, the part of the Sentinel that knows named bad habits from day one.

They fire on single events and on whole windows. Each alert carries a plain-language reason and a key used for de-duplication and for
"this was fine" feedback. An instinct can only produce an alert; nothing here acts.
"""
import time
from collections import OrderedDict

from .features import shannon
from .paths import norm

SEV = {"info": 0, "low": 1, "medium": 2, "high": 3}


def alert(rule, severity, title, why, key, evidence=None, target=None):
    return {"rule": rule, "severity": severity, "title": title, "why": why, "key": f"{rule}:{key}", "evidence": evidence or [], "target": target or {}}


class Memory:
    """Small time-limited memory so separate events can be joined (a program was created, then started, then went online)."""

    def __init__(self, ttl=600, cap=2000):
        self.ttl, self.cap, self.d = ttl, cap, OrderedDict()

    def put(self, k, v, now):
        self.d[k] = (now, v)
        self.d.move_to_end(k)
        while len(self.d) > self.cap:
            self.d.popitem(last=False)

    def get(self, k, now):
        r = self.d.get(k)
        return r[1] if r and now - r[0] <= self.ttl else None


class Instincts:
    def __init__(self, cfg=None):
        c = {"mass_files": 60, "mass_ext": 15, "mass_entropy": 5, "dga_min": 6, "dga_entropy": 3.8, "dga_nx": 0.4}
        c.update(cfg or {})
        self.cfg = c
        self.procs = Memory(600)             # pid -> process event
        self.dropped = Memory(1800)          # lowercase path -> file event (programs written lately)
        self.suppressed = {}                 # key -> until_ts

    def suppress(self, key, until):
        self.suppressed[key] = until

    # ------------------------------------------------------------------ per event
    def observe(self, ev, now=None):
        now = now or ev.get("ts") or time.time()
        k = ev["kind"]
        out = []
        if k == "process_new":
            self.procs.put(ev["pid"], ev, now)
            where = f"{ev['name']} ({ev['exe_class']} folder)"
            if ev["script_host"] and ev["obfuscated"]:
                out.append(alert("script_obfuscated", "high", f"{ev['name']} started with an encoded or hidden command",
                                 f"A script interpreter was started with a command line that looks encoded, hidden or downloads code. Started by {ev['parent'] or 'unknown'}.",
                                 ev["name"], [ev["cmd"]], {"kind": "process", "pid": ev["pid"]}))
            if ev["office_parent"] and (ev["script_host"] or ev["lolbin"]):
                out.append(alert("office_spawns_tool", "high", f"{ev['parent']} started {ev['name']}",
                                 "Office and PDF programs normally do not start command shells or system tools; this is how malicious documents work.",
                                 f"{ev['parent']}>{ev['name']}", [ev["cmd"]], {"kind": "process", "pid": ev["pid"]}))
            p = norm(ev["exe"])
            if p and self.dropped.get(p, now) is not None:
                out.append(alert("fresh_download_started", "medium", f"{ev['name']} was started minutes after being written to disk",
                                 f"The program file {ev['exe']} was created recently in {ev['exe_class']} and has just been run.", ev["name"],
                                 [ev["exe"]], {"kind": "process", "pid": ev["pid"]}))
            elif ev["exe_class"] in ("temp", "downloads", "appdata") and ev.get("novel_program"):
                out.append(alert("new_program_user_dir", "low", f"First run of a program from your {ev['exe_class']} folder: {ev['name']}",
                                 "Programs that run straight from temporary or download folders are common for malware, but also for installers.",
                                 ev["name"], [ev["exe"]], {"kind": "process", "pid": ev["pid"]}))
        elif k == "conn_new":
            pe = self.procs.get(ev["pid"], now)
            if not ev["private"]:
                if pe and pe["lolbin"]:
                    out.append(alert("lolbin_network", "high", f"{pe['name']} is talking to the internet ({ev['remote_ip']})",
                                     "This built-in system tool is often used to download malware; it rarely needs the internet.", pe["name"],
                                     [f"{ev['remote_ip']}:{ev['remote_port']}"], {"kind": "connection", "pid": ev["pid"], "ip": ev["remote_ip"]}))
                if pe and pe["exe_class"] in ("temp", "downloads") and ev["novel"]:
                    out.append(alert("temp_program_online", "high", f"{pe['name']}, running from {pe['exe_class']}, contacted a new address {ev['remote_ip']}",
                                     "A program living in a temporary/download folder just reached out to an address it has never contacted.", pe["name"],
                                     [pe["exe"], f"{ev['remote_ip']}:{ev['remote_port']}"], {"kind": "connection", "pid": ev["pid"], "ip": ev["remote_ip"]}))
        elif k == "listen_new":
            if ev["public_bind"] and ev["exe_class"] not in ("system", "programs"):
                out.append(alert("public_listener", "medium", f"{ev['prog']} opened port {ev['port']} to the whole network",
                                 "A program from a user folder is accepting connections from other machines.", f"{ev['prog']}:{ev['port']}",
                                 [], {"kind": "process", "pid": ev["pid"]}))
        elif k == "file":
            p = norm(ev["path"])
            if ev["is_exec"] and ev["op"] in ("created", "moved") and ev["loc"] in ("temp", "downloads", "appdata", "startup", "user"):
                self.dropped.put(p, ev, now)
            if ev["loc"] == "startup" and ev["op"] in ("created", "modified", "moved"):
                out.append(alert("startup_change", "high" if ev["is_exec"] else "medium", "Something changed in a start-up location",
                                 "Files in start-up folders run every time you sign in; malware uses them to survive restarts.",
                                 p.rsplit("/", 1)[-1], [ev["path"]], {"kind": "file", "path": ev["path"]}))
        elif k == "lan":
            if ev.get("gw_changed"):
                out.append(alert("gateway_changed", "high", "Your router's hardware address changed",
                                 f"The gateway {ev['gateway']} used to answer as {ev['old_gw_mac']} and now answers as {ev['gw_mac']}. "
                                 "That happens when the router is replaced, and when someone on the network is impersonating it.", ev["gateway"],
                                 [ev["old_gw_mac"], ev["gw_mac"]], {"kind": "device", "mac": ev["gw_mac"]}))
            if ev.get("arp_dup"):
                out.append(alert("arp_duplicate", "high", "One device is answering for several network addresses",
                                 "A single hardware address claims the router's address or several others, a sign of address spoofing.",
                                 ",".join(ev.get("dup_macs", [])), list(ev.get("dup_macs", [])), {"kind": "device", "mac": (ev.get("dup_macs") or [""])[0]}))
            if ev.get("new_devices"):
                out.append(alert("new_lan_device", "info", f"{ev['new_devices']} new device(s) appeared on your network",
                                 "A phone, TV or guest may have joined. If you do not recognise it, check your router's device list.", "lan"))
        elif k == "wifi":
            if ev.get("weak"):
                out.append(alert("wifi_weak", "low", f"Wi-Fi '{ev['ssid']}' uses weak security ({ev['auth']}/{ev['cipher']})",
                                 "Open, WEP or old WPA networks let nearby people read or tamper with traffic. Prefer WPA2/WPA3.", ev["ssid"]))
            if ev.get("ap_changed"):
                out.append(alert("wifi_ap_changed", "medium", f"A different access point is serving '{ev['ssid']}'",
                                 f"The Wi-Fi name is the same but the access point address is new ({ev['bssid']}). Normal with mesh/roaming; "
                                 "also how a fake hotspot imitating your network behaves.", ev["ssid"], [ev["bssid"]]))
        return out

    # ------------------------------------------------------------------ per window
    def end_window(self, raw, events, now=None):
        c, out = self.cfg, []
        if raw["n_file_events"] >= c["mass_files"] and (raw["n_ext_change"] >= c["mass_ext"] or raw["n_entropy_anomaly"] >= c["mass_entropy"]):
            out.append(alert("ransomware_pattern", "high", "Many files are being renamed and rewritten at once",
                             f"{raw['n_file_events']:.0f} file changes, {raw['n_ext_change']:.0f} with a new extension, {raw['n_entropy_anomaly']:.0f} text-like files that now "
                             "look random. This is what file-encrypting malware does.", "mass"))
        elif raw["n_file_events"] >= 4 * c["mass_files"]:
            out.append(alert("mass_file_change", "medium", f"{raw['n_file_events']:.0f} file changes in one window",
                             "Large bursts happen during installs and syncs, and also during encryption or wiping.", "mass"))
        names = [e["name"] for e in events if e["kind"] == "dns"]
        if len(names) >= c["dga_min"]:
            labels = [n.split(".")[-2] if n.count(".") >= 1 else n for n in names]
            long_ = [l for l in labels if len(l) >= 10]
            nx = raw["dns_nx"] / max(raw["dns_queries"], 1)
            if len(long_) >= c["dga_min"] and shannon(long_) / 1.0 > c["dga_entropy"] and nx >= c["dga_nx"]:
                out.append(alert("dns_generated_names", "medium", "Many look-ups of random-looking names that mostly fail",
                                 "Malware that cannot reach its controller tries long lists of machine-generated names.", "dga", names[:5]))
        return out
