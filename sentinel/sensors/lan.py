"""The local network and the Wi-Fi link: who else is on it, whether the router looks the same, how the Wi-Fi is secured.
Linux reads /proc; Windows uses the fixed read-only commands in probes.py (Windows path untested here, parsers tested on sample text)."""
import os
import sys

from . import Sensor
from .. import probes


def read_arp_linux(path="/proc/net/arp"):
    out = {}
    try:
        for line in open(path).read().splitlines()[1:]:
            f = line.split()
            if len(f) >= 4 and f[3] != "00:00:00:00:00:00" and f[2] != "0x0":
                out[f[0]] = f[3].lower()
    except OSError:
        pass
    return out


def read_gateway_linux(path="/proc/net/route"):
    try:
        for line in open(path).read().splitlines()[1:]:
            f = line.split()
            if len(f) > 3 and f[1] == "00000000" and int(f[3], 16) & 2:
                b = bytes.fromhex(f[2])[::-1]
                return ".".join(str(x) for x in b)
    except (OSError, ValueError):
        pass
    return ""


class LanSensor(Sensor):
    name = "lan_wifi"

    def __init__(self, arp_fn=None, gateway_fn=None, wifi_fn=None, first_is_baseline=True):
        win = sys.platform.startswith("win")
        self.arp_fn = arp_fn or ((lambda: probes.parse_arp_windows(probes.run_readonly("arp_table"))) if win else read_arp_linux)
        self.gateway_fn = gateway_fn or ((lambda: probes.parse_route_windows(probes.run_readonly("routes_v4"))) if win else read_gateway_linux)
        self.wifi_fn = wifi_fn or ((lambda: probes.parse_netsh_wlan(probes.run_readonly("wifi_interfaces"))) if win else (lambda: {}))
        self.macs = set()
        self.gw_mac = {}
        self.aps = {}                      # ssid -> set(bssid)
        self.weak_told = set()
        self.baseline_done = False

    def poll(self, now=None):
        arp, gw, events = self.arp_fn(), self.gateway_fn(), []
        macs = set(arp.values())
        new = macs - self.macs
        gw_mac = arp.get(gw, "")
        known_gw = self.gw_mac.get(gw)
        gw_changed = bool(known_gw and gw_mac and gw_mac != known_gw)
        by_mac = {}
        for ip, m in arp.items():
            by_mac.setdefault(m, []).append(ip)
        dup = [m for m, ips in by_mac.items() if len(ips) > 1 and (gw in ips or len(ips) > 2)]
        if self.baseline_done and (new or gw_changed or dup):
            events.append({"kind": "lan", "new_devices": len(new), "gw_changed": gw_changed, "arp_dup": len(dup), "gateway": gw,
                           "gw_mac": gw_mac, "old_gw_mac": known_gw or "", "dup_macs": dup[:3], "ts": now})
        if gw and gw_mac:
            self.gw_mac[gw] = self.gw_mac.get(gw) if (known_gw and not gw_changed) else gw_mac
            if gw_changed:
                self.gw_mac[gw] = gw_mac        # alert once, then accept as the new normal until a human says otherwise
        self.macs |= macs
        w = self.wifi_fn()
        if w:
            weak = probes.wifi_is_weak(w)
            seen = self.aps.setdefault(w["ssid"], set())
            ap_changed = bool(seen) and w["bssid"] not in seen
            told = w["ssid"] in self.weak_told
            if self.baseline_done and ((weak and not told) or ap_changed):
                events.append({"kind": "wifi", "ssid": w["ssid"], "weak": weak and not told, "ap_changed": ap_changed, "auth": w["auth"],
                               "cipher": w["cipher"], "bssid": w["bssid"], "signal": w.get("signal"), "ts": now})
            if weak:
                self.weak_told.add(w["ssid"])
            seen.add(w["bssid"])
        self.baseline_done = True
        return events

    def state(self):
        return {"macs": sorted(self.macs)[-1000:], "gw_mac": self.gw_mac, "aps": {k: sorted(v) for k, v in self.aps.items()},
                "weak_told": sorted(self.weak_told)}

    def load(self, st):
        self.macs, self.gw_mac = set(st.get("macs", [])), dict(st.get("gw_mac", {}))
        self.aps = {k: set(v) for k, v in st.get("aps", {}).items()}
        self.weak_told = set(st.get("weak_told", []))
        self.baseline_done = bool(self.macs or self.aps)
