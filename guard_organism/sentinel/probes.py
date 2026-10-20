"""The ONE place in the Sentinel that may start an operating-system command.

Only fixed, read-only inspection commands from READONLY can be run: no shell, no arguments from outside, no output kept beyond what the
parsers extract, a short timeout. Everything else in the package is forbidden to start programs (tests/test_sentinel.py scans for it).
The parsers are pure functions, so they are tested against sample output; the commands themselves are Windows-only and untested here.
"""
import re
import subprocess

READONLY = {
    "wifi_interfaces": ("netsh", "wlan", "show", "interfaces"),
    "arp_table": ("arp", "-a"),
    "routes_v4": ("route", "print", "-4"),
    "dns_cache": ("ipconfig", "/displaydns"),
}
MAX_OUT = 400_000


def run_readonly(key: str, timeout: float = 8.0) -> str:
    argv = READONLY[key]                       # KeyError for anything not in the table: nothing else can be run
    try:
        r = subprocess.run(list(argv), shell=False, capture_output=True, timeout=timeout, text=True, errors="replace")
    except (OSError, subprocess.SubprocessError):
        return ""
    return (r.stdout or "")[:MAX_OUT]


# ------------------------------------------------------------------------------------------------ parsers
def parse_netsh_wlan(text: str) -> dict:
    """First connected interface from `netsh wlan show interfaces`. Empty dict when not on Wi-Fi."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Za-z][A-Za-z0-9 \-/]*?)\s*:\s*(.*?)\s*$", line)
        if not m:
            continue
        k, v = m.group(1).strip().lower(), m.group(2)
        if k in out:                                    # second interface block: keep the first
            if k == "name":
                break
            continue
        out[k] = v
    if out.get("state", "").lower() not in ("connected",) or not out.get("ssid"):
        return {}
    sig = re.sub(r"[^0-9]", "", out.get("signal", ""))
    return {"ssid": out["ssid"], "bssid": out.get("bssid", "").lower(), "auth": out.get("authentication", ""), "cipher": out.get("cipher", ""),
            "radio": out.get("radio type", ""), "signal": int(sig) if sig else None}


def wifi_is_weak(info: dict) -> bool:
    a, c = info.get("auth", "").lower(), info.get("cipher", "").lower()
    return bool(info) and (a.startswith("open") or "wep" in a or a in ("wpa-personal", "wpa-enterprise", "wpa") or c in ("tkip", "wep", "none"))


def norm_mac(m: str) -> str:
    return m.lower().replace("-", ":")


def parse_arp_windows(text: str) -> dict:
    """`arp -a` -> {ip: mac} for dynamic/static entries (multicast and broadcast skipped)."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+\.\d+\.\d+\.\d+)\s+([0-9a-fA-F]{2}(?:[-:][0-9a-fA-F]{2}){5})\s+(\w+)", line)
        if m:
            mac = norm_mac(m.group(2))
            if not (mac.startswith("ff:ff") or mac.startswith("01:00:5e") or mac.startswith("33:33")):
                out[m.group(1)] = mac
    return out


def parse_route_windows(text: str) -> str:
    """Default gateway from `route print -4` (lowest metric)."""
    best = None
    for line in text.splitlines():
        m = re.match(r"^\s*0\.0\.0\.0\s+0\.0\.0\.0\s+(\d+\.\d+\.\d+\.\d+)\s+\d+\.\d+\.\d+\.\d+\s+(\d+)", line)
        if m and (best is None or int(m.group(2)) < best[1]):
            best = (m.group(1), int(m.group(2)))
    return best[0] if best else ""


def parse_displaydns(text: str) -> list:
    """Names in the Windows DNS client cache, from `ipconfig /displaydns`. Each item: (name, failed)."""
    out, cur = [], None
    for line in text.splitlines():
        m = re.match(r"^\s*Record Name[ .]*:\s*(\S+)", line)
        if m:
            cur = m.group(1).lower()
            out.append([cur, False])
        elif cur and re.search(r"Name does not exist", line, re.I):
            out[-1][1] = True
    return [(n, f) for n, f in out]
