"""The Sentinel's hands. It has none: the only thing it can do about a problem is write a REQUEST for a human, with a suggested command
shown as text. Nothing here runs the command. Suggestions are built only from validated numbers and addresses, never from file or program
names (an attacker controls those)."""
import re

from .constitution import check_request

IP4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def propose(alert):
    """alert -> (kind, target, suggestion) or None."""
    if alert["severity"] not in ("medium", "high"):
        return None
    t = alert.get("target") or {}
    kind = t.get("kind")
    if kind == "connection" and isinstance(t.get("ip"), str) and IP4.match(t["ip"]) and all(0 <= int(x) <= 255 for x in t["ip"].split(".")):
        ip = t["ip"]
        return (check_request("block_connection"), {"ip": ip},
                f"To block this address (Windows, as administrator):  netsh advfirewall firewall add rule name=\"block {ip}\" dir=out action=block remoteip={ip}")
    if kind == "process" and isinstance(t.get("pid"), int) and t["pid"] > 4:
        pid = t["pid"]
        return (check_request("end_process"), {"pid": pid},
                f"Check what it is first (Task Manager, Details tab, PID {pid}). To stop it:  taskkill /PID {pid}   (Linux/macOS: kill {pid}).")
    if kind == "file" and t.get("path"):
        startup = "start-up" in (alert.get("title") or "").lower()
        return (check_request("disable_startup_entry" if startup else "quarantine_file"), {"path": str(t["path"])[:260]},
                "Open the folder shown in the evidence, check the file, and move it to a folder you keep for isolated files. The Sentinel will not touch it.")
    return None
