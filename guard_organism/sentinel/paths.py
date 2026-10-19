"""Where a program or file lives says a lot about it. Works for Windows and POSIX paths."""
import os
import re

SCRIPT_HOSTS = {"powershell", "powershell_ise", "pwsh", "cmd", "wscript", "cscript", "mshta", "bash", "sh", "python", "python3", "node"}
LOLBINS = {"certutil", "bitsadmin", "mshta", "regsvr32", "rundll32", "wmic", "msiexec", "installutil", "regasm", "regsvcs", "curl", "wget"}
OFFICE_PARENTS = {"winword", "excel", "powerpnt", "outlook", "onenote", "acrord32", "acrobat"}
OBFUSCATION = re.compile(r"(?i)(-enc(odedcommand)?\b|frombase64string|\biex\b|invoke-expression|downloadstring|-w(indowstyle)?\s+hidden|-nop\b|-noni\b|"
                         r"bypass|\bcertutil\b.*-urlcache|[A-Za-z0-9+/]{80,}={0,2})")
EXEC_EXT = {".exe", ".dll", ".scr", ".sys", ".msi", ".com", ".bat", ".cmd", ".ps1", ".vbs", ".js", ".jse", ".wsf", ".hta", ".lnk", ".jar", ".so", ".sh", ".elf"}
DOC_EXT = {".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pdf", ".txt", ".csv", ".jpg", ".jpeg", ".png", ".mp3", ".mp4", ".zip", ".rar", ".7z"}
COMMON_PORTS = {80, 443, 53, 123, 22, 21, 25, 110, 143, 993, 995, 587, 465, 8080, 8443, 3389, 445, 139, 5353, 1900, 137, 138}

CLASSES = ("system", "programs", "user", "temp", "downloads", "appdata", "startup", "other")


def norm(path: str) -> str:
    return (path or "").replace("\\", "/").lower()


def classify_path(path: str) -> str:
    p = norm(path)
    if not p:
        return "other"
    if "/startup/" in p or "/start menu/programs/startup" in p or "/etc/cron" in p or "/.config/autostart" in p:
        return "startup"
    if "/temp/" in p or p.startswith("/tmp/") or p.startswith("/var/tmp/") or p.startswith("/dev/shm/") or "/appdata/local/temp" in p:
        return "temp"
    if "/downloads/" in p:
        return "downloads"
    if "/appdata/" in p or "/.local/share/" in p or "/.cache/" in p:
        return "appdata"
    if re.match(r"^[a-z]:/windows(/|$)", p) or p.startswith(("/usr/", "/bin/", "/sbin/", "/lib", "/system/", "/snap/")):
        return "system"
    if "/program files" in p or p.startswith(("/opt/", "/applications/")):
        return "programs"
    if re.match(r"^[a-z]:/users/", p) or p.startswith(("/home/", "/users/", "/root/")):
        return "user"
    return "other"


def base_name(path_or_name: str) -> str:
    n = norm(path_or_name).rsplit("/", 1)[-1]
    n = re.sub(r"\.(exe|com|bat|cmd)$", "", n)
    m = re.match(r"^(python|pwsh|node|perl|ruby|php)[\d.]+$", n)
    return m.group(1) if m else n


def ext_of(path: str) -> str:
    return os.path.splitext(norm(path))[1]


def is_private_ip(ip: str) -> bool:
    if not ip:
        return True
    if ":" in ip:
        return ip.startswith(("fe80", "fc", "fd", "::1")) or ip == "::"
    parts = ip.split(".")
    try:
        a, b = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return True
    return a == 10 or a == 127 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168) or (a == 169 and b == 254) or a == 0
