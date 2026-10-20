"""The constitution: hard limits that live in CODE, outside the neurons.

Neurons decide how well the organism notices things. They never decide what it is ALLOWED to do. What it may do is fixed here and
enforced by construction (the modules that watch and think contain no code that deletes, moves, kills, or sends data off the machine),
by a static test that scans the source for forbidden calls, and, in deployment, by the operating system (a low-privilege account with
read-only access). If you change a limit, change it here, in review, not in a model.
"""
import os
import re

# the only things it can do to the world. 'request_action' records a request for a HUMAN; nothing in this package can carry one out.
ALLOWED_ACTIONS = ("alert", "report", "request_action")
REQUESTABLE_ACTIONS = ("quarantine_file", "block_connection", "end_process", "disable_startup_entry")

# What the watching and thinking code may never contain. Checked on the syntax tree by scan_file() (and by tests/test_sentinel.py), so a comment
# or a string cannot hide a call and a mention in a docstring cannot trigger a false alarm.
FORBIDDEN_IMPORTS = ("subprocess", "ctypes", "pickle", "marshal", "smtplib", "ftplib", "telnetlib", "requests", "httpx", "http.client", "urllib.request",
                     "urllib3", "socket", "winreg", "shutil", "anthropic")
FORBIDDEN_CALLS = ("os.remove", "os.unlink", "os.rmdir", "os.removedirs", "os.rename", "os.renames", "os.replace", "os.system", "os.popen", "os.kill",
                   "os.killpg", "os.truncate", "os.chmod", "os.chown", "eval", "exec", "compile", "__import__")
FORBIDDEN_ATTRS = ("kill", "terminate", "unlink", "rmdir", "rmtree", "send_signal", "truncate", "chmod")
# the few files that are allowed one forbidden import, and why
IMPORT_EXCEPTIONS = {
    "probes.py": {"subprocess"},          # fixed read-only inspection commands, nothing else (see probes.READONLY)
    "forge/llm.py": {"anthropic"},        # the Forge asks a model to DESIGN a spec; the running Sentinel never imports it
}
WRITE_MODES = set("wax+")


def _dotted(node):
    import ast
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def scan_file(path, rel=None):
    """-> (violations, write_opens). Violations are forbidden imports/calls; write_opens lists every open() that can write, as (line, mode)."""
    import ast
    rel = rel or os.path.basename(path)
    allowed = IMPORT_EXCEPTIONS.get(rel, set())
    tree = ast.parse(open(path, encoding="utf8").read())
    bad, writes = [], []
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.Import):
            mods = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
        for m in mods:
            for f in FORBIDDEN_IMPORTS:
                if (m == f or m.startswith(f + ".")) and f not in allowed:
                    bad.append(f"{rel}:{n.lineno} imports {m}")
        if isinstance(n, ast.Call):
            d = _dotted(n.func)
            if d in FORBIDDEN_CALLS:
                bad.append(f"{rel}:{n.lineno} calls {d}()")
            if isinstance(n.func, ast.Attribute) and n.func.attr in FORBIDDEN_ATTRS:
                bad.append(f"{rel}:{n.lineno} calls .{n.func.attr}()")
            if d == "open":
                mode = ""
                if len(n.args) > 1 and isinstance(n.args[1], ast.Constant):
                    mode = str(n.args[1].value)
                for k in n.keywords:
                    if k.arg == "mode" and isinstance(k.value, ast.Constant):
                        mode = str(k.value.value)
                if set(mode) & WRITE_MODES:
                    writes.append((n.lineno, mode))
    return bad, writes


# content of these is never read; their existence/changes are counted but names are not stored
SENSITIVE_PATTERNS = [re.compile(p, re.I) for p in (
    r"/\.ssh/", r"/\.aws/", r"/\.kube/", r"/\.gnupg/", r"/appdata/(local|roaming)/(google/chrome|microsoft/edge|mozilla|brave)",
    r"login data", r"/cookies", r"keychain", r"wallet", r"password", r"\.kdbx$", r"\.pfx$", r"\.pem$", r"id_rsa",
)]

SECRET_PATTERNS = [
    (re.compile(r"(?i)(password|passwd|pwd|secret|token|apikey|api_key|authorization)(=|:|\s+)(\S+)"), r"\1\2<redacted>"),
    (re.compile(r"(?i)(-p|--password)(\s+)(\S+)"), r"\1\2<redacted>"),
    (re.compile(r"[A-Za-z0-9+/]{60,}={0,2}"), "<long-blob>"),
]


def is_sensitive(path: str) -> bool:
    p = path.replace("\\", "/")
    return any(r.search(p) for r in SENSITIVE_PATTERNS)


def redact(text: str, limit: int = 300) -> str:
    """Remove secrets from anything before it is stored or shown."""
    for pat, rep in SECRET_PATTERNS:
        text = pat.sub(rep, text)
    return text[:limit]


def own_path(home: str, *parts: str) -> str:
    """The only place the organism may write: inside its own home directory."""
    root = os.path.realpath(home)
    p = os.path.realpath(os.path.join(root, *parts))
    if p != root and not p.startswith(root + os.sep):
        raise PermissionError(f"refusing to touch {p}: outside the organism's home {root}")
    return p


def check_action(action: str) -> str:
    if action not in ALLOWED_ACTIONS:
        raise PermissionError(f"action {action!r} is not allowed; allowed: {ALLOWED_ACTIONS}")
    return action


def check_request(kind: str) -> str:
    if kind not in REQUESTABLE_ACTIONS:
        raise PermissionError(f"{kind!r} cannot even be requested; requestable: {REQUESTABLE_ACTIONS}")
    return kind
