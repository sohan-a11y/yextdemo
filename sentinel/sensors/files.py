"""Changes in watched folders (watchdog). Metadata for every change; for a few changed text-like files, the randomness of the first 4 KB
(encrypted data is random, ordinary text is not), never the content itself. Files that look sensitive are counted, not named or opened."""
import math
import os
import queue

from . import Sensor
from ..constitution import is_sensitive, redact
from ..paths import classify_path, ext_of, EXEC_EXT

LOW_ENTROPY_EXT = {".txt", ".csv", ".json", ".xml", ".html", ".htm", ".md", ".log", ".ini", ".cfg", ".py", ".js", ".c", ".cpp", ".java", ".sql", ".yaml", ".yml", ".rtf", ".tex"}
MAX_READS = 20
MAX_BYTES = 50_000_000


def byte_entropy(b: bytes) -> float:
    if not b:
        return 0.0
    counts = [0] * 256
    for x in b:
        counts[x] += 1
    n = len(b)
    return -sum(c / n * math.log2(c / n) for c in counts if c)


def default_watch_dirs():
    home = os.path.expanduser("~")
    cands = [os.path.join(home, d) for d in ("Downloads", "Desktop", "Documents")]
    if os.name == "nt":
        ad = os.environ.get("APPDATA", os.path.join(home, "AppData", "Roaming"))
        cands += [os.path.join(os.environ.get("TEMP", os.path.join(home, "AppData", "Local", "Temp"))),
                  os.path.join(ad, "Microsoft", "Windows", "Start Menu", "Programs", "Startup")]
    else:
        cands += ["/tmp", os.path.join(home, ".config", "autostart")]
    return [d for d in cands if os.path.isdir(d)]


class FileSensor(Sensor):
    name = "files"

    def __init__(self, dirs=None, recursive=True, polling=False, max_queue=20000):
        self.dirs = list(dirs) if dirs is not None else default_watch_dirs()
        self.recursive = recursive
        self.q = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        self.observer = None
        try:
            from watchdog.events import FileSystemEventHandler
            if polling:
                from watchdog.observers.polling import PollingObserver as Obs
            else:
                from watchdog.observers import Observer as Obs
        except ImportError as ex:
            self.available, self.reason = False, f"watchdog not installed: {ex}"
            return
        sensor = self

        class H(FileSystemEventHandler):
            def on_any_event(self, e):
                if e.is_directory or e.event_type not in ("created", "modified", "moved", "deleted"):
                    return
                try:
                    sensor.q.put_nowait((e.event_type, e.src_path, getattr(e, "dest_path", "") or ""))
                except queue.Full:
                    sensor.dropped += 1

        self.observer = Obs()
        for d in self.dirs:
            try:
                self.observer.schedule(H(), d, recursive=recursive)
            except OSError as ex:
                self.reason += f"cannot watch {d}: {ex}; "
        self.observer.start()

    def _entropy_anomaly(self, path, ext):
        """`ext` is the type the file had BEFORE (a text file that is renamed and turns random is the ransomware pattern)."""
        if ext not in LOW_ENTROPY_EXT or is_sensitive(path):
            return False
        try:
            if os.path.getsize(path) > MAX_BYTES:
                return False
            with open(path, "rb") as f:
                head = f.read(4096)
        except OSError:
            return False
        return len(head) >= 512 and byte_entropy(head) > 7.2

    def poll(self, now=None):
        seen, events, reads = set(), [], 0
        while True:
            try:
                op, src, dst = self.q.get_nowait()
            except queue.Empty:
                break
            path = dst if op == "moved" and dst else src
            if (op, path) in seen or (op == "modified" and ("created", path) in seen):
                continue
            seen.add((op, path))
            sens = is_sensitive(path) or is_sensitive(src)
            ext = ext_of(path)
            ev = {"kind": "file", "op": op, "loc": classify_path(path), "is_exec": ext in EXEC_EXT, "sensitive": sens, "ts": now,
                  "ext_changed": op == "moved" and ext_of(src) != ext_of(dst), "path": "<sensitive>" if sens else redact(path, 260),
                  "entropy_anomaly": False}
            if op in ("created", "modified", "moved") and reads < MAX_READS and not sens:
                reads += 1
                ev["entropy_anomaly"] = self._entropy_anomaly(path, ext_of(src))
            events.append(ev)
        return events

    def close(self):
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=3)

    def info(self):
        d = super().info()
        d.update(dirs=self.dirs, dropped=self.dropped)
        return d
