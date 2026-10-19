"""Fire drill for the Sentinel: makes HARMLESS activity that looks like three classic attacks, so you can see it notice them.

    python deploy/sentinel_drill.py --dir "C:\\Users\\YOU\\Downloads"        (a folder the Sentinel watches)

What it does, and only this:
  1. makes a new sub-folder `sentinel_drill_<time>` inside --dir, writes 80 small text files in it, fills them with random bytes and renames
     each to `.locked` (what file-encrypting malware does to its victims, without touching any of your files),
  2. starts one idle Python process for 20 seconds whose command line looks like an encoded PowerShell command (it is a comment, nothing runs),
  3. opens a listening port on 127.0.0.1 and connects to it once.
It deletes nothing. When you are done, delete the `sentinel_drill_*` folder yourself. This script is NOT part of the Sentinel (which can start
no programs and cannot write outside its own folder); it is a tool you run, like a smoke-alarm test button.
"""
import argparse
import os
import socket
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="a folder the Sentinel watches (e.g. your Downloads)")
    a = ap.parse_args()
    root = os.path.join(a.dir, time.strftime("sentinel_drill_%Y%m%d_%H%M%S"))
    os.makedirs(root)
    print("1/3 pretend file encryption in", root)
    for i in range(80):
        p = os.path.join(root, f"report_{i}.txt")
        open(p, "w").write("quarterly numbers " * 50)
    time.sleep(1)
    for i in range(80):
        p = os.path.join(root, f"report_{i}.txt")
        open(p, "wb").write(os.urandom(4096))
        os.rename(p, p + ".locked")
    print("2/3 idle process with an encoded-looking command line (20 s)")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)  # powershell -nop -w hidden -enc " + "QUJDREVGR0g=" * 12])
    print("3/3 local connection on 127.0.0.1")
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0)); srv.listen(1)
    cli = socket.create_connection(srv.getsockname())
    time.sleep(8)
    cli.close(); srv.close()
    print("Drill done. Within about a minute the Sentinel's dashboard should show: files being renamed in bulk, and an encoded command.")
    print(f"Delete the folder {root} yourself when you are finished. (The idle process ends on its own.)")
    proc.wait(timeout=30)


if __name__ == "__main__":
    main()
