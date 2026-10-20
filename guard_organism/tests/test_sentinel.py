"""Tests of the Sentinel (read-only host watcher) and the Neuron Forge.

Sensors are tested against the real operating system of the test machine (Linux): real processes, real files, real sockets.
Windows-only code paths (netsh/arp/route/ipconfig) are tested through their PARSERS on sample text, never through the commands, and are
marked as such. Nothing here needs the network or an API key: the Claude teacher is tested against a local fake server.
"""
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from tests import _common  # noqa
from sentinel import actuators, constitution as C, features as F, paths as PA, probes
from sentinel.brain import Brain
from sentinel.forge import spec as S
from sentinel.forge.demo import synthetic_baseline
from sentinel.forge.generate import apply_effects, synth, weaken
from sentinel.forge.llm import ClaudeTeacher, MockTeacher, TeacherError, TeacherRefused, LIBRARY
from sentinel.forge.train import forge as forge_neuron
from sentinel.forge import __main__ as FM
from sentinel.instincts import Instincts
from sentinel.neurons import KitEnsemble, TinyMLP, ForgeNeuron
from sentinel.sensors import Sensor
from sentinel.sensors.connections import ConnectionSensor
from sentinel.sensors.files import FileSensor, byte_entropy
from sentinel.sensors.lan import LanSensor
from sentinel.sensors.netflow import NetflowSensor, parse_dns_packet
from sentinel.sensors.processes import ProcessSensor, describe
from sentinel.service import Sentinel, acquire_lock
from sentinel.store import SentinelStore

ROOT = os.path.join(os.path.dirname(__file__), "..", "sentinel")


def tmp():
    d = tempfile.mkdtemp(prefix="sentinel_test_")
    return d


class Fake(Sensor):
    name = "fake"

    def __init__(self):
        self.q = []

    def poll(self, now=None):
        q, self.q = self.q, []
        return q

    def state(self):
        return {"x": 1}


def proc_ev(**kw):
    ev = {"kind": "process_new", "pid": 4242, "ppid": 1, "name": "python", "parent": "explorer", "exe_class": "user", "exe": "C:/Users/a/x.exe", "cmd": "x",
          "script_host": False, "lolbin": False, "obfuscated": False, "office_parent": False, "novel_program": False, "ts": 0}
    ev.update(kw)
    return ev


def make_service(home, sensors=None, clock=None, cfg=None):
    T = clock or [1.8e9]
    c = {"window_s": 30, "poll_s": 5, "port": 0, "reflex": {"fm_windows": 5, "ad_windows": 5, "cal_windows": 5}, "cortex": {"enabled": False}}
    c.update(cfg or {})
    return Sentinel(home, sensors if sensors is not None else [Fake()], c, clock=lambda: T[0], log=lambda *a: None).boot(), T


def run_windows(s, T, n, every=30):
    out = []
    for _ in range(n):
        T[0] += every
        r = s.step()
        if r.get("alerts") is not None:
            out.append(r)
    return out


# ============================================================================================ the constitution
class TestConstitution(unittest.TestCase):
    def test_static_scan_finds_nothing_forbidden(self):
        bad, writes = [], {}
        for f in sorted(glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True)):
            rel = os.path.relpath(f, ROOT).replace(os.sep, "/")
            b, w = C.scan_file(f, rel)
            bad += b
            if w:
                writes[rel] = [m for _, m in w]
        self.assertEqual(bad, [], "forbidden call or import in the Sentinel")
        # the only places that can open a file for writing, each of which goes through own_path()
        self.assertEqual(set(writes), {"__main__.py", "service.py", "forge/__main__.py"})

    def test_scanner_catches_what_it_should(self):
        d = tmp()
        p = os.path.join(d, "evil.py")
        open(p, "w").write("import os, shutil, subprocess\nos.remove('x')\nshutil.rmtree('y')\nproc.kill()\nfrom pathlib import Path\nPath('z').unlink()\neval('1')\nopen('f','w')\n")
        bad, writes = C.scan_file(p, "evil.py")
        text = " ".join(bad)
        for needle in ("subprocess", "shutil", "os.remove", ".kill()", ".unlink()", "eval", "rmtree"):
            self.assertIn(needle, text)
        self.assertEqual(len(writes), 1)
        # a mention in a string or comment is not a call
        q = os.path.join(d, "fine.py")
        open(q, "w").write('"""never call os.remove or pickle"""\nx = "subprocess.run"  # os.system\n')
        self.assertEqual(C.scan_file(q, "fine.py")[0], [])
        shutil.rmtree(d)

    def test_only_probes_may_start_programs_only_forge_llm_may_reach_the_internet(self):
        for f in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
            src = open(f).read()
            rel = os.path.relpath(f, ROOT).replace(os.sep, "/")
            if rel != "probes.py":
                self.assertNotIn("import subprocess", src, rel)
            if rel != "forge/llm.py":
                self.assertNotIn("import anthropic", src, rel)
        self.assertEqual(set(probes.READONLY), {"wifi_interfaces", "arp_table", "routes_v4", "dns_cache"})
        with self.assertRaises(KeyError):
            probes.run_readonly("rm -rf /")
        for argv in probes.READONLY.values():
            self.assertIsInstance(argv, tuple)

    def test_own_path_refuses_to_leave_home(self):
        d = tmp()
        self.assertTrue(C.own_path(d, "a", "b").startswith(os.path.realpath(d)))
        for bad in ("..", "../x", "/etc/passwd"):
            with self.assertRaises(PermissionError):
                C.own_path(d, bad)
        os.symlink("/etc", os.path.join(d, "link"))
        with self.assertRaises(PermissionError):
            C.own_path(d, "link", "passwd")
        shutil.rmtree(d)

    def test_actions_are_limited(self):
        for a in ("alert", "report", "request_action"):
            C.check_action(a)
        for a in ("delete", "kill", "quarantine_file", "exec"):
            with self.assertRaises(PermissionError):
                C.check_action(a)
        with self.assertRaises(PermissionError):
            C.check_request("delete_file")

    def test_redaction_and_sensitive_paths(self):
        r = C.redact("run.exe --password hunter2 token=abc123 " + "A" * 90)
        self.assertNotIn("hunter2", r)
        self.assertNotIn("abc123", r)
        self.assertIn("<long-blob>", r)
        self.assertTrue(C.is_sensitive("C:\\Users\\a\\.ssh\\id_rsa"))
        self.assertTrue(C.is_sensitive("/home/a/passwords.kdbx"))
        self.assertFalse(C.is_sensitive("/home/a/notes.txt"))


# ============================================================================================ paths, features, probes
class TestPathsFeaturesProbes(unittest.TestCase):
    def test_path_classes_windows_and_posix(self):
        c = PA.classify_path
        self.assertEqual(c("C:\\Users\\a\\AppData\\Local\\Temp\\x.exe"), "temp")
        self.assertEqual(c("C:\\Users\\a\\Downloads\\setup.exe"), "downloads")
        self.assertEqual(c("C:\\Windows\\System32\\cmd.exe"), "system")
        self.assertEqual(c("C:\\Program Files\\App\\a.exe"), "programs")
        self.assertEqual(c("C:\\Users\\a\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Startup\\a.lnk"), "startup")
        self.assertEqual(c("/tmp/x"), "temp")
        self.assertEqual(c("/usr/bin/python3"), "system")
        self.assertEqual(PA.base_name("C:\\Windows\\System32\\PowerShell.exe"), "powershell")
        self.assertEqual(PA.base_name("/usr/bin/python3.13"), "python")
        self.assertTrue(PA.is_private_ip("192.168.1.4") and PA.is_private_ip("10.1.2.3") and PA.is_private_ip("172.20.0.1"))
        self.assertFalse(PA.is_private_ip("8.8.8.8") or PA.is_private_ip("172.32.0.1"))

    def test_window_builder_counts(self):
        b = F.WindowBuilder()
        b.add(proc_ev(exe_class="temp", script_host=True, obfuscated=True))
        b.add({"kind": "conn_new", "remote_ip": "8.8.8.8", "novel": True, "rare_port": True, "private": False})
        b.add({"kind": "conn_new", "remote_ip": "8.8.8.8", "novel": False, "rare_port": False, "private": False})
        b.add({"kind": "file", "op": "moved", "loc": "startup", "is_exec": True, "ext_changed": True, "entropy_anomaly": True})
        b.add({"kind": "dns", "name": "abc.example.com", "nx": True})
        b.add({"kind": "lan", "new_devices": 2, "gw_changed": True, "arp_dup": 1})
        b.add({"kind": "wifi", "weak": True, "ap_changed": False})
        b.add({"kind": "netstat", "bytes_out": 100, "bytes_in": 50, "pkts": 3})
        r = b.close()
        self.assertEqual((r["n_proc_new"], r["n_proc_user_dir"], r["n_script_host"], r["n_obfuscated"]), (1, 1, 1, 1))
        self.assertEqual((r["n_conn_new"], r["n_conn_novel"], r["n_remote_ips"], r["n_rare_port"], r["n_public_conn"]), (2, 1, 1, 1, 2))
        self.assertEqual((r["n_startup_touch"], r["n_ext_change"], r["n_entropy_anomaly"], r["n_file_renamed"]), (1, 1, 1, 1))
        self.assertEqual((r["dns_queries"], r["dns_nx"], r["n_lan_new"], r["gw_changed"], r["arp_dup"], r["wifi_weak"], r["net_bytes_out"]), (1, 1, 2, 1, 1, 1, 100))
        self.assertEqual(sum(b.close().values()), 0)                      # reset
        v = F.to_vector(r)
        self.assertEqual(v.shape, (len(F.FEATURES),))
        self.assertTrue(np.all(np.isfinite(v)) and np.all(v >= 0))

    def test_every_feature_has_description_and_bound(self):
        self.assertEqual(set(F.FEATURES), set(F.DESCRIPTIONS))
        self.assertEqual(set(F.FEATURES), set(F.BOUNDS))
        self.assertEqual(len(set(F.FEATURES)), len(F.FEATURES))

    def test_dns_entropy(self):
        self.assertGreater(F.shannon(["xk3jf9alqz0d", "q8w7e6r5t4y3"]), F.shannon(["google", "google"]))

    # --- parsers on SAMPLE text (the commands themselves only exist on Windows and are not run here)
    NETSH = """
There is 1 interface on the system:

    Name                   : Wi-Fi
    Description            : Intel(R) Wi-Fi 6 AX201
    State                  : connected
    SSID                   : HomeNet
    BSSID                  : AA:BB:CC:DD:EE:FF
    Network type           : Infrastructure
    Radio type             : 802.11ac
    Authentication         : WPA2-Personal
    Cipher                 : CCMP
    Signal                 : 94%
"""
    ARP = """
Interface: 192.168.1.5 --- 0x4
  Internet Address      Physical Address      Type
  192.168.1.1           aa-bb-cc-dd-ee-ff     dynamic
  192.168.1.20          11-22-33-44-55-66     dynamic
  192.168.1.255         ff-ff-ff-ff-ff-ff     static
  224.0.0.22            01-00-5e-00-00-16     static
"""
    ROUTE = """
IPv4 Route Table
Active Routes:
Network Destination        Netmask          Gateway       Interface  Metric
          0.0.0.0          0.0.0.0      192.168.1.1     192.168.1.5     25
          0.0.0.0          0.0.0.0      10.0.0.1        10.0.0.9       300
        127.0.0.0        255.0.0.0         On-link         127.0.0.1    331
"""
    DNS = """
    google.com
    ----------------------------------------
    Record Name . . . . . : google.com
    Record Type . . . . . : 1
    Record Name . . . . . : xk3jf9alqzd0.example.net
    Name does not exist.
"""

    def test_parse_netsh(self):
        w = probes.parse_netsh_wlan(self.NETSH)
        self.assertEqual((w["ssid"], w["bssid"], w["auth"], w["cipher"], w["signal"]), ("HomeNet", "aa:bb:cc:dd:ee:ff", "WPA2-Personal", "CCMP", 94))
        self.assertFalse(probes.wifi_is_weak(w))
        self.assertTrue(probes.wifi_is_weak({"auth": "Open", "cipher": "None"}))
        self.assertTrue(probes.wifi_is_weak({"auth": "WEP", "cipher": "WEP"}))
        self.assertTrue(probes.wifi_is_weak({"auth": "WPA-Personal", "cipher": "TKIP"}))
        self.assertEqual(probes.parse_netsh_wlan(self.NETSH.replace("connected", "disconnected")), {})
        self.assertEqual(probes.parse_netsh_wlan(""), {})

    def test_parse_arp_route_dns(self):
        self.assertEqual(probes.parse_arp_windows(self.ARP), {"192.168.1.1": "aa:bb:cc:dd:ee:ff", "192.168.1.20": "11:22:33:44:55:66"})
        self.assertEqual(probes.parse_route_windows(self.ROUTE), "192.168.1.1")
        self.assertEqual(probes.parse_displaydns(self.DNS), [("google.com", False), ("xk3jf9alqzd0.example.net", True)])
        self.assertEqual(probes.parse_arp_windows("garbage"), {})
        self.assertEqual(probes.parse_route_windows(""), "")


# ============================================================================================ sensors on the real OS
class TestSensors(unittest.TestCase):
    def test_file_sensor_sees_create_rename_and_random_content(self):
        d = tmp()
        fs = FileSensor([d])
        self.addCleanup(fs.close)
        time.sleep(0.4)
        open(os.path.join(d, "a.txt"), "wb").write(os.urandom(3000))
        os.rename(os.path.join(d, "a.txt"), os.path.join(d, "a.txt.locked"))
        open(os.path.join(d, "setup.exe"), "w").write("MZ")
        open(os.path.join(d, "notes.txt"), "w").write("hello world. " * 200)
        time.sleep(1.0)
        ev = fs.poll(5)
        ops = {(os.path.basename(e["path"]), e["op"]) for e in ev}
        self.assertIn(("a.txt.locked", "moved"), ops)
        self.assertIn(("setup.exe", "created"), ops)
        moved = next(e for e in ev if e["op"] == "moved")
        self.assertTrue(moved["ext_changed"])
        self.assertTrue(moved["entropy_anomaly"])                          # a text file that turned random
        self.assertTrue(next(e for e in ev if e["path"].endswith("setup.exe") and e["op"] == "created")["is_exec"])
        calm = [e for e in ev if e["path"].endswith("notes.txt")]
        self.assertTrue(calm and not any(e["entropy_anomaly"] for e in calm))
        self.assertEqual(fs.poll(6), [])
        shutil.rmtree(d)

    def test_file_sensor_never_names_or_reads_sensitive_files(self):
        d = tmp()
        os.makedirs(os.path.join(d, ".ssh"))
        fs = FileSensor([d])
        self.addCleanup(fs.close)
        time.sleep(0.4)
        open(os.path.join(d, ".ssh", "id_rsa"), "wb").write(os.urandom(3000))
        time.sleep(0.8)
        ev = [e for e in fs.poll(1) if e["sensitive"]]
        self.assertTrue(ev)
        for e in ev:
            self.assertEqual(e["path"], "<sensitive>")
            self.assertFalse(e["entropy_anomaly"])
        shutil.rmtree(d)

    def test_byte_entropy(self):
        self.assertLess(byte_entropy(b"hello world " * 100), 4.0)
        self.assertGreater(byte_entropy(os.urandom(4096)), 7.5)
        self.assertEqual(byte_entropy(b""), 0.0)

    def test_process_sensor_sees_a_new_obfuscated_process(self):
        ps = ProcessSensor()
        self.assertEqual(ps.poll(0), [])                                   # first look is the baseline
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(4)  # powershell -enc " + "QUJD" * 30])
        self.addCleanup(p.kill)
        time.sleep(0.6)
        ev = [e for e in ps.poll(1) if e["pid"] == p.pid]
        self.assertEqual(len(ev), 1)
        self.assertTrue(ev[0]["obfuscated"] and ev[0]["script_host"])
        self.assertEqual(ev[0]["name"], "python")
        self.assertNotIn("QUJD" * 30, ev[0]["cmd"])                         # long blobs are redacted
        self.assertEqual([e for e in ps.poll(2) if e["pid"] == p.pid], [])  # not new twice

    def test_process_describe_flags(self):
        ev, key = describe({"pid": 1, "ppid": 2, "name": "powershell.exe", "exe": "C:\\Users\\a\\AppData\\Local\\Temp\\powershell.exe",
                            "cmdline": ["powershell.exe", "-nop", "-w", "hidden", "-enc", "AAAA"]}, "WINWORD.EXE", set())
        self.assertTrue(ev["script_host"] and ev["obfuscated"] and ev["office_parent"] and ev["novel_program"])
        self.assertEqual(ev["exe_class"], "temp")
        ev2, _ = describe({"pid": 1, "ppid": 2, "name": "x", "exe": "C:\\u\\.ssh\\x.exe", "cmdline": []}, "", {key})
        self.assertEqual(ev2["exe"], "<sensitive>")

    def test_connection_sensor_sees_loopback_connection_and_listener(self):
        cs = ConnectionSensor()
        cs.poll(0)
        srv = socket.socket()
        self.addCleanup(srv.close)
        srv.bind(("127.0.0.1", 0)); srv.listen(1)
        cli = socket.create_connection(srv.getsockname())
        self.addCleanup(cli.close)
        ev = cs.poll(1)
        kinds = {e["kind"] for e in ev}
        self.assertEqual(kinds, {"conn_new", "listen_new"})
        c = next(e for e in ev if e["kind"] == "conn_new")
        self.assertTrue(c["private"] and c["novel"])
        self.assertFalse(next(e for e in ev if e["kind"] == "listen_new")["public_bind"])
        self.assertEqual(cs.poll(2), [])
        st = cs.state(); cs2 = ConnectionSensor(); cs2.load(st)
        self.assertEqual(cs2.dest.keys(), cs.dest.keys())

    def test_lan_sensor_detects_gateway_change_duplicates_new_devices_and_wifi(self):
        state = {"arp": {"192.168.1.1": "aa:aa:aa:aa:aa:01", "192.168.1.20": "bb:bb:bb:bb:bb:02"}, "wifi": {"ssid": "Home", "bssid": "11:11", "auth": "WPA2-Personal", "cipher": "CCMP"}}
        ls = LanSensor(arp_fn=lambda: dict(state["arp"]), gateway_fn=lambda: "192.168.1.1", wifi_fn=lambda: dict(state["wifi"]))
        self.assertEqual(ls.poll(0), [])                                   # baseline
        self.assertEqual(ls.poll(1), [])                                   # nothing changed
        state["arp"]["192.168.1.30"] = "cc:cc:cc:cc:cc:03"
        e = ls.poll(2)
        self.assertEqual((e[0]["kind"], e[0]["new_devices"], e[0]["gw_changed"]), ("lan", 1, False))
        state["arp"]["192.168.1.1"] = "dd:dd:dd:dd:dd:04"
        e = [x for x in ls.poll(3) if x["kind"] == "lan"][0]
        self.assertTrue(e["gw_changed"])
        self.assertEqual(e["old_gw_mac"], "aa:aa:aa:aa:aa:01")
        self.assertEqual([x for x in ls.poll(4) if x["kind"] == "lan" and x["gw_changed"]], [])   # told once
        state["arp"]["192.168.1.99"] = "dd:dd:dd:dd:dd:04"                  # same hardware address answers for the router's too
        e = [x for x in ls.poll(5) if x["kind"] == "lan"][0]
        self.assertEqual(e["arp_dup"], 1)
        del state["arp"]["192.168.1.99"]
        state["wifi"] = {"ssid": "Home", "bssid": "99:99", "auth": "Open", "cipher": "None"}
        w = [x for x in ls.poll(6) if x["kind"] == "wifi"][0]
        self.assertTrue(w["ap_changed"] and w["weak"])
        self.assertEqual([x for x in ls.poll(7) if x["kind"] == "wifi"], [])
        st = ls.state()
        ls2 = LanSensor(arp_fn=lambda: dict(state["arp"]), gateway_fn=lambda: "192.168.1.1", wifi_fn=lambda: dict(state["wifi"]))
        ls2.load(st)
        self.assertEqual(ls2.poll(8), [])                                   # a restart does not re-announce everything

    def test_netflow_sensor_counts_and_names(self):
        nf = NetflowSensor()
        self.assertEqual([e for e in nf.poll(0) if e["kind"] == "netstat"], [])
        nf.feed_name("Example.com", False); nf.feed_name("qzxw.bad", True)
        ev = nf.poll(1)
        self.assertEqual(sorted(e["kind"] for e in ev), ["dns", "dns", "netstat"])
        self.assertTrue(any(e["nx"] for e in ev if e["kind"] == "dns"))
        self.assertTrue(all(e["bytes_out"] >= 0 for e in ev if e["kind"] == "netstat"))

    def test_parse_dns_packet_with_scapy(self):
        try:
            from scapy.all import IP, UDP, DNS, DNSQR, DNSRR
        except ImportError:
            self.skipTest("scapy not installed")
        q = IP(dst="1.1.1.1") / UDP(dport=53) / DNS(qr=1, rcode=3, qd=DNSQR(qname="xk3jf9alqzd0.example.net"))
        self.assertEqual(parse_dns_packet(q), ("xk3jf9alqzd0.example.net", True))
        a = IP(dst="1.1.1.1") / UDP(dport=53) / DNS(qr=1, rcode=0, qd=DNSQR(qname="good.com"), an=DNSRR(rrname="good.com", rdata="1.2.3.4"))
        self.assertEqual(parse_dns_packet(a), ("good.com", False))
        self.assertIsNone(parse_dns_packet(IP() / UDP()))
        self.assertIsNone(parse_dns_packet(object()))


# ============================================================================================ neurons
def normal_vec(rng):
    return F.to_vector(synthetic_baseline(1, int(rng.integers(1 << 30)))[0])


class TestNeurons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.B = synthetic_baseline(4000, 7)
        cls.kit = KitEnsemble(fm=100, ad=200, cal=200)
        for w in cls.B[:1000]:
            cls.kit.process(F.to_vector(w))

    def test_phases_and_no_alarm_in_childhood(self):
        k = KitEnsemble(fm=20, ad=20, cal=20)
        phases = []
        for w in self.B[:70]:
            s, _ = k.process(F.to_vector(w))
            phases.append((k.phase, s is None))
        self.assertEqual([p for p, _ in phases][:2], ["mapping", "mapping"])
        self.assertTrue(all(none for ph, none in phases[:60]))            # no score until armed
        self.assertEqual(k.phase, "armed")
        self.assertIsNotNone(k.thr)

    def test_quiet_on_normal_and_loud_on_big_changes(self):
        k = self.kit
        self.assertEqual(k.phase, "armed")
        fa = np.mean([k.process(F.to_vector(w), learn=False)[0] > k.thr for w in self.B[1000:3000]])
        self.assertLess(fa, 0.015)
        big = {"ransom": ({"n_file_events": 300, "n_file_renamed": 200, "n_ext_change": 200}, 0.9),
               "scan": ({"n_conn_new": 120, "n_remote_ips": 100, "n_rare_port": 80}, 0.9),
               "dns": ({"dns_queries": 40, "dns_nx": 30, "dns_entropy": 4.4}, 0.6)}        # a few quiet signals among 30 are diluted: the Forge exists for those
        for name, (add, need) in big.items():
            hits = 0
            for w in self.B[3000:3100]:
                r = dict(w)
                for f, v in add.items():
                    r[f] = v if f == "dns_entropy" else r.get(f, 0) + v
                hits += k.process(F.to_vector(r), learn=False)[0] > k.thr
            self.assertGreater(hits / 100, need, name)

    def test_explanation_names_the_unusual_signals(self):
        k = self.kit
        r = dict(self.B[3200]); r["n_file_events"] += 300; r["n_ext_change"] += 200
        s, info = k.process(F.to_vector(r), learn=False)
        why = " ".join(k.explain(info, r))
        self.assertIn("changed the file extension", why)

    def test_save_and_load_are_exact(self):
        a = self.kit.to_arrays()
        k2 = KitEnsemble.from_arrays(a)
        for w in self.B[3300:3310]:
            v = F.to_vector(w)
            self.assertAlmostEqual(self.kit.process(v, learn=False)[0], k2.process(v, learn=False)[0], places=12)
        # arrays only: no pickle needed
        for key, arr in a.items():
            self.assertNotEqual(arr.dtype, object, key)

    def test_attacker_cannot_teach_it_in_one_step(self):
        k = KitEnsemble.from_arrays(self.kit.to_arrays())
        thr0 = k.thr
        r = dict(self.B[3400]); r.update(n_conn_new=150, n_remote_ips=120, n_rare_port=100, n_public_conn=140)
        v = F.to_vector(r)
        for _ in range(300):                                               # the same attack window, again and again, learning allowed
            s, _ = k.process(v, learn=True)
        self.assertGreater(s, k.thr)                                       # still flagged: it was never judged normal, so it was never learned
        self.assertLess(k.thr, 1.5 * thr0)

    def test_a_human_can_say_it_was_fine(self):
        k = KitEnsemble.from_arrays(self.kit.to_arrays())
        r = dict(self.B[3500]); r.update(n_conn_new=40, n_remote_ips=30, n_public_conn=35, net_bytes_out=2e7)
        v = F.to_vector(r)
        before = k.process(v, learn=False)[0]
        k.teach_normal(v, repeat=40)
        after = k.process(v, learn=False)[0]
        self.assertLess(after, before)

    def test_tinymlp_learns_and_roundtrips(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 4)); y = (X[:, 0] + X[:, 1] > 0).astype(float)
        m = TinyMLP(4, 8).fit(X, y, epochs=300)
        self.assertGreater(((m.predict(X) > 0.5) == y).mean(), 0.95)
        m2 = TinyMLP.from_arrays(m.arrays())
        np.testing.assert_allclose(m.predict(X), m2.predict(X))


# ============================================================================================ instincts
class TestInstincts(unittest.TestCase):
    def rules(self, evs, ins=None):
        ins = ins or Instincts()
        out = []
        for i, e in enumerate(evs):
            out += ins.observe(e, e.get("ts") or 100 + i)
        return [a["rule"] for a in out]

    def test_obfuscated_script_and_office_child(self):
        self.assertEqual(sorted(self.rules([proc_ev(script_host=True, obfuscated=True, office_parent=True, parent="winword", name="powershell")])),
                         ["office_spawns_tool", "script_obfuscated"])
        self.assertEqual(self.rules([proc_ev(script_host=True, obfuscated=False)]), [])                   # ordinary script: nothing
        self.assertEqual(self.rules([proc_ev(office_parent=True, parent="winword", name="calc")]), [])    # office starting a normal program

    def test_download_then_run_then_connect(self):
        ins = Instincts()
        dl = {"kind": "file", "op": "created", "loc": "downloads", "is_exec": True, "path": "C:/Users/a/Downloads/tool.exe", "ts": 100}
        self.assertEqual([a["rule"] for a in ins.observe(dl, 100)], [])
        run = proc_ev(exe="C:/Users/a/Downloads/tool.exe", exe_class="downloads", name="tool", pid=77)
        self.assertEqual([a["rule"] for a in ins.observe(run, 160)], ["fresh_download_started"])
        conn = {"kind": "conn_new", "pid": 77, "remote_ip": "203.0.113.9", "remote_port": 443, "private": False, "novel": True, "rare_port": False, "prog": "tool"}
        r = ins.observe(conn, 170)
        self.assertEqual([a["rule"] for a in r], ["temp_program_online"])
        self.assertEqual(r[0]["target"], {"kind": "connection", "pid": 77, "ip": "203.0.113.9"})
        # an hour later the same file is not 'fresh' any more
        ins2 = Instincts()
        ins2.observe(dl, 100)
        self.assertNotIn("fresh_download_started", [a["rule"] for a in ins2.observe(run, 100 + 4000)])

    def test_lolbin_and_listener_and_startup(self):
        ins = Instincts()
        ins.observe(proc_ev(pid=5, name="certutil", lolbin=True, exe_class="system"), 10)
        r = ins.observe({"kind": "conn_new", "pid": 5, "remote_ip": "198.51.100.7", "remote_port": 80, "private": False, "novel": True, "rare_port": False, "prog": "certutil"}, 11)
        self.assertEqual([a["rule"] for a in r], ["lolbin_network"])
        self.assertEqual([a["rule"] for a in ins.observe({"kind": "listen_new", "pid": 6, "prog": "x", "exe_class": "temp", "port": 4444, "public_bind": True}, 12)], ["public_listener"])
        self.assertEqual(ins.observe({"kind": "listen_new", "pid": 6, "prog": "x", "exe_class": "temp", "port": 4444, "public_bind": False}, 12), [])
        self.assertEqual(ins.observe({"kind": "listen_new", "pid": 6, "prog": "svc", "exe_class": "system", "port": 135, "public_bind": True}, 12), [])
        a = ins.observe({"kind": "file", "op": "created", "loc": "startup", "is_exec": True, "path": "C:/x/Startup/a.lnk"}, 13)
        self.assertEqual((a[0]["rule"], a[0]["severity"]), ("startup_change", "high"))

    def test_network_rules(self):
        ins = Instincts()
        r = ins.observe({"kind": "lan", "gw_changed": True, "gateway": "192.168.1.1", "old_gw_mac": "a", "gw_mac": "b", "new_devices": 0, "arp_dup": 0}, 1)
        self.assertEqual([a["rule"] for a in r], ["gateway_changed"])
        self.assertEqual([a["rule"] for a in ins.observe({"kind": "lan", "new_devices": 2, "gw_changed": False, "arp_dup": 1, "dup_macs": ["m"]}, 2)], ["arp_duplicate", "new_lan_device"])
        self.assertEqual([a["severity"] for a in ins.observe({"kind": "wifi", "ssid": "H", "weak": True, "ap_changed": False, "auth": "Open", "cipher": "None", "bssid": "x"}, 3)], ["low"])

    def test_window_rules(self):
        ins = Instincts()
        raw = {f: 0.0 for f in F.FEATURES}
        raw.update(n_file_events=200, n_ext_change=100, n_entropy_anomaly=20)
        self.assertEqual([a["rule"] for a in ins.end_window(raw, [])], ["ransomware_pattern"])
        raw.update(n_file_events=70, n_ext_change=0, n_entropy_anomaly=0)
        self.assertEqual(ins.end_window(raw, []), [])                       # a busy window alone is not an alarm (installs, syncs)
        raw.update(n_file_events=500)
        self.assertEqual([a["rule"] for a in ins.end_window(raw, [])], ["mass_file_change"])
        names = [{"kind": "dns", "name": f"{''.join(chr(97 + (i * 7 + j * 3) % 26) for j in range(14))}.net"} for i in range(8)]
        raw = {f: 0.0 for f in F.FEATURES}; raw.update(dns_queries=8, dns_nx=7)
        self.assertEqual([a["rule"] for a in ins.end_window(raw, names)], ["dns_generated_names"])
        calm = [{"kind": "dns", "name": n} for n in ["google.com", "microsoft.com", "github.com", "python.org", "wikipedia.org", "example.com", "openai.com", "apple.com"]]
        raw.update(dns_nx=0)
        self.assertEqual(ins.end_window(raw, calm), [])


# ============================================================================================ actuators
class TestActuators(unittest.TestCase):
    def test_requests_use_only_validated_values(self):
        a = {"severity": "high", "title": "x", "target": {"kind": "connection", "ip": "203.0.113.9"}}
        k, t, s = actuators.propose(a)
        self.assertEqual((k, t), ("block_connection", {"ip": "203.0.113.9"}))
        self.assertIn("remoteip=203.0.113.9", s)
        for bad_ip in ("1.2.3.4; del *", "8.8.8.8 & calc", "999.1.1.1", "$(whoami)"):
            self.assertIsNone(actuators.propose({**a, "target": {"kind": "connection", "ip": bad_ip}}))
        k, t, s = actuators.propose({"severity": "medium", "title": "x", "target": {"kind": "process", "pid": 1234}})
        self.assertEqual(k, "end_process")
        self.assertIsNone(actuators.propose({"severity": "medium", "title": "x", "target": {"kind": "process", "pid": "1234; calc"}}))
        self.assertIsNone(actuators.propose({"severity": "medium", "title": "x", "target": {"kind": "process", "pid": 4}}))   # system pid
        self.assertIsNone(actuators.propose({"severity": "low", "title": "x", "target": {"kind": "process", "pid": 1234}}))
        evil = "C:/a/$(calc).exe & del c:/"
        k, t, s = actuators.propose({"severity": "high", "title": "x", "target": {"kind": "file", "path": evil}})
        self.assertNotIn("calc", s)                                          # a file name never ends up in a command suggestion
        self.assertNotIn("del", s)


# ============================================================================================ the living service
class TestService(unittest.TestCase):
    def test_alert_is_stored_with_a_request_and_nothing_else_happens(self):
        d = tmp()
        f = Fake()
        s, T = make_service(d, [f])
        run_windows(s, T, 20)
        f.q = [proc_ev(pid=99, name="powershell", parent="winword", script_host=True, obfuscated=True, office_parent=True, ts=T[0])]
        out = run_windows(s, T, 1)
        titles = [a["title"] for a in out[-1]["alerts"]]
        self.assertTrue(any("encoded" in t for t in titles))
        self.assertTrue(any("started powershell" in t for t in titles))
        reqs = s.store.requests()
        self.assertEqual([(r["kind"], r["target"]) for r in reqs], [("end_process", {"pid": 99})])         # one request, de-duplicated
        self.assertEqual(s.status()["can_do"], ["alert", "report", "request_action"])
        s.shutdown(); shutil.rmtree(d)

    def test_cooldown_and_hourly_cap(self):
        d = tmp()
        f = Fake()
        s, T = make_service(d, [f], cfg={"max_alerts_per_hour": 3})
        run_windows(s, T, 12)
        count = 0
        for i in range(10):
            f.q = [proc_ev(pid=100 + i, name=f"powershell{i}", script_host=True, obfuscated=True, ts=T[0])]
            out = run_windows(s, T, 1)
            count += len(out[-1]["alerts"])
        self.assertEqual(count, 3)
        self.assertGreater(s.status()["dropped_by_cap"], 0)
        # same key inside the cool-down: silent
        s2, T2 = make_service(tmp(), [Fake()])
        run_windows(s2, T2, 12)
        s2.sensors[0].q = [proc_ev(name="powershell", script_host=True, obfuscated=True)]
        self.assertEqual(len(run_windows(s2, T2, 1)[-1]["alerts"]), 1)
        s2.sensors[0].q = [proc_ev(name="powershell", script_host=True, obfuscated=True)]
        self.assertEqual(len(run_windows(s2, T2, 1)[-1]["alerts"]), 0)
        s.shutdown(); s2.shutdown()

    def test_restart_continues_where_it_stopped(self):
        d = tmp()
        s, T = make_service(d)
        run_windows(s, T, 20)
        s.sensors[0].q = [proc_ev(name="powershell", script_host=True, obfuscated=True)]
        run_windows(s, T, 1)
        n_before, phase, cool = s.brain.kit.n, s.brain.kit.phase, dict(s.brain.cool)
        self.assertEqual(phase, "armed")
        s.shutdown()
        s2, T2 = make_service(d, clock=T)
        self.assertEqual((s2.boots, s2.brain.kit.phase, s2.brain.kit.n), (2, "armed", n_before))
        self.assertEqual(s2.brain.cool, cool)                                # cool-downs survive: no repeat alert after a restart
        s2.sensors[0].q = [proc_ev(name="powershell", script_host=True, obfuscated=True)]
        self.assertEqual(len(run_windows(s2, T2, 1)[-1]["alerts"]), 0)
        s2.shutdown(); shutil.rmtree(d)

    def test_survives_a_crash_between_snapshots(self):
        d = tmp()
        s, T = make_service(d, cfg={"snapshot_every": 5})
        run_windows(s, T, 23)
        saved_n = s.store.load_snapshot()["cycle"]
        self.assertEqual(saved_n, 20)
        s._lock.close()                                                      # simulate kill -9: no shutdown, no final save
        s.store.close()
        s2, T2 = make_service(d, clock=T)
        self.assertEqual(s2.cycle, 20)
        self.assertEqual(s2.brain.kit.phase, "armed")
        s2.shutdown()

    def test_single_instance(self):
        d = tmp()
        s, T = make_service(d)
        with self.assertRaises(RuntimeError):
            Sentinel(d, [Fake()], {}, log=lambda *a: None).boot()
        s.shutdown()
        s3, _ = make_service(d)                                              # after a clean shutdown the lock is free again
        s3.shutdown()
        f = acquire_lock(os.path.join(d, "x.lock"))
        self.assertIsNone(acquire_lock(os.path.join(d, "x.lock")))
        f.close()

    def test_pause_file_and_api_pause(self):
        d = tmp()
        f = Fake()
        s, T = make_service(d, [f])
        run_windows(s, T, 12)
        open(os.path.join(d, "PAUSE"), "w").close()
        f.q = [proc_ev(name="powershell", script_host=True, obfuscated=True)]
        T[0] += 30
        self.assertEqual(s.step(), {"paused": True})
        self.assertEqual(f.q != [], True)                                    # a paused sentinel does not even poll its sensors
        self.assertEqual(s.status()["paused"], "file")
        os.remove(os.path.join(d, "PAUSE"))                                  # a person (the test) removes it; the Sentinel itself cannot
        s.set_paused(True)
        self.assertEqual(s.status()["paused"], "api")
        T[0] += 30
        self.assertEqual(s.step(), {"paused": True})
        s.set_paused(False)
        self.assertIsNone(s.status()["paused"])
        s.shutdown()

    def test_sensor_failure_does_not_stop_the_others(self):
        class Broken(Sensor):
            name = "broken"

            def poll(self, now=None):
                raise OSError("denied")
        d = tmp()
        f = Fake()
        s, T = make_service(d, [Broken(), f])
        f.q = [proc_ev(name="powershell", script_host=True, obfuscated=True)]
        out = run_windows(s, T, 1)
        self.assertEqual(len(out[-1]["alerts"]), 1)
        self.assertIn("denied", s.status()["sensors"][0]["error"])
        s.shutdown()

    def test_feedback_benign_silences_and_teaches_real_is_never_baseline(self):
        d = tmp()
        f = Fake()
        s, T = make_service(d, [f])
        run_windows(s, T, 20)
        f.q = [proc_ev(name="powershell", script_host=True, obfuscated=True)]
        a = run_windows(s, T, 1)[-1]["alerts"][0]
        wid = s.store.alert(a["id"])["window_id"]
        self.assertEqual(s.feedback(a["id"], "real"), "ok")
        self.assertEqual(s.store.alert(a["id"])["status"], "real")
        self.assertEqual(s.store.window(wid)["label"], "real")
        n_normal = len(s.store.baseline_windows())
        self.assertNotIn(s.store.window(wid)["raw"], s.store.baseline_windows())
        f.q = [proc_ev(name="python", script_host=True, obfuscated=True, pid=555)]
        T[0] += 700                                                          # beyond the cool-down
        b = run_windows(s, T, 1)[-1]["alerts"][0]
        s.feedback(b["id"], "benign")
        f.q = [proc_ev(name="python", script_host=True, obfuscated=True, pid=556)]
        T[0] += 3000
        again = [a for a in run_windows(s, T, 1)[-1]["alerts"] if a["rule"] == "script_obfuscated"]
        self.assertEqual(again, [])                                          # that rule + program is silenced for 30 days (the reflex has its own key)
        self.assertEqual(s.status()["judged"], {"real": 1, "benign": 1, "precision": 0.5})
        with self.assertRaises(ValueError):
            s.feedback(a["id"], "maybe")
        self.assertEqual(s.feedback(99999, "real"), "unknown alert")
        s.shutdown()
        s2, _ = make_service(d, clock=T)
        self.assertTrue(s2.brain.suppress)                                   # suppression survives a restart
        s2.shutdown()

    def test_prune_keeps_open_alerts(self):
        d = tmp()
        s, T = make_service(d)
        st = s.store
        old = T[0] - 40 * 86400
        wid = st.add_window(old, {"a": 1}, 0.1, 0)
        a1 = st.add_alert({"rule": "r", "severity": "low", "title": "t", "why": "w", "key": "k1"}, old)
        a2 = st.add_alert({"rule": "r", "severity": "low", "title": "t", "why": "w", "key": "k2"}, old)
        st.judge(a2, "benign")
        st.prune(T[0], 30)
        self.assertIsNotNone(st.alert(a1))
        self.assertIsNone(st.alert(a2))
        self.assertIsNone(st.window(wid))
        s.shutdown()

    def test_nothing_is_written_outside_home(self):
        d = tmp()
        before = {p for p in glob.glob("/tmp/*")}
        s, T = make_service(d)
        run_windows(s, T, 15)
        s.shutdown()
        extra = {p for p in glob.glob("/tmp/*")} - before
        self.assertEqual({os.path.realpath(p) for p in extra if os.path.isfile(p)}, set())


# ============================================================================================ the brain with forged neurons
class TestBrainWithNeurons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = synthetic_baseline(4000, 3)

    def forged_store(self, name="ransom", n_pos=150, kind="machine"):
        d = tmp()
        st = SentinelStore(d)
        spec = S.validate(MockTeacher().design(name))
        nid, ver, state, m = FM.create(st, spec, name, "mock", self.base, kind, seed=0)
        return d, st, nid, state, m

    def test_forged_neuron_end_to_end_proposal_approval_live_alert_disable(self):
        d, st, nid, state, m = self.forged_store("catch ransomware")
        self.assertEqual(state, "proposed")
        self.assertTrue(m["gate"]["passed"])
        with self.assertRaises(SystemExit):
            FM.approve(st, nid, "")                                           # a name is required
        # not live until a person approves
        T = [1.8e9]
        s = Sentinel(d, [Fake()], {"window_s": 30, "port": 0, "reflex": {"fm_windows": 5, "ad_windows": 5, "cal_windows": 5}, "cortex": {"enabled": False}}, clock=lambda: T[0], log=lambda *a: None).boot()
        raw = dict(self.base[10]); raw.update(n_file_events=250, n_file_renamed=120, n_ext_change=120, n_entropy_anomaly=15)
        vec = F.to_vector(raw)
        self.assertEqual(s.brain.neurons, [])
        FM.approve(st, nid, "tester")                                         # approved on the command line while the service runs
        s.brain.reload_neurons()
        self.assertEqual([n.name for n in s.brain.neurons], ["files_being_encrypted"])
        alerts, score, flagged = s.brain.close_window(raw, [], T[0])
        self.assertTrue(any(a["source"].startswith("neuron:files_being_encrypted") for a in alerts))
        self.assertTrue(flagged)
        calm, _, fl = s.brain.close_window(dict(self.base[11]), [], T[0])
        self.assertFalse(any(a["source"].startswith("neuron:") for a in calm))
        # disable: silent again, and the record stays
        st.set_neuron_state(nid, "disabled", "tester")
        s.brain.reload_neurons()
        self.assertEqual(s.brain.neurons, [])
        self.assertEqual([n["state"] for n in st.neurons()], ["disabled"])
        s.shutdown(); shutil.rmtree(d)

    def test_demo_and_failed_neurons_cannot_be_approved(self):
        d, st, nid, state, m = self.forged_store("catch ransomware", kind="synthetic")
        self.assertEqual(state, "demo")
        with self.assertRaises(SystemExit):
            FM.approve(st, nid, "me")
        d2 = tmp(); st2 = SentinelStore(d2)
        spec = S.validate(MockTeacher().design("catch ransomware"))
        nid2, _, state2, m2 = FM.create(st2, spec, "p", "mock", self.base[:900], "machine")      # too little history
        self.assertEqual(state2, "failed")
        self.assertFalse(m2["gate"]["passed"])
        with self.assertRaises(SystemExit):
            FM.approve(st2, nid2, "me")
        self.assertEqual([c["check"] for c in m2["gate"]["checks"] if not c["ok"]][:1], ["untouched later normal windows to test false alarms"])

    def test_versions_increment_and_nothing_is_deleted(self):
        d, st, nid, state, m = self.forged_store("catch ransomware")
        spec = S.validate(MockTeacher().design("catch ransomware"))
        nid2, ver2, *_ = FM.create(st, spec, "again", "mock", self.base, "machine", seed=1)
        self.assertEqual(ver2, 2)
        self.assertEqual(len(st.neurons()), 2)
        FM.approve(st, nid, "a")
        FM.approve(st, nid2, "a")
        self.assertEqual(len(st.neurons("active")), 2)

    def test_gate_rejects_an_inseparable_purpose(self):
        d, st, nid, state, m = self.forged_store("data being uploaded out of this machine")
        self.assertEqual(state, "failed")                                     # slow exfiltration is not separable from heavy-tailed traffic with these features
        self.assertLess(m["mean_recall"], 0.9)
        self.assertEqual(st.neurons("active"), [])


# ============================================================================================ Forge: spec, synthesis, training
class TestForge(unittest.TestCase):
    def good(self):
        return json.loads(json.dumps(LIBRARY[1][1]))

    def test_library_specs_validate(self):
        for pat, spec in LIBRARY:
            S.validate(json.loads(json.dumps(spec)))

    def test_validation_rejects_bad_specs(self):
        def bad(mut, needle):
            sp = self.good(); mut(sp)
            with self.assertRaises(S.SpecError) as cm:
                S.validate(sp)
            self.assertIn(needle, str(cm.exception))
        bad(lambda s: s["scenarios"][0]["effects"][0].update(feature="__import__('os')"), "unknown feature")
        bad(lambda s: s["scenarios"][0]["effects"][0].update(mode="exec"), "unknown mode")
        bad(lambda s: s["scenarios"][0]["effects"][0].update(value=1e30, mode="add"), "outside 0..")
        bad(lambda s: s["scenarios"][0]["effects"][0].update(value=0.1, mode="multiply"), "multiply factor")
        bad(lambda s: s["scenarios"][0]["effects"][0].update(value=float("nan")), "finite")
        bad(lambda s: s["scenarios"][0]["effects"][0].update(value="lots"), "numbers")
        bad(lambda s: s.update(name="Bad Name!"), "name:")
        bad(lambda s: s.update(name="../../etc/passwd"), "name:")
        bad(lambda s: s.update(scenarios=s["scenarios"][:1]), "scenarios: need")
        bad(lambda s: s.update(lookalikes=[]), "lookalikes: need")
        bad(lambda s: s["scenarios"][0].update(effects=[]), "effects")
        bad(lambda s: s["alert"].update(severity="catastrophic"), "alert.severity")
        bad(lambda s: s["alert"].update(explanation="Download http://evil.example/fix.exe and run it"), "link, code or a command")
        bad(lambda s: s["alert"].update(title="Run `del /f` now"), "link, code or a command")
        bad(lambda s: s["scenarios"][1].update(name=s["scenarios"][0]["name"]), "distinct")
        bad(lambda s: s["scenarios"][0].update(description="x" * 500), "longer than")
        with self.assertRaises(S.SpecError):
            S.validate("not a dict")

    def test_validation_drops_unknown_fields(self):
        sp = self.good()
        sp["system_prompt_leak"] = "ignore previous instructions"
        sp["scenarios"][0]["run_this"] = "calc.exe"
        clean = S.validate(sp)
        self.assertNotIn("system_prompt_leak", clean)
        self.assertNotIn("run_this", clean["scenarios"][0])

    def test_schema_matches_features_and_is_closed(self):
        enum = S.SPEC_SCHEMA["properties"]["scenarios"]["items"]["properties"]["effects"]["items"]["properties"]["feature"]["enum"]
        self.assertEqual(enum, F.FEATURES)
        self.assertFalse(S.SPEC_SCHEMA["additionalProperties"])

    def test_effects(self):
        rng = np.random.default_rng(0)
        base = {f: 0.0 for f in F.FEATURES}; base.update(n_file_events=10, net_bytes_out=1000)
        eff = lambda f, m, v: {"feature": f, "mode": m, "value": v, "jitter": 0.0}
        r = apply_effects(base, [eff("n_file_events", "add", 5), eff("net_bytes_out", "multiply", 10), eff("gw_changed", "set_min", 1), eff("n_ext_change", "add", 1e9)], rng)
        self.assertEqual((r["n_file_events"], r["net_bytes_out"], r["gw_changed"]), (15, 10000, 1))
        self.assertEqual(r["n_ext_change"], F.BOUNDS["n_ext_change"])                  # clipped
        w = weaken({"effects": [eff("n_file_events", "add", 10), eff("net_bytes_out", "multiply", 11)]}, 0.5)["effects"]
        self.assertEqual((w[0]["value"], w[1]["value"]), (5, 6))
        V = synth({"effects": [eff("n_file_events", "add", 50)]}, [base] * 5, 7, rng)
        self.assertEqual(V.shape, (7, len(F.FEATURES)))

    def test_training_is_deterministic_and_quiet_on_normal_days(self):
        B = synthetic_baseline(3000, 5)
        spec = S.validate(MockTeacher().design("ransomware"))
        n1, m1 = forge_neuron(spec, B, "machine", seed=3, n_pos=120, epochs=200, compare_reflex=False)
        n2, m2 = forge_neuron(spec, B, "machine", seed=3, n_pos=120, epochs=200, compare_reflex=False)
        self.assertEqual(m1["threshold"], m2["threshold"])
        self.assertEqual(m1["n_train"] + m1["n_calibration"] + m1["n_test"], 3000)
        Zte = n1.score_many(np.array([F.to_vector(w) for w in B[2400:]]))
        self.assertEqual(int((Zte >= n1.thr).sum()), m1["test_false_alarms"])
        self.assertLess(m1["test_false_alarms"] / m1["n_test"], 0.01)
        blob = n1.to_blob()
        n3 = ForgeNeuron.from_blob(blob, spec["name"], spec)
        v = F.to_vector(B[5])
        self.assertAlmostEqual(n1.score(v), n3.score(v), places=12)

    def test_test_windows_are_never_used_to_pick_the_alarm_line(self):
        # make the last quarter very different: false alarms must show up in the test count rather than be tuned away
        B = synthetic_baseline(3000, 5)
        for w in B[2250:]:
            w["n_file_events"] += 40; w["n_ext_change"] = 30; w["n_file_renamed"] = 30
        spec = S.validate(MockTeacher().design("ransomware"))
        n, m = forge_neuron(spec, B, "machine", seed=0, n_pos=120, epochs=200, compare_reflex=False)
        self.assertGreater(m["test_false_alarms"], 0)
        self.assertFalse(m["gate"]["passed"])

    def test_reflex_reference_is_reported(self):
        B = synthetic_baseline(3000, 5)
        spec = S.validate(MockTeacher().design("dns tunnel"))
        n, m = forge_neuron(spec, B, "machine", seed=0, n_pos=120, epochs=200)
        self.assertIsNotNone(m["reflex_reference"])
        self.assertGreater(m["mean_recall"], m["reflex_reference"]["mean_recall"] + 0.3)   # the forged neuron adds what the reflex lacks here

    def test_report_is_honest(self):
        d = tmp(); st = SentinelStore(d)
        spec = S.validate(MockTeacher().design("ransomware"))
        FM.create(st, spec, "p", "mock", synthetic_baseline(3000, 5), "synthetic")
        text = FM.report(st.neurons()[0])
        self.assertIn("SYNTHETIC", text)
        self.assertIn("does not prove", text)
        self.assertIn("NOT PASSED", text)

    def test_cli_create_list_show_approve(self):
        d = tmp()
        import io, contextlib
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            FM.main(["create", "--home", d, "--purpose", "wifi router spoofing", "--synthetic-baseline", "3000"])
            FM.main(["list", "--home", d])
        self.assertIn("local_network_tampering", out.getvalue())
        self.assertIn("demo", out.getvalue())
        with self.assertRaises(SystemExit):
            FM.main(["approve", "1", "--home", d, "--by", "me"])
        with self.assertRaises(SystemExit):
            FM.main(["create", "--home", d, "--purpose", "x", "--spec", os.path.join(d, "missing.json")])


# ============================================================================================ Forge: the Claude teacher against a fake server
class FakeClaude:
    """A local stand-in for the Messages API. Records every request."""

    def __init__(self, reply=None, status=200, stop_reason="end_turn", text=None):
        self.requests = []
        outer = self
        spec = reply if reply is not None else json.loads(json.dumps(LIBRARY[1][1]))

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(self.rfile.read(n))})
                body = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "stop_reason": stop_reason, "stop_sequence": None,
                        "content": [] if stop_reason == "refusal" else [{"type": "text", "text": text if text is not None else json.dumps(spec)}],
                        "usage": {"input_tokens": 10, "output_tokens": 20}}
                if status != 200:
                    body = {"type": "error", "error": {"type": "api_error", "message": "boom"}}
                b = json.dumps(body).encode()
                self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown(); self.srv.server_close()


class TestClaudeTeacher(unittest.TestCase):
    def teacher(self, fake, **kw):
        import anthropic
        client = anthropic.Anthropic(api_key="test-key", base_url=fake.url, max_retries=0)
        return ClaudeTeacher(client=client, **kw)

    def test_request_shape_and_answer(self):
        fake = FakeClaude()
        self.addCleanup(fake.close)
        t = self.teacher(fake)
        spec = t.design("detect files being encrypted", {"n_file_events": {"median": 5, "p95": 20}})
        S.validate(spec)
        r = fake.requests[0]
        self.assertTrue(r["path"].startswith("/v1/messages"))
        b = r["body"]
        self.assertEqual(b["model"], "claude-opus-5-5")
        self.assertEqual(b["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(b["output_config"]["format"]["schema"], S.SPEC_SCHEMA)
        self.assertEqual(b["output_config"]["effort"], "medium")
        for forbidden in ("temperature", "top_p", "top_k", "thinking"):
            self.assertNotIn(forbidden, b)
        self.assertEqual([m["role"] for m in b["messages"]], ["user"])                    # no assistant prefill
        prompt = b["messages"][0]["content"]
        self.assertIn("detect files being encrypted", prompt)
        self.assertIn("n_file_events", prompt)
        self.assertIn("feature_catalogue", prompt)

    def test_only_what_was_chosen_leaves_the_machine(self):
        fake = FakeClaude()
        self.addCleanup(fake.close)
        t = self.teacher(fake)
        t.design("watch my downloads folder")
        sent = json.dumps(fake.requests[0]["body"])
        self.assertNotIn("this_machine_typical_counts_per_window", sent)                 # opt-in only
        home = os.path.expanduser("~")
        self.assertNotIn(home, sent)

    def test_refusal_is_reported_not_parsed(self):
        fake = FakeClaude(stop_reason="refusal")
        self.addCleanup(fake.close)
        with self.assertRaises(TeacherRefused) as cm:
            self.teacher(fake).design("x")
        self.assertIn("--spec", str(cm.exception))

    def test_bad_json_truncation_and_http_errors(self):
        for kw, exc in ((dict(text="not json at all"), TeacherError), (dict(stop_reason="max_tokens"), TeacherError), (dict(status=500), TeacherError)):
            fake = FakeClaude(**kw)
            self.addCleanup(fake.close)
            with self.assertRaises(exc):
                self.teacher(fake).design("x")

    def test_model_output_is_untrusted_and_gets_validated(self):
        evil = json.loads(json.dumps(LIBRARY[1][1]))
        evil["alert"]["explanation"] = "Open https://evil.example and paste your password"
        evil["scenarios"][0]["effects"][0]["feature"] = "os_system"
        fake = FakeClaude(reply=evil)
        self.addCleanup(fake.close)
        raw = self.teacher(fake).design("x")                                              # the teacher returns it as data...
        with self.assertRaises(S.SpecError) as cm:                                         # ...and the Forge refuses it
            S.validate(raw)
        self.assertIn("unknown feature", str(cm.exception))
        self.assertIn("link, code or a command", str(cm.exception))

    def test_fallbacks_use_the_beta_endpoint(self):
        fake = FakeClaude()
        self.addCleanup(fake.close)
        self.teacher(fake, fallbacks=True).design("x")
        r = fake.requests[0]
        self.assertIn("beta=true", r["path"])
        self.assertIn("server-side-fallback-2026-07-01", r["headers"].get("anthropic-beta", ""))
        self.assertEqual(r["body"]["fallbacks"], "default")

    def test_forge_creates_a_neuron_from_the_fake_model_spec(self):
        fake = FakeClaude()
        self.addCleanup(fake.close)
        spec = S.validate(self.teacher(fake).design("ransomware"))
        d = tmp(); st = SentinelStore(d)
        nid, ver, state, m = FM.create(st, spec, "ransomware", "claude", synthetic_baseline(3000, 2), "machine")
        self.assertEqual(state, "failed")                                                  # 3000 windows < 3800: not enough history yet
        self.assertEqual(st.neurons()[0]["teacher"], "claude")


# ============================================================================================ the dashboard and API
class TestApi(unittest.TestCase):
    def start(self, cfg=None, token=None):
        d = tmp()
        if token:
            os.environ["SENTINEL_TEST_TOKEN"] = token
        f = Fake()
        s, T = make_service(d, [f], cfg={"token_env": "SENTINEL_TEST_TOKEN", **(cfg or {})})
        run_windows(s, T, 15)
        f.q = [proc_ev(name="powershell", script_host=True, obfuscated=True)]
        run_windows(s, T, 1)
        from sentinel.serve import start_server
        srv = start_server(s)
        self.addCleanup(lambda: (srv.shutdown(), srv.server_close(), s.shutdown(), os.environ.pop("SENTINEL_TEST_TOKEN", None)))
        return s, f"http://127.0.0.1:{s.port_bound}"

    def req(self, url, data=None, headers=None, method=None):
        r = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(r, timeout=5) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as ex:
            return ex.code, ex.read().decode()

    def test_dashboard_status_and_feedback(self):
        s, u = self.start()
        code, body = self.req(u + "/")
        self.assertEqual(code, 200)
        self.assertIn("Open alerts", body)
        self.assertIn("encoded or hidden command", body)
        self.assertIn("What I cannot see", body)
        self.assertEqual(json.loads(self.req(u + "/health")[1]), {"ok": True})
        st = json.loads(self.req(u + "/status")[1])
        self.assertEqual(st["can_do"], ["alert", "report", "request_action"])
        al = [a for a in json.loads(self.req(u + "/alerts?status=open")[1])["alerts"] if a["rule"] == "script_obfuscated"]
        self.assertEqual(len(al), 1)
        code, body = self.req(u + "/feedback", json.dumps({"alert_id": al[0]["id"], "verdict": "benign"}).encode(), {"Content-Type": "application/json"})
        self.assertEqual(code, 200)
        self.assertEqual(s.store.alert(al[0]["id"])["status"], "benign")
        code, _ = self.req(u + "/feedback", b"alert_id=1&verdict=banana", {"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(code, 400)
        self.assertEqual(self.req(u + "/pause", b"{}")[0], 200)
        self.assertEqual(s.status()["paused"], "api")
        self.assertEqual(self.req(u + "/resume", b"{}")[0], 200)
        self.assertIsNone(s.status()["paused"])
        self.assertEqual(self.req(u + "/nope")[0], 404)

    def test_html_escapes_hostile_text(self):
        s, u = self.start()
        s.store.add_alert({"rule": "x", "severity": "high", "title": "<script>alert(1)</script>", "why": "<img src=x onerror=1>", "key": "k", "evidence": ["<b>boom</b>"]}, time.time())
        body = self.req(u + "/")[1]
        self.assertNotIn("<script>alert(1)", body)
        self.assertNotIn("<img src=x", body)
        self.assertIn("&lt;script&gt;", body)

    def test_form_post_from_the_dashboard_redirects(self):
        s, u = self.start()
        aid = next(a["id"] for a in s.store.alerts("open") if a["rule"] == "script_obfuscated")

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None
        op = urllib.request.build_opener(NoRedirect)
        r = urllib.request.Request(u + "/feedback", data=f"alert_id={aid}&verdict=real".encode(), headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            op.open(r, timeout=5)
        except urllib.error.HTTPError as ex:
            self.assertEqual(ex.code, 303)
        self.assertEqual(s.store.alert(aid)["status"], "real")

    def test_token_enforced(self):
        s, u = self.start(token="s3cret")
        self.assertEqual(self.req(u + "/status")[0], 401)
        self.assertEqual(self.req(u + "/status", headers={"Authorization": "Bearer wrong"})[0], 401)
        self.assertEqual(self.req(u + "/status", headers={"Authorization": "Bearer s3cret"})[0], 200)
        self.assertEqual(self.req(u + "/status?token=s3cret")[0], 200)
        self.assertEqual(self.req(u + "/pause", b"{}")[0], 401)
        self.assertEqual(self.req(u + "/health")[0], 200)

    def test_refuses_to_listen_publicly_without_a_token(self):
        d = tmp()
        s, T = make_service(d, cfg={"host": "0.0.0.0", "token_env": "SENTINEL_NO_SUCH_TOKEN"})
        from sentinel.serve import start_server
        with self.assertRaises(RuntimeError):
            start_server(s)
        s.shutdown()

    def test_cross_site_and_rebinding_requests_are_refused(self):
        s, u = self.start()
        aid = next(a["id"] for a in s.store.alerts("open") if a["rule"] == "script_obfuscated")
        body = json.dumps({"alert_id": aid, "verdict": "benign"}).encode()
        code, _ = self.req(u + "/feedback", body, {"Origin": "http://evil.example", "Content-Type": "application/json"})
        self.assertEqual(code, 401)
        code, _ = self.req(u + "/feedback", body, {"Host": "evil.example", "Content-Type": "application/json"})
        self.assertEqual(code, 401)
        self.assertEqual(s.store.alert(aid)["status"], "open")
        code, _ = self.req(u + "/status", headers={"Host": "attacker.example:80"})
        self.assertEqual(code, 401)


# ============================================================================================ the command line
class TestCli(unittest.TestCase):
    def test_init_run_status_alerts(self):
        d = tmp(); w = tmp()
        import io, contextlib
        from sentinel import __main__ as M
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            M.main(["init", "--home", d, "--window_s", "1", "--poll_s", "0.2", "--port", "0", "--watch", w])
        cfg = json.load(open(os.path.join(d, "sentinel.json")))
        cfg["reflex"] = {"fm_windows": 4, "ad_windows": 4, "cal_windows": 4}
        json.dump(cfg, open(os.path.join(d, "sentinel.json"), "w"))
        with contextlib.redirect_stdout(out):
            M.main(["run", "--home", d, "--max_steps", "20"])
            M.main(["status", "--home", d])
            M.main(["alerts", "--home", d])
            M.main(["pause", "--home", d])
        self.assertIn("windows", out.getvalue())
        st = SentinelStore(d)
        self.assertGreaterEqual(st.n_windows(), 3)
        self.assertTrue(st.meta_get("paused"))
        shutil.rmtree(d); shutil.rmtree(w)


if __name__ == "__main__":
    unittest.main()
