"""How much traffic, and what names are being looked up. Volumes come from the interface counters (no admin rights needed).
Name lookups come from, in order: a packet sniffer when allowed (scapy, needs Npcap/administrator or root), else the Windows DNS cache
(read-only `ipconfig /displaydns`). Payloads are encrypted for the most part and are never looked at; only names, counts and timing are."""
import sys
import threading

import psutil

from . import Sensor
from .. import probes


def parse_dns_packet(pkt):
    """scapy packet -> (name, failed) for a DNS response, or None."""
    try:
        from scapy.layers.dns import DNS
        if pkt.haslayer(DNS) and pkt[DNS].qr == 1 and pkt[DNS].qd is not None:
            name = pkt[DNS].qd.qname.decode("ascii", "ignore").rstrip(".").lower()
            return name, pkt[DNS].rcode == 3
    except Exception:                                  # a malformed packet must never take the sensor down
        return None
    return None


class NetflowSensor(Sensor):
    name = "netflow_dns"

    def __init__(self, sniff=False, iface=None):
        self.last = None
        self.names = []
        self.lock = threading.Lock()
        self.cache_seen = None
        self.mode = "dns-cache" if sys.platform.startswith("win") else "none"
        self.thread = None
        self._stop = threading.Event()
        if sniff:
            try:
                from scapy.all import sniff as _sniff
            except ImportError as ex:
                self.reason = f"scapy missing ({ex}); name lookups not sniffed"
                return

            def run():
                try:
                    _sniff(filter="udp port 53", prn=self._on_pkt, store=False, iface=iface, stop_filter=lambda p: self._stop.is_set())
                except Exception as ex:               # no rights / no driver: say so and carry on without it
                    self.reason = f"packet sniffer unavailable: {ex}"
                    self.mode = "dns-cache" if sys.platform.startswith("win") else "none"

            self.mode = "sniff"
            self.thread = threading.Thread(target=run, daemon=True)
            self.thread.start()

    def _on_pkt(self, pkt):
        r = parse_dns_packet(pkt)
        if r:
            with self.lock:
                self.names.append(r)

    def feed_name(self, name, failed=False):
        with self.lock:
            self.names.append((name, failed))

    def poll(self, now=None):
        events = []
        io = psutil.net_io_counters()
        cur = (io.bytes_sent, io.bytes_recv, io.packets_sent + io.packets_recv)
        if self.last is not None:
            events.append({"kind": "netstat", "bytes_out": max(cur[0] - self.last[0], 0), "bytes_in": max(cur[1] - self.last[1], 0),
                           "pkts": max(cur[2] - self.last[2], 0), "ts": now})
        self.last = cur
        if self.mode == "dns-cache":
            cache = {n for n, _ in probes.parse_displaydns(probes.run_readonly("dns_cache"))}
            fails = {n for n, f in probes.parse_displaydns(probes.run_readonly("dns_cache")) if f}
            if self.cache_seen is not None:
                for n in sorted(cache - self.cache_seen):
                    self.names.append((n, n in fails))
            self.cache_seen = cache
        with self.lock:
            names, self.names = self.names, []
        for n, failed in names:
            events.append({"kind": "dns", "name": n, "nx": bool(failed), "ts": now})
        return events

    def close(self):
        self._stop.set()

    def info(self):
        d = super().info()
        d["dns_mode"] = self.mode
        return d
