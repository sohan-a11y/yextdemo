"""How answers come back from the outside world.

A verifier has `poll(habitat, now) -> list of (entity_id, label, source)` and is called at the start of every cycle.
Real deployments normally need NO verifier: people or systems send answers through the API / CSV inbox. Two helpers:

  ReplayVerifier        answers pending checks from a table of known outcomes after a delay. THIS IS A SIMULATION OF THE
                        OUTSIDE WORLD (used for tests and for demonstrations on historical data).
  HttpLivenessVerifier  a real, automatic but PROXY check: label 1 ("needs attention") when the entity's website is unreachable
                        or answers with an error. A dead website is evidence, not proof, that a business changed.
"""
import time
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor


class ReplayVerifier:
    def __init__(self, truth, delay_cycles=1):
        self.truth, self.delay = truth, delay_cycles
        self.simulated = True

    def poll(self, habitat, now):
        out = []
        cyc = habitat.cycle_no
        for p in habitat.store.pending():
            if p["cycle"] is not None and cyc - p["cycle"] >= self.delay and p["entity_id"] in self.truth:
                out.append((p["entity_id"], self.truth[p["entity_id"]], "replay"))
        return out


class HttpLivenessVerifier:
    def __init__(self, url_of, timeout=8, workers=8, user_agent="organism-liveness-check/1.0", max_per_poll=200):
        self.url_of, self.timeout, self.workers, self.ua, self.cap = url_of, timeout, workers, user_agent, max_per_poll
        self.simulated = False
        self.checked = set()

    def _check(self, entity_id):
        url = self.url_of(entity_id)
        if not url:
            return None
        if not url.startswith("http"):
            url = "http://" + url
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.ua}, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return 0 if r.status < 400 else 1
        except urllib.error.HTTPError as e:
            return 1 if e.code in (404, 410) or e.code >= 500 else 0       # 403/401 means a live site that refuses robots
        except Exception:
            return 1

    def poll(self, habitat, now):
        todo = [p["entity_id"] for p in habitat.store.pending() if p["entity_id"] not in self.checked][: self.cap]
        if not todo:
            return []
        with ThreadPoolExecutor(self.workers) as ex:
            res = list(ex.map(self._check, todo))
        out = []
        for e, lab in zip(todo, res):
            self.checked.add(e)
            if lab is not None:
                out.append((e, lab, "http-liveness"))
        return out
