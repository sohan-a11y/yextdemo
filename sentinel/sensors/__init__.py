"""Sensors: read-only eyes. Each one turns what the operating system shows into small event dicts. None of them changes anything."""
import time


class Sensor:
    name = "sensor"
    available = True
    reason = ""

    def poll(self, now=None):
        """Return the events seen since the last call."""
        return []

    def state(self):
        return {}

    def load(self, state):
        pass

    def close(self):
        pass

    def info(self):
        return {"name": self.name, "available": self.available, "reason": self.reason}
