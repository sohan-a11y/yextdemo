"""The contract between the teacher (an LLM or a person) and the Forge: a JSON spec, validated before anything is trained.

A spec describes a threat as CHANGES TO THE MONITOR'S FEATURE COUNTS in one window ("20 more renamed files, 5 more extension changes"), because
that is all a neuron can see. Everything else is text for humans, and is shown escaped.
"""
import re

from ..features import FEATURES, DESCRIPTIONS, BOUNDS

MODES = ("add", "multiply", "set_min")
SEVERITIES = ("low", "medium", "high")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,39}$")
BAD_TEXT = re.compile(r"(?i)(https?://|www\.|`|<\s*script|<\s*iframe|javascript:|\bpowershell\b.*-enc|rm\s+-rf|del\s+/)")

EFFECT_SCHEMA = {"type": "object", "properties": {
    "feature": {"type": "string", "enum": FEATURES},
    "mode": {"type": "string", "enum": list(MODES)},
    "value": {"type": "number"},
    "jitter": {"type": "number"}}, "required": ["feature", "mode", "value", "jitter"], "additionalProperties": False}
SCENARIO_SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"}, "description": {"type": "string"},
    "effects": {"type": "array", "items": EFFECT_SCHEMA}}, "required": ["name", "description", "effects"], "additionalProperties": False}
SPEC_SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"}, "summary": {"type": "string"},
    "scenarios": {"type": "array", "items": SCENARIO_SCHEMA},
    "lookalikes": {"type": "array", "items": SCENARIO_SCHEMA},
    "alert": {"type": "object", "properties": {"severity": {"type": "string", "enum": list(SEVERITIES)}, "title": {"type": "string"},
                                               "explanation": {"type": "string"}}, "required": ["severity", "title", "explanation"], "additionalProperties": False}},
    "required": ["name", "summary", "scenarios", "lookalikes", "alert"], "additionalProperties": False}

LIMITS = {"scenarios": (2, 6), "lookalikes": (1, 5), "effects": (1, 8), "text": 300}


class SpecError(ValueError):
    pass


def _text(s, field, problems, limit=None, required=True):
    limit = limit or LIMITS["text"]
    if not isinstance(s, str):
        problems.append(f"{field}: must be text")
        return ""
    s = "".join(ch for ch in s if ch.isprintable()).strip()
    if required and not s:
        problems.append(f"{field}: empty")
    if len(s) > limit:
        problems.append(f"{field}: longer than {limit} characters")
        s = s[:limit]
    if BAD_TEXT.search(s):
        problems.append(f"{field}: contains a link, code or a command (text from a model is shown to people and must not carry instructions)")
    return s


def _effect(e, field, problems):
    if not isinstance(e, dict):
        problems.append(f"{field}: must be an object")
        return None
    f, mode = e.get("feature"), e.get("mode")
    if f not in FEATURES:
        problems.append(f"{field}: unknown feature {f!r}")
        return None
    if mode not in MODES:
        problems.append(f"{field}: unknown mode {mode!r}")
        return None
    try:
        v, j = float(e.get("value")), float(e.get("jitter", 0.2))
    except (TypeError, ValueError):
        problems.append(f"{field}: value/jitter must be numbers")
        return None
    if not (v == v and j == j) or abs(v) == float("inf"):
        problems.append(f"{field}: value must be finite")
        return None
    if mode == "multiply" and not (1.0 <= v <= 1000.0):
        problems.append(f"{field}: multiply factor {v} outside 1..1000")
        return None
    if mode in ("add", "set_min") and not (0.0 <= v <= BOUNDS[f]):
        problems.append(f"{field}: {f} value {v} outside 0..{BOUNDS[f]:g}")
        return None
    return {"feature": f, "mode": mode, "value": v, "jitter": min(max(j, 0.0), 1.0)}


def _scenario(s, field, problems):
    if not isinstance(s, dict):
        problems.append(f"{field}: must be an object")
        return None
    name = _text(s.get("name"), f"{field}.name", problems, 60)
    desc = _text(s.get("description"), f"{field}.description", problems)
    eff = s.get("effects")
    if not isinstance(eff, list) or not (LIMITS["effects"][0] <= len(eff) <= LIMITS["effects"][1]):
        problems.append(f"{field}.effects: need {LIMITS['effects'][0]}..{LIMITS['effects'][1]} effects")
        return None
    effects = [x for i, e in enumerate(eff) if (x := _effect(e, f"{field}.effects[{i}]", problems))]
    if len(effects) != len(eff):
        return None
    return {"name": name, "description": desc, "effects": effects}


def validate(spec):
    """Return a clean spec or raise SpecError listing every problem."""
    problems = []
    if not isinstance(spec, dict):
        raise SpecError("spec must be a JSON object")
    name = spec.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        problems.append("name: lowercase letters, digits and underscores, 3-40 characters, starting with a letter")
    out = {"name": name, "summary": _text(spec.get("summary"), "summary", problems)}
    for key, (lo, hi) in (("scenarios", LIMITS["scenarios"]), ("lookalikes", LIMITS["lookalikes"])):
        v = spec.get(key)
        if not isinstance(v, list) or not (lo <= len(v) <= hi):
            problems.append(f"{key}: need {lo}..{hi} items")
            out[key] = []
            continue
        out[key] = [x for i, s in enumerate(v) if (x := _scenario(s, f"{key}[{i}]", problems))]
    al = spec.get("alert")
    if not isinstance(al, dict) or al.get("severity") not in SEVERITIES:
        problems.append("alert.severity: one of " + ", ".join(SEVERITIES))
        al = al if isinstance(al, dict) else {}
    out["alert"] = {"severity": al.get("severity", "medium"), "title": _text(al.get("title"), "alert.title", problems, 100),
                    "explanation": _text(al.get("explanation"), "alert.explanation", problems)}
    names = [s["name"] for s in out["scenarios"] + out["lookalikes"]]
    if len(set(names)) != len(names):
        problems.append("scenario names must be distinct")
    if problems:
        raise SpecError("; ".join(problems))
    return out


def feature_catalogue():
    return [{"feature": f, "meaning": DESCRIPTIONS[f], "plausible_max": BOUNDS[f]} for f in FEATURES]
