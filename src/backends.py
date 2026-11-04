"""Swappable classifiers: text -> a distribution over the deployed classes.

The DEPLOYED encoder verdict is three classes: debug/fix_input/fix_call.
CLASSES_3 is the live contract -- validate(), uniform fallback, and every
backend that feeds the hook enforce it. The fourth Glimmer verdict
(retry_wait) lives only in janus3 + the Glimmer trainer; it must NEVER pass
this gate, or the 3-way deployed head would receive a distribution it cannot
produce. A 4-key dist here is a caller bug, not an upgrade.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import janus_interface as _JI
from encoder_backend import JanusBackend
from janus_backend import (JANUS_INSTRUCTIONS, JANUS_URL, KEYCHAIN_TS, JanusTeacherBackend,
                         _typesafe_key)

CLASSES = _JI.CLASSES_3
CRITERIA = {c: _JI.CRITERIA[c] for c in _JI.CLASSES_3}


def validate(dist, name):
    """Reject anything that is not a usable distribution over CLASSES."""
    if not isinstance(dist, dict):
        raise TypeError(f"{name}: expected dict, got {type(dist).__name__}")
    if set(dist) != set(CLASSES):
        missing = set(CLASSES) - set(dist)
        extra = set(dist) - set(CLASSES)
        raise ValueError(f"{name}: key mismatch missing={missing} extra={extra}")
    for k, v in dist.items():
        if not isinstance(v, (int, float)) or v != v:
            raise ValueError(f"{name}: {k} is not a number: {v!r}")
        if not -1e-9 <= v <= 1 + 1e-9:
            raise ValueError(f"{name}: {k}={v} is outside [0,1]")
    total = sum(dist.values())
    if abs(total - 1.0) > 0.05:
        raise ValueError(f"{name}: probabilities sum to {total:.3f}, not 1")
    return {k: float(v) for k, v in dist.items()}


def _uniform():
    return {c: 1 / len(CLASSES) for c in CLASSES}


class LocalBackend:
    """Llama + optional LoRA, letter-code read-out (legacy comparison)."""

    name = "local"

    def __init__(self, base=None, adapter=None):
        from janus_route import LocalTier
        self._t = LocalTier(base or os.environ.get("JANUS_BASE_MODEL", "unsloth/Llama-3.2-1B"),
                            adapter or os.environ.get("JANUS_ADAPTER"))

    def __call__(self, text):
        return validate(self._t(text), "local")


class FrontierBackend:
    """A declared flat distribution around a label, not calibrated probabilities."""

    name = "frontier"

    def __init__(self, effort=None, mass=0.7):
        self.effort = effort or os.environ.get("JANUS_EFFORT", "low")
        self.mass = mass

    def __call__(self, text):
        from janus_route import escalate
        lab, raw, _usage = escalate(text, effort=self.effort)
        if lab not in CLASSES:
            return _uniform()
        rest = (1.0 - self.mass) / (len(CLASSES) - 1)
        d = {c: rest for c in CLASSES}
        d[lab] = self.mass
        return validate(d, "frontier")


# Ordered shallow rules are the null, not a competitor disguised as one.
KEYWORDS = (
    ("fix_input", ("no such file", "not found", "enoent", "does not exist",
                   "beyond end of file", "path not found", "oldtext not found",
                   "no such string", "string not found", "file has not been read")),
    ("fix_call", ("invalid argument", "unexpected argument", "must be of type",
                  "could not find edits", "must be unique", "overlap",
                  "missing required", "required parameter", "must be a string",
                  "unknown parameter", "not a valid", "invalid params")),
    ("debug", ("traceback", "exception", "syntaxerror", "typeerror",
               "assertionerror", "error ts", "failed with exit code",
               "test failed", "tests failed", "compilation error", "linter error")),
)


class KeywordBackend:
    name = "keyword"

    def __init__(self, peak=0.8):
        self.peak = peak

    def __call__(self, text):
        low = text.lower()
        for cls, keys in KEYWORDS:
            if any(k in low for k in keys):
                rest = (1.0 - self.peak) / (len(CLASSES) - 1)
                d = {c: rest for c in CLASSES}
                d[cls] = self.peak
                return validate(d, "keyword")
        return _uniform()


class GlimmerBackend:
    """Muse Glimmer 30B tier: declared route, deployed separately.

    No checkpoint ships in-repo, so without an endpoint this raises instead
    of faking a distribution. Set JANUS_GLIMMER_ENDPOINT (Baseten deployment
    URL, see docs/BASETEN.md) to serve it. Swappable by contract."""

    name = "glimmer"

    def __init__(self, endpoint=None):
        self.endpoint = (endpoint or os.environ.get("JANUS_GLIMMER_ENDPOINT", "")).strip()

    def __call__(self, text):
        if not self.endpoint:
            raise RuntimeError("glimmer tier not deployed: set JANUS_GLIMMER_ENDPOINT")
        return self._fetch(text)

    def _fetch(self, text):
        """POST encoder input to the Baseten endpoint, validate the reply."""
        import json
        import urllib.request
        body = json.dumps({"text": text}).encode()
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("JANUS_GLIMMER_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Api-Key {api_key}"
        request = urllib.request.Request(self.endpoint, data=body,
                                         headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=60) as reply:
                payload = json.loads(reply.read().decode())
        except Exception as error:
            raise RuntimeError(f"glimmer tier unreachable: {type(error).__name__}") from error
        if not isinstance(payload, dict) or "dist" not in payload:
            raise RuntimeError("glimmer tier malformed reply: no dist")
        return validate(payload["dist"], "glimmer")


def encoder_transport():
    """How `encoder` is reached. Default local: existing callers pass weights.

    The installed Muse hook asks for `remote` explicitly, so only the hook ever
    depends on the cluster.
    """
    return (os.environ.get("JANUS_ENCODER_TRANSPORT") or "local").strip().lower()


def get_backend(name=None, **kw):
    """Resolve a backend by name, or from JANUS_BACKEND (default: teacher).

    `teacher` is the hosted teacher that produced the labels; `janus` (alias
    `encoder`) is the deployed local encoder. The default stays `teacher` so
    labeling runs never silently swap to the student."""
    name = (name or os.environ.get("JANUS_BACKEND") or "teacher").strip().lower()
    table = {"teacher": JanusTeacherBackend, "local": LocalBackend, "encoder": JanusBackend,
             "janus": JanusBackend, "glimmer": GlimmerBackend,
             "frontier": FrontierBackend, "keyword": KeywordBackend}
    if name not in table:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(table)}")
    if name in ("encoder", "janus") and encoder_transport() != "local":
        from remote_slurm import RemoteEncoder
        return RemoteEncoder(**kw)
    return table[name](**kw)
