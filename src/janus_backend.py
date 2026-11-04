"""Live Janus transport; request shape retained for existing teacher labels."""

import json
import os
import subprocess

import janus_interface as JI

JANUS_URL = "https://api.typesafe.ai/v1/systemone"
KEYCHAIN_TS = "pi-secrets:typesafe-api-key"
JANUS_INSTRUCTIONS = [
    "You are diagnosing why a tool call made by a software agent failed, and what "
    "the agent should do next.",
    "Choose the single response class whose criterion best matches the failure.",
    "Judge only the failure and the call that produced it. Do not assume what the "
    "agent intended beyond the given text.",
    "If the failure looks like a slow environment but also shows a real error, "
    "prefer the class that describes the actual error.",
]


def _typesafe_key():
    env = os.environ.get("TYPESAFE_API_KEY")
    if env:
        return env.strip()
    return subprocess.run(
        ["security", "find-generic-password", "-s", KEYCHAIN_TS, "-w"],
        capture_output=True, text=True, check=True).stdout.strip()


def request_body(text, model):
    # The teacher was queried on the 3-class contract only. Sending the 4th
    # Glimmer verdict would change the question mid-corpus and invalidate every
    # stored teacher distribution, so teacher requests stay on CLASSES_3.
    criteria = {c: JI.CRITERIA[c] for c in JI.CLASSES_3}
    return {"model": model, "state": {"failure": text[:6000], "tool": None},
            "questions": {"response": {"type": "choice", "criteria": criteria,
                                      "instructions": {"rules": JANUS_INSTRUCTIONS}}}}


def request(body, key, timeout):
    import urllib.error
    import urllib.request
    from backends import validate

    req = urllib.request.Request(
        JANUS_URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Janus HTTP {error.code}: {error.read().decode()[:300]}") from None
    probs = ((data.get("answers") or {}).get("response") or {}).get("probabilities") or {}
    if not probs:
        raise RuntimeError(f"Janus returned no probabilities: {json.dumps(data)[:300]}")
    return validate(probs, "teacher")


class JanusTeacherBackend:
    name = "teacher"

    def __init__(self, model=None, timeout=60):
        self.model = model or os.environ.get("TYPESAFE_MODEL", "jev-latest")
        self.timeout = timeout
        self._key = None

    def _body(self, text):
        return request_body(text, self.model)

    def __call__(self, text):
        if self._key is None:
            self._key = _typesafe_key()
        return request(self._body(text), self._key, self.timeout)
