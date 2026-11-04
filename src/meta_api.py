"""Minimal Meta Model API client.

Only what this project needs. Three measured facts are baked in because each one
cost a failed call to discover:

1. `reasoning: {effort: ...}` in the body is REJECTED ("unknown parameter"). The
   Chat Completions wire wants top-level `reasoning_effort`.
2. Muse Spark is a reasoning model and `"none"` is unsupported, so a small
   max_tokens returns `content: null` with `finish_reason: length` -- the whole
   budget is consumed by hidden reasoning tokens. A 20-token request produced 18
   reasoning tokens and no answer. Budgets must be generous even for one-word
   answers.
3. `usage.completion_tokens_details.reasoning_tokens` is reported, so the real
   cost of a "one word" answer is visible rather than inferred.

The API key is read from the macOS Keychain and never logged, printed, or passed
through a shell variable that could end up in output.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request

BASE = "https://api.meta.ai/v1"
KEYCHAIN_SERVICE = "ai.meta.dev.credentials"


def api_key() -> str:
    """Read the key from Keychain, or from MODEL_API_KEY if already exported."""
    env = os.environ.get("MODEL_API_KEY")
    if env:
        return env.strip()
    blob = subprocess.run(
        ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
        capture_output=True, text=True, check=True).stdout
    # the item stores a small JSON object, not the bare key
    return json.loads(blob)["api_key"].strip()


def chat(messages, model="muse-spark-1.3", effort="low", max_tokens=1200,
         temperature=0.0, timeout=120, key=None):
    """One chat completion. Returns content plus usage, never a bare string.

    Returning usage is deliberate: the escalation tier is only defensible if its
    cost is measured, and reasoning tokens are most of that cost.
    """
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if effort:
        body["reasoning_effort"] = effort      # top-level, NOT nested

    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key or api_key()}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}") from None

    choice = d["choices"][0]
    return {
        "content": choice["message"].get("content") or "",
        "finish": choice.get("finish_reason"),
        "usage": d.get("usage", {}),
        "model": d.get("model"),
    }


def reasoning_tokens(res) -> int:
    det = res.get("usage", {}).get("completion_tokens_details") or {}
    return int(det.get("reasoning_tokens") or 0)


def total_tokens(res) -> int:
    return int(res.get("usage", {}).get("total_tokens") or 0)
