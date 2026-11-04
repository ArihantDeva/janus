"""Janus Muse PostToolUseFailure hook (npm-bundled, standalone-safe).

Order: (1) warm repo daemon via JANUS_SRC_DIR / JANUS_PKG_ROOT/src / repo src;
(2) keyword backend (pure python, offline); (3) exit 0 silently.
Never breaks the tool turn; diagnostics go to stderr only.
"""
from __future__ import annotations

import json
import os
import sys

CRITERIA = {
    "debug": "Read the error and change the code.",
    "fix_input": "Correct the path/handle. The input refers to something absent.",
    "fix_call": "Fix the call shape: arguments are missing, unknown, or mistyped.",
}
KEYWORDS = (
    ("fix_input", ("no such file", "not found", "enoent", "does not exist",
                   "path not found", "no such string", "file has not been read")),
    ("fix_call", ("invalid argument", "unexpected argument", "must be of type",
                  "missing required", "required parameter", "unknown parameter",
                  "not a valid", "invalid params")),
    ("debug", ("traceback", "exception", "syntaxerror", "typeerror",
               "assertionerror", "compilation error", "test failed")),
)


def keyword_verdict(text):
    low = text.lower()
    for cls, keys in KEYWORDS:
        if any(k in low for k in keys):
            return cls, 0.80
    return None, 0.0


def emit(cls, conf, backend):
    msg = f"[janus] {cls} ({conf:.2f}) -- {CRITERIA[cls]}"
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PostToolUseFailure",
        "additionalContext": msg}}))
    return 0


def main():
    data = sys.stdin.read()
    try:
        payload = json.loads(data or "{}")
        if isinstance(payload, str):  # muse plugin hooks get stringified stdin
            payload = json.loads(payload or "{}")
            if isinstance(payload, dict):
                data = json.dumps(payload)  # daemon path wants the raw object
    except Exception:
        payload = {}
    cands = [os.environ.get("JANUS_SRC_DIR", ""),
             os.path.join(os.environ.get("JANUS_PKG_ROOT", ""), "src"),
             os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                 os.path.abspath(__file__)))), "src")]
    for src in cands:
        if src and os.path.isfile(os.path.join(src, "daemon_hook.py")):
            sys.path.insert(0, src)
            try:
                import daemon_hook as DH
                sys.stdin = open("/dev/null")  # payload already consumed
                import io
                sys.stdin = io.StringIO(data)
                return DH.main()
            except Exception as error:  # fall through to keyword
                print(f"janus: daemon path unavailable: {type(error).__name__}",
                      file=sys.stderr)
            break
    if not isinstance(payload, dict):
        payload = {}
    text = " ".join(str(payload.get(k) or "") for k in
                    ("tool_name", "error")) + " " + json.dumps(
                        payload.get("tool_input") or {}, default=str)
    cls, conf = keyword_verdict(text)
    if cls is None:
        return 0
    return emit(cls, conf, "keyword")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"janus hook unavailable: {type(error).__name__}", file=sys.stderr)
        sys.exit(0)
