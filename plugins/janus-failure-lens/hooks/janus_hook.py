"""Janus hook entry: warm daemon first, direct path on miss.

The daemon keeps the torch stack loaded. If it is not answering, the
daemon_hook falls back to the direct invocation through the repo's uv
environment. Never breaks the tool turn.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO_SRC = os.environ.get("JANUS_SRC_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if os.path.isdir(_REPO_SRC):
    sys.path.insert(0, _REPO_SRC)

import daemon_hook as DH


def main():
    # Muse plugin hooks receive stdin as a JSON-encoded string, not a raw
    # object; unwrap once so daemon_hook parses a dict (2026-11-04).
    import io
    import json as _json
    _data = sys.stdin.read()
    try:
        _p = _json.loads(_data or "{}")
        if isinstance(_p, str):
            _p = _json.loads(_p or "{}")
            if isinstance(_p, dict):
                _data = _json.dumps(_p)
    except Exception:
        pass
    sys.stdin = io.StringIO(_data)
    try:
        return DH.main()
    except Exception as error:
        print(f"Tool failure classifier unavailable: {type(error).__name__}",
              file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
