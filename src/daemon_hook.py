"""Janus hook entry: warm daemon first, direct path on miss.

The daemon keeps the torch stack loaded. If it is not answering, this falls
back to the direct invocation through the repo's uv environment
(mise exec -- uv run), because system python has no torch. The payload is
read once and replayed, so stdout stays exactly the hook JSON. Never breaks
the tool turn.
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import janus_daemon as D  # noqa: E402


def cold(data, timeout=280):
    """Run the direct hook with the repo's full environment, replaying data."""
    here = os.path.dirname(os.path.abspath(__file__))
    command = ["mise", "exec", "--", "uv", "run", "--with", "numpy",
               "--with", "torch", "--with", "transformers", "--with", "peft",
               "--with", "safetensors", "python",
               os.path.join(here, "muse_hook.py")]
    try:
        done = subprocess.run(command, input=data, text=True,
                              capture_output=True, timeout=timeout)
    except Exception as error:
        print(f"Tool failure classifier unavailable: {type(error).__name__}",
              file=sys.stderr)
        return 0
    sys.stdout.write(done.stdout)
    sys.stdout.flush()
    sys.stderr.write(done.stderr)
    return 0


def main():
    import muse_hook as H

    if os.environ.get("JANUS_DAEMON_OFF"):
        return H.main()
    data = sys.stdin.read()
    import json as _json

    if os.environ.get("JANUS_COMPACT") not in (None, "", "0"):
        print(f"[daemon-hook] injecting janus_compact (env JANUS_COMPACT={os.environ.get('JANUS_COMPACT')!r})",
              file=sys.stderr, flush=True)
        try:
            payload = _json.loads(data)
            if isinstance(payload, dict):
                payload["janus_compact"] = True
                data = _json.dumps(payload)
        except Exception:
            pass
    try:
        reply = D.query_text(data)
    except (FileNotFoundError, ConnectionRefusedError, TimeoutError,
            ConnectionError, ValueError, OSError, _json.JSONDecodeError):
        return cold(data)
    if not reply.get("ok"):
        return cold(data)
    sys.stdout.write(reply.get("stdout", ""))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
