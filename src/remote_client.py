"""One encoder request, executed INSIDE the owned allocation.

stdin carries the canonical encoder input; stdout carries exactly one JSON
object. Diagnostics stay on stderr so the transport stays parseable.
"""

from __future__ import annotations

import json
import os
import socket
import sys

import remote_service as S


def ask(sock, text, timeout):
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(timeout)
    try:
        conn.connect(sock)
        S.send_message(conn, {"text": text})
        return S.recv_message(conn)
    finally:
        conn.close()


def main():
    job = os.environ.get("JANUS_REMOTE_JOB") or ""
    if not job.isdigit():
        print("client needs JANUS_REMOTE_JOB", file=sys.stderr)
        return 2
    try:
        reply = ask(S.socket_path(job, os.environ.get("JANUS_REMOTE_RUN_DIR")),
                    sys.stdin.read(), float(os.environ.get("JANUS_REMOTE_TIMEOUT", 120)))
    except Exception as error:
        print(f"encoder request failed: {type(error).__name__}", file=sys.stderr)
        return 3
    if "distribution" not in reply:
        print(f"encoder refused request: {reply.get('error')}", file=sys.stderr)
        return 3
    sys.stdout.write(json.dumps({"distribution": reply["distribution"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
