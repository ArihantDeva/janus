"""Ask the owned allocation what it serves: checkpoint, head and calibration."""

from __future__ import annotations

import json
import os
import socket
import sys

import remote_service as S


def main():
    job = os.environ.get("JANUS_REMOTE_JOB") or ""
    if not job.isdigit():
        print("describe needs JANUS_REMOTE_JOB", file=sys.stderr)
        return 2
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(30)
    try:
        conn.connect(S.socket_path(job, os.environ.get("JANUS_REMOTE_RUN_DIR")))
        S.send_message(conn, {"describe": True})
        print(json.dumps(S.recv_message(conn)))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
