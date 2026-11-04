"""Warm encoder service inside one owned Slurm allocation.

Loads the checkpoint once and answers canonical encoder inputs over a unix
socket on the node's own filesystem. The login node never talks to this socket
directly: `scripts/client.py` runs inside the allocation and does.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys

MAX_REQUEST_BYTES = 1 << 20


def socket_path(job, run_dir=None):
    base = run_dir or os.environ.get("JANUS_REMOTE_RUN_DIR", "run")
    return os.path.join(base, f"encoder-{job}.sock")


def ready_path(job, run_dir=None):
    return f"{socket_path(job, run_dir)}.ready"


def recv_message(conn):
    header = _recv_exact(conn, 4)
    length = struct.unpack("!I", header)[0]
    if not 0 < length <= MAX_REQUEST_BYTES:
        raise ValueError("message size out of range")
    return json.loads(_recv_exact(conn, length))


def send_message(conn, payload):
    body = json.dumps(payload).encode()
    conn.sendall(struct.pack("!I", len(body)) + body)


def _recv_exact(conn, count):
    chunks, seen = [], 0
    while seen < count:
        chunk = conn.recv(count - seen)
        if not chunk:
            raise ConnectionError("peer closed early")
        chunks.append(chunk)
        seen += len(chunk)
    return b"".join(chunks)


def load_classifier():
    """Construct the real Janus backend from its existing contract."""
    from backends import get_backend

    os.environ["JANUS_ENCODER_TRANSPORT"] = "local"
    return get_backend("janus")


def answer(backend, request):
    """Distribution for one canonical input; `describe` reports what is serving."""
    if request.get("describe"):
        return describe(backend)
    text = request.get("text")
    if not isinstance(text, str) or not text.strip():
        return {"error": "empty request"}
    return {"distribution": backend(text)}


def describe(backend):
    """What this allocation is actually serving: checkpoint, head, temperature."""
    temperature = getattr(backend, "temperature", None)
    return {"serving": True,
            "checkpoint": os.environ.get("JANUS_ENCODER"),
            "head": os.environ.get("JANUS_ENCODER_HEAD"),
            "temperature": temperature,
            "calibrated": temperature is not None}


def serve(backend, job, run_dir=None, rounds=None):
    path = socket_path(job, run_dir)
    for leftover in (path, ready_path(job, run_dir)):
        if os.path.exists(leftover):
            os.unlink(leftover)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    os.chmod(path, 0o600)
    server.listen(8)
    with open(ready_path(job, run_dir), "w") as handle:
        handle.write(str(os.getpid()))
    try:
        handled = 0
        while rounds is None or handled < rounds:
            conn, _ = server.accept()
            with conn:
                conn.settimeout(120)
                try:
                    send_message(conn, answer(backend, recv_message(conn)))
                except Exception as error:
                    send_message(conn, {"error": type(error).__name__})
            handled += 1
    finally:
        server.close()
        for leftover in (path, ready_path(job, run_dir)):
            if os.path.exists(leftover):
                os.unlink(leftover)


def main(argv=None):
    argv = argv or sys.argv[1:]
    job = os.environ.get("SLURM_JOB_ID") or (argv[0] if argv else "")
    if not job.isdigit():
        print("service needs an allocation id", file=sys.stderr)
        return 2
    serve(load_classifier(), job, os.environ.get("JANUS_REMOTE_RUN_DIR"), None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
