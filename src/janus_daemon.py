"""Warm Janus daemon: one process, no per-request spin-up.

Listens on a unix socket. The backend (torch stack + 1B encoder) is built
exactly once in serve(); every request replays the same hook entry with it.
If the daemon is absent, deaf, or refusing, the hook entry exits 3 and the
caller falls back to the direct path.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_BYTES = 1 << 20


def sock_path():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "janus-daemon", "hook.sock")


def send(conn, payload):
    body = json.dumps(payload).encode()
    conn.sendall(struct.pack("!I", len(body)) + body)


def recv(conn, count):
    chunks, seen = [], 0
    while seen < count:
        chunk = conn.recv(count - seen)
        if not chunk:
            raise ConnectionError("peer closed")
        chunks.append(chunk)
        seen += len(chunk)
    return b"".join(chunks)


def read_message(conn):
    (length,) = struct.unpack("!I", recv(conn, 4))
    if not 0 < length <= MAX_BYTES:
        raise ValueError("message size out of range")
    return json.loads(recv(conn, length))


def query(payload, timeout=None):
    """Ask the warm daemon with a decoded object; raise if not answering."""
    return query_text(json.dumps(payload), timeout)


def query_text(data, timeout=None):
    """Ask the warm daemon with one hook payload as JSON text."""
    if timeout is None:
        timeout = float(os.environ.get("JANUS_DAEMON_TIMEOUT", "30"))
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(timeout)
    try:
        conn.connect(sock_path())
        json.loads(data)  # wire carries the payload itself; validate first
        body = data.encode()
        if len(body) > MAX_BYTES:
            raise ValueError("message size out of range")
        conn.sendall(struct.pack("!I", len(body)) + body)
        return read_message(conn)
    finally:
        conn.close()


def is_compact(payload):
    """Strict compact flag: the string "0" is falsy here.

    Muse passes hook-command VAR= assignments through into the payload as
    strings, so `JANUS_COMPACT=0` arrives as "0" — which is truthy in
    Python and silently enabled compact mode on every request (measured
    2026-11-04: daemon logged compact=True with both hooks hardcoded to 0).
    """
    if not isinstance(payload, dict):
        return False
    return payload.get("janus_compact") in (True, 1, "1", "true", "True")


def handle(conn, backend_factory):
    """One request. Backend failure replies not-ok; a dead client just ends."""
    payload = read_message(conn)
    log = os.environ.get("JANUS_DAEMON_LOG")

    def trace(msg):
        line = f"[janus-daemon] {msg}"
        print(line, file=sys.stderr, flush=True)
        if log:
            with open(log, "a") as fh:
                fh.write(line + "\n")

    try:
        backend = backend_factory()
        import muse_hook as H

        _, dist, shown = H.classify(payload, lambda: backend)
        threshold = H.threshold()
        best, conf = (max(dist, key=dist.get), dist[max(dist, key=dist.get)]) if dist else (None, None)
        keys = sorted(payload.keys()) if isinstance(payload, dict) else []
        trace(f"keys={keys}")
        compact = is_compact(payload)
        trace(f"tool={payload.get('tool_name') if isinstance(payload, dict) else '?'} "
              f"best={best} conf={conf:.3f} thr={threshold:.2f} "
              f"compact={bool(compact)} backend={backend.name}")
        if compact:
            context = H.inline_verdict(backend, dist, threshold)
            reply = {"ok": True,
                     "stdout": H._advisory_json(context)}
        else:
            context = H.advisory(backend, dist, threshold, shown)
            inline = H.inline_verdict(backend, dist, threshold)
            out = {"hookSpecificOutput": {
                "hookEventName": "PostToolUseFailure",
                "additionalContext": context,
                "systemMessage": inline}}
            reply = {"ok": True, "stdout": json.dumps(out)}
        trace(f"dual emit: systemMessage={bool(not compact)}")
    except Exception as error:
        print(f"[janus-daemon] ERROR {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
        reply = {"ok": False, "error": type(error).__name__}
    with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
        send(conn, reply)


def serve(backend_factory=None):
    """Build the backend once, then answer forever, surviving disconnects."""
    if backend_factory is None:
        from backends import get_backend

        backend_factory = get_backend
    backend = backend_factory()  # warm: paid once here, never per request

    path = sock_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    os.chmod(path, 0o600)
    server.listen(8)
    with open(path + ".ready", "w") as ready:
        ready.write(str(os.getpid()))
    print(f"janus daemon ready on {path}", flush=True)
    try:
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    handle(conn, lambda: backend)
                except (ConnectionError, ValueError, OSError,
                        json.JSONDecodeError):
                    continue  # client went away mid-message; keep serving
    finally:
        server.close()


def main(argv=None):
    argv = argv or sys.argv[1:]
    if argv[:1] == ["serve"]:
        serve()
        return 0
    payload = json.load(sys.stdin)
    try:
        reply = query(payload)
    except (FileNotFoundError, ConnectionRefusedError, socket.timeout,
            ConnectionError, ValueError, OSError):
        sys.exit(3)
    if not reply.get("ok"):
        sys.exit(3)  # cold path rebuilds the backend; never strand the turn
    sys.stdout.write(reply.get("stdout", ""))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
