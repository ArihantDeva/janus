"""Relabel transient-environment failures to retry_wait.

The teacher labelled every row into three classes, so genuine transients
(timeouts, 429s, refused connections, cold starts) landed in debug/fix_input/
fix_call. Retrying those unchanged-vs-debugging is exactly the judgment the
fourth verdict exists to make, so the rows are relabelled by rule, not by
teacher re-query.

RULE (conservative, head-of-error only):
  the error's FIRST 300 chars match TIMEOUT | RATE_LIMIT | CONN_FAIL, where
  TIMEOUT    = timeout|timed out|deadline exceeded
  RATE_LIMIT = rate.?limit|429|too many requests|backoff|retry.*later
  CONN_FAIL  = econnreset|econnrefused|socket hang up|operation timed out
               (ssh/connect-flavoured only -- see guard below)
  and the error does NOT match RUINERS: traceback|diff --git|sed:|No such file.

WHY the guard: mid-error mentions ("caches transient empty results", a
traceback line quoting a timeout inside user code) are content ABOUT the
words, not an environment that was slow. The verdict is about what the agent
should do, and "Command timed out after 120 seconds" as the whole error
means wait-and-retry while a 4000-char log that merely contains "timeout"
does not.

The old teacher argmax is kept in `teacher_argmax_prev` so teacher-vs-rule
disagreement on the fourth class stays measurable.
"""

from __future__ import annotations

import json
import re
import sys

HEAD = 300

TIMEOUT = re.compile(r"time.?out|timed out|deadline exceeded", re.I)
RATE_LIMIT = re.compile(r"rate.?limit|\b429\b|too many requests|backoff|retry.*later", re.I)
CONN_FAIL = re.compile(
    r"econnreset|econnrefused|socket hang up|operation timed out"
    r"|connect to host.*timed out|connection refused|network is unreachable",
    re.I)
RUINERS = re.compile(r"traceback|diff --git|\bno such file\b|command not found", re.I)


def is_transient(error: str) -> bool:
    head = (error or "")[:HEAD]
    if RUINERS.search(head):
        return False
    return bool(TIMEOUT.search(head) or RATE_LIMIT.search(head)
                or CONN_FAIL.search(head))


def relabel(rows):
    n = 0
    for r in rows:
        if is_transient(r.get("error") or ""):
            if r.get("janus_label", r.get("teacher_argmax")) != "retry_wait":
                r["teacher_argmax_prev"] = r.get("teacher_argmax")
                r["janus_label"] = "retry_wait"
                n += 1
    return n


def main(src, dst):
    rows = [json.loads(l) for l in open(src) if l.strip()]
    n = relabel(rows)
    with open(dst, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"relabelled {n}/{len(rows)} -> retry_wait")
    print("labels:", dict(Counter(
        r.get("janus_label", r.get("teacher_argmax")) for r in rows)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
