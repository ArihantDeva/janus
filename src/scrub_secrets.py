"""Redact credential material from a dataset file IN PLACE.

Why this exists rather than being folded into the relabeller: the v1 files are
already committed to git and contain a live-looking OpenAI key (`sk-aBikx...`,
67 chars) plus Anthropic OAuth token heads (`sk-ant-oat01-...`) and presigned
S3 URLs. Those came from real session logs where an agent printed a key. They
must not sit in a repo, in history, or in any artifact we hand a judge.

The pattern set is deliberately over-broad: a false positive costs one unreadable
row, a false negative leaks a credential.
"""

from __future__ import annotations

import json
import re
import sys

PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "<redacted-openai-key>"),
    (re.compile(r"sk-ant-[a-z0-9]+-[A-Za-z0-9_-]{16,}"), "<redacted-anthropic-token>"),
    (re.compile(r"gh[opsu]_[A-Za-z0-9]{30,}"), "<redacted-github-token>"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "<redacted-github-pat>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<redacted-aws-key-id>"),
    (re.compile(r"LLM_\d+_[A-Za-z0-9]{10,}"), "<redacted-meta-key>"),
    (re.compile(r"xox[abpsr]-[A-Za-z0-9-]{10,}"), "<redacted-slack-token>"),
    (re.compile(r"AIza[0-9A-Za-z_-]{35}"), "<redacted-google-key>"),
    (re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
     "<redacted-jwt>"),
    # presigned URLs: the credential is in the query string
    (re.compile(r"X-Amz-Credential=[^&\s\"']+"), "X-Amz-Credential=<redacted>"),
    (re.compile(r"X-Amz-Signature=[^&\s\"']+"), "X-Amz-Signature=<redacted>"),
    (re.compile(r"(?i)(?:password|passwd|secret|api[_-]?key|token)\s*[=:]\s*"
                r"[\"']?[A-Za-z0-9_\-./+]{12,}"), None),   # None = keep the key name
]


def scrub(text):
    n = 0
    for rx, repl in PATTERNS:
        if repl is None:
            def keep_key(m):
                return re.sub(r"([=:]\s*[\"']?)\S+", r"\1<redacted>", m.group(0))
            text, k = rx.subn(keep_key, text)
        else:
            text, k = rx.subn(repl, text)
        n += k
    return text, n


def _scrub_obj(r):
    hits = 0
    for k in ("x", "body", "text"):
        if isinstance(r.get(k), str):
            new, n = scrub(r[k])
            r[k] = new
            hits += n
    return hits


def main(paths):
    total = 0
    for p in paths:
        hits = 0
        raw = open(p).read()
        stripped = raw.lstrip()
        if stripped.startswith("["):
            # a JSON ARRAY file (demo cases), not JSONL -- parse it as a whole
            data = json.loads(raw)
            rows = data if isinstance(data, list) else [data]
            hits = sum(_scrub_obj(r) for r in rows)
            if hits:
                open(p, "w").write(json.dumps(data, indent=2) + "\n")
            print(f"{p}: {hits} redactions in {len(rows)} array items")
        else:
            rows = []
            for line in raw.splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                hits += _scrub_obj(r)
                rows.append(r)
            if hits:
                with open(p, "w") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
            print(f"{p}: {hits} redactions in {len(rows)} rows")
        total += hits
    print(f"total redactions: {total}")
    return total


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/responses/train.jsonl",
                          "data/responses/rule_held.jsonl",
                          "data/responses/unm_dev.jsonl",
                          "data/responses/unmatched_held.jsonl",
                          "data/responses/unmatched.jsonl"])
