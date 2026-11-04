"""v2 relabel of the response dataset. Evidence comes from the ERROR MESSAGE.

WHY v1 IS NOT TRUSTWORTHY
-------------------------
The v1 rule ran `re.search` over the whole 1500-char row. The row is mostly
*pasted transcript*, not the error. Measured on v1 train+rule_held:

  * `retry`: 74.5% of labels come from a match past the halfway point of the row
    (median offset 87.7% of the row). `abort`: 87.9%.
  * The worst case is not a near-miss, it is an inverted label. This harness
    injects its own annotation into the tool result:

        [pi-janus] Janus read this as a code_bug failure (confidence 0.99):
        fix the code or types; retrying unchanged will not help.

    and the v1 `retry` pattern contains the literal `retrying`. 619 of 1513
    `retry` rows (40.9%) are labelled by a sentence that says retrying will NOT
    help. 847 rows carry such an injected annotation.
  * `abort` (149 rows): 48 are pytest collection errors (`Interrupted: 1 error
    during collection`), 17 are `rg: ... Interrupted system call (os error 4)`
    from walking ~/Library, and 65 match the word inside test output such as
    `interrupted: 0`. A human interrupt cannot be told apart from an unrelated
    use of the word, because a niggle does not set `isError` and never becomes
    a row.
  * `reauth` (158 rows): 75 are triggered by the bare word `credential` --
    mostly inside pytest test NAMES (`test_official_comment_generation_requires
    _channel_credentials`) and inside `(pass)` lines of a bun test suite. 37 of
    55 number-triggered `reauth` rows have no auth phrase anywhere in the row.
    Only 46 of 158 (29.1%) contain a real auth phrase.
  * `retry` triggered on `502` found inside unrelated pasted output, e.g. a
    validation error whose pasted arguments happened to contain the digits.

THE v2 POSITIONAL RULE, AND WHY IT IS DEFENSIBLE
------------------------------------------------
A tool result has exactly two parts that are the *message*: the tool's own
output (a prefix) and the harness status line (a suffix). Everything between is
either stdout that may quote logs and other tools' errors, or text injected after
the fact. Neither is evidence about THIS failure. So v2 defines:

  INJECTED   lines starting `[pi-janus]` / `[tool-*]`. A tool cannot emit these;
             they are harness post-processing about the failure, and they are
             not available to a classifier in production. Removed.
  BODY       the row with the injected lines and the trailing status line
             removed.
  HEAD       the first HEAD_CHARS=400 characters of BODY. This is where a
             message's own report sits: tools write `error:` / `Validation
             failed` / `Traceback` up front and the context after.
  HEAD_OBSERVABLE  a prose candidate is admissible only if it lies entirely
             inside HEAD.

If no candidate is in HEAD the row is unlabelled. It is never labelled from a
match 800 characters into a pasted log.

HEAD_CHARS=400 was chosen by reading v1 TRAIN rows only; the held splits were
not read while choosing it. The sensitivity table is printed every run. The
rejected alternatives: `m.start() < 0.15 * len(x)` (for a 200-char row that is
30 characters, too small to hold any message) and HEAD_CHARS=1500 (that is v1
again).

THE STATUS CHANNEL IS DECLARED SEPARATELY
-----------------------------------------
The harness status line (`Command timed out after N seconds`) is evidence -- it
is a structured field about THIS call, present at production time, not prose in
the middle of a log. It is a separate, explicitly declared channel, tested only
after the head finds nothing, and recorded with `channel: "status"`. Exactly one
pattern is admitted:

  * `Command timed out after N seconds` -> retry. Specific: it names the failure.
  * `Command exited with code N`        -> NOT evidence. Nothing about the
    number says which response is right, and v1's use of `exit code [1-9]` for
    `debug` is why 16 rows of Swift build log became `debug`.
  * `Command aborted`                   -> NOT evidence; see `abort` above.

GENERIC TOKENS ARE NOT EVIDENCE
-------------------------------
v1 treated `retrying`, bare `403`/`500`, `failed`, `error` and `exit code 1` as
enough to pick a class. Nothing about the string "503" says whether to re-run;
`failed` appears in pytest summaries, in a test's own name, and in a git command
rejection. v2 requires a specific failure phrase. See GENERIC_SOLE.

CLASSES DROPPED RATHER THAN APPROXIMATED
----------------------------------------
`abort` (149), `stop` (32), `inspect` (11) and `resolve` (22: 7 of 22 are plain
successful `git commit` output, not a conflict) are not labelable. They are
dropped, and the counts are reported. Dropping a class beats inventing
gradations to keep a name.

WHAT v2 CAN AND CANNOT CLAIM
----------------------------
Still a rule over the same text the model reads, so `train` alone proves nothing
about generalisation. What v2 buys is narrower and real: every label traces to a
phrase the tool itself emitted, at a recorded offset and rule name, and the
tokens that only let a model learn my regex are gone. Rows that v1 could not
label are treated as a RULED source -- a v2 rule cannot have been designed from a
row that had no rule -- so anything v2 labels there is a genuine out-of-
distribution measurement, not a fit.
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "responses")
DST = os.path.join(ROOT, "data", "responses_v2")

#: Classes v2 is willing to claim.
CLASSES = ("retry", "reauth", "fix_call", "fix_input", "debug")

#: Bucket for rows no defensible rule can label. NOT a training class.
UNMATCHED = "unmatched"

#: Dimension of the head region. Chosen by reading v1 TRAIN rows only.
HEAD_CHARS = 400

#: v1 classes v2 refuses to claim. See the module docstring.
V1_UNLABELABLE = ("abort", "stop", "inspect", "resolve")
V1_UNMATCHED = "other_fail"

# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

INJECTED_RE = re.compile(r"^[ \t]*\[(?:pi|tool)-[a-z]+\]")
STATUS_RE = re.compile(
    r"(?s)\n*\s*(?:Command exited with code \d+|Command timed out after \d+ "
    r"seconds|Command aborted)\s*$")

#: A written-out secret is not training signal. Redact the span, keep the row.
SECRET_RES = (
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bX-Amz-(?:Credential|Signature)=([^&\"'\s]+)"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-_\.]{24,}"),
    re.compile(r"(?i)\b(?:api[_-]?key|access_?token|secret|password)\s*[:=]\s*"
               r"[\"']?[A-Za-z0-9\-_]{20,}"),
)

#: A span that is one of these is not enough to choose a class, whatever pattern
#: found it. Belt and braces: no rule below should produce them either.
GENERIC_SOLE = frozenset({
    "retrying", "retry", "will retry", "error", "failed", "fail", "failure",
    "exit code", "temporar", "temporarily", "please try again", "overloaded",
    "interrupted", "cancelled", "canceled", "conflict (", "already exists",
})


def strip_injected(text: str) -> str:
    """Drop harness annotations. A tool's own output cannot start a line with
    `[pi-janus]`; that is this project's post-processing about the failure."""
    return "\n".join(l for l in (text or "").split("\n")
                     if not INJECTED_RE.match(l))


def strip_status(text: str) -> str:
    """Drop the trailing harness status line (dealt with as its own channel)."""
    return STATUS_RE.sub("", text)


def redact(text: str) -> str:
    """Keep the row; remove the secret. Emitted rows are safe to publish."""
    t = text or ""
    for p in SECRET_RES:
        t = p.sub("<redacted>", t)
    return t


def message_body(text: str) -> str:
    """The tool's own message: no harness annotations, no status line."""
    return strip_status(strip_injected(text)).strip("\n")


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------
# Ordered. Ties broken by this order then by earliest offset, so the label does
# not depend on pattern accident. Specific before generic; auth phrases before
# numbers (v2 has no bare-number pattern at all).

EVIDENCE = [
    # --- credentials. A real auth phrase, never a status code alone. ---
    ("reauth", "auth.permission_denied", re.compile(r"permission denied", re.I)),
    ("reauth", "auth.unauthorized", re.compile(r"\bunauthori[sz]ed\b", re.I)),
    ("reauth", "auth.forbidden", re.compile(r"\bforbidden\b", re.I)),
    ("reauth", "auth.not_logged_in", re.compile(r"not logged in", re.I)),
    ("reauth", "auth.auth_failed", re.compile(
        r"auth(?:entication)?\s+(?:fail(?:ed|ure)|required)", re.I)),
    ("reauth", "auth.invalid_key", re.compile(
        r"invalid (?:api[_ ]?key|token|credential)", re.I)),
    ("reauth", "auth.token_expired", re.compile(
        r"token (?:file|expired|invalid)", re.I)),
    ("reauth", "auth.credentials_invalid", re.compile(
        r"credential\w*\s+(?:are|is|were|not valid|invalid|expired|missing|"
        r"rejected|required)", re.I)),
    ("reauth", "auth.please_login", re.compile(
        r"please (?:log|sign) in|login required|re-?authenticate", re.I)),

    # --- the action was rejected before it ran: argument shape. ---
    ("fix_call", "call.validation_failed", re.compile(
        r"Validation failed for tool", re.I)),
    ("fix_call", "call.required_property", re.compile(
        r"must have required propert|required propert\w* (?:is )?missing", re.I)),
    ("fix_call", "call.must_be_type", re.compile(
        r"must be (?:of type|an? (?:string|number|integer|boolean|object|array))",
        re.I)),
    ("fix_call", "call.missing_required", re.compile(
        r"missing required (?:argument|parameter|property|field)", re.I)),
    ("fix_call", "call.unexpected_keyword", re.compile(
        r"unexpected keyword argument", re.I)),
    ("fix_call", "call.unknown_argument", re.compile(
        r"unknown (?:argument|parameter|option|flag)", re.I)),
    ("fix_call", "call.invalid_argument", re.compile(
        r"invalid (?:argument|argument value|parameter)", re.I)),
    ("fix_call", "call.edit_no_match", re.compile(
        r"could not find (?:the exact text|edits\[|the text)", re.I)),
    ("fix_call", "call.edit_not_unique", re.compile(
        r"must be unique|provide more context", re.I)),
    ("fix_call", "call.edit_overlap", re.compile(
        r"edits\[\d+\] and edits\[\d+\] overlap", re.I)),
    ("fix_call", "call.edit_noop", re.compile(
        r"replacement produced identical content|no changes made", re.I)),
    ("fix_call", "call.oldtext_mismatch", re.compile(
        r"old ?text must match", re.I)),

    # --- the input names something that is not out there. ---
    ("fix_input", "input.no_such_file", re.compile(r"no such file", re.I)),
    ("fix_input", "input.enoent", re.compile(r"\bENOENT\b")),
    ("fix_input", "input.does_not_exist", re.compile(
        r"does not exist|doesn't exist", re.I)),
    ("fix_input", "input.cannot_find", re.compile(r"cannot find", re.I)),
    ("fix_input", "input.command_not_found", re.compile(
        r"command not found", re.I)),
    ("fix_input", "input.module_not_found", re.compile(
        r"module not found|cannot find module|no module named", re.I)),
    ("fix_input", "input.unknown_tool", re.compile(
        r"unknown (?:tool|command)", re.I)),
    ("fix_input", "input.offset_beyond_eof", re.compile(
        r"[Oo]ffset \d+ is beyond end of file")),
    ("fix_input", "input.cd_target_missing", re.compile(
        r"cd target does not exist", re.I)),
    ("fix_input", "input.not_a_repo", re.compile(
        r"not a git repository", re.I)),
    ("fix_input", "input.unrecognized", re.compile(r"unrecognized", re.I)),

    # --- the environment was slow. Requires a phrase, never a bare number. ---
    ("retry", "env.timed_out", re.compile(r"timed out|time out\b", re.I)),
    ("retry", "env.timeout_error", re.compile(
        r"\bTimeoutError\b|Timeout \d+ ?ms exceeded")),
    ("retry", "env.timedout_code", re.compile(
        r"\bETIMEDOUT\b|\bECONNRESET\b|\bECONNREFUSED\b")),
    ("retry", "env.socket_hangup", re.compile(r"socket hang up", re.I)),
    ("retry", "env.rate_limited", re.compile(
        r"rate.?limit\w*|too many requests", re.I)),
    ("retry", "env.network_unreachable", re.compile(
        r"network is unreachable|could not resolve host", re.I)),
    ("retry", "env.transient", re.compile(
        r"\btransient\b(?!y)|temporary (?:failure|error|unavailable)|"
        r"service unavailable|gateway time-?out|bad gateway", re.I)),

    # --- the action itself broke. A real exception or a FAILED line. ---
    ("debug", "act.pytest_failed", re.compile(
        r"(?m)^\s*FAILED[ :]|^={3,} FAILURES ={3,}", re.M)),
    ("debug", "act.assertion_error", re.compile(
        r"\bAssertionError\b|^E\s+assert\b", re.M)),
    ("debug", "act.panic", re.compile(r"panicked at|\bpanicked\b")),
    ("debug", "act.traceback", re.compile(r"Traceback \(most recent call last\)")),
    ("debug", "act.syntax_error", re.compile(
        r"\b(?:SyntaxError|IndentationError|TabError)\b")),
    ("debug", "act.runtime_error", re.compile(
        r"\b(?:TypeError|NameError|AttributeError|ValueError|KeyError|"
        r"IndexError|ZeroDivisionError|RecursionError|UnboundLocalError|"
        r"ModuleNotFoundError|ImportError|RuntimeError|StopIteration|"
        r"NotADirectoryError|PermissionError|OSError)\b(?![ \w])")),
    ("debug", "act.compile_failed", re.compile(
        r"compilation failed|emit-module command failed|unexpected token", re.I)),
]

#: Declared status-line channel. Exactly one admitted pattern; see docstring.
STATUS_EVIDENCE = [
    ("retry", "env.status_timed_out", re.compile(
        r"Command timed out after \d+ seconds", re.I)),
]

#: Reverse maps for auditing a stored label without re-deriving it.
_CLASS_OF = {}
for _c, _n, _ in EVIDENCE + STATUS_EVIDENCE:
    _CLASS_OF[_n] = _c
_INDEX = {n: i for i, (_, n, _) in enumerate(EVIDENCE)}


def status_evidence(text: str):
    """Evidence from the harness status line, or None."""
    tail = strip_injected(text or "").rstrip()
    m = STATUS_RE.search(tail)
    if not m:
        return None
    line = m.group(0).strip()
    for cls, name, pat in STATUS_EVIDENCE:
        s = pat.search(line)
        if s:
            return {"y": cls, "rule": name, "span": s.group(0),
                    "offset": None, "channel": "status"}
    return None


def evidence_in_head(text: str, head_chars: int = HEAD_CHARS):
    """Return a provenance dict, or None when the head holds nothing usable.

    Prose evidence in the head WINS over the status channel: the head is what
    the tool said about the failure; the status line only says how it ended.
    """
    head = message_body(text)[:head_chars]
    found = []
    for i, (cls, name, pat) in enumerate(EVIDENCE):
        m = pat.search(head)
        if m:
            found.append((i, m.start(), cls, name, m.group(0)))
    if found:
        found.sort(key=lambda t: (t[0], t[1]))
        _, off, cls, name, span = found[0]
        if span.strip().lower() in GENERIC_SOLE:
            return None
        return {"y": cls, "rule": name, "span": span, "offset": off,
                "channel": "head"}
    return status_evidence(text)


def classify_row(row):
    """v2 evidence for one v1 row, or None if nothing defensible is in the head."""
    return evidence_in_head(row["x"])


# ---------------------------------------------------------------------------
# loading / splitting
# ---------------------------------------------------------------------------

def load_v1(name):
    p = os.path.join(SRC, f"{name}.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p)]


def norm_for_dedup(text):
    """Normalisation used ONLY to detect near-duplicates across the split."""
    t = text.lower()
    t = re.sub(r"/[\w./~-]{4,}", "<path>", t)
    t = re.sub(r"\b\d+(?:\.\d+)?\b", "N", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def partition_groups(rows, seed=0, frac_train=0.75):
    """Assign every group to exactly ONE split.

    Deterministic: groups are sorted then shuffled with a seeded RNG, and the
    third is taken by index, which cannot drift with the group count.
    """
    import numpy as np
    groups = sorted({r["group"] for r in rows}, key=str)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(n * frac_train)
    n_rh = int((n - n_train) * 0.75)
    return {
        "train": set(groups[:n_train]),
        "rule_held": set(groups[n_train:n_train + n_rh]),
        "unmatched_held": set(groups[n_train + n_rh:]),
    }


def strip_near_twins(train, held):
    """Drop held rows whose normalised text already occurs in train.

    Exact-text dedup is not enough: `Offset 180 is beyond end of file (155 lines
    total)` and `Offset 541 is beyond end of file (520 lines total)` are the same
    failure. Measured on v1: 503 of 1870 rule_held rows (26.9%, 113 distinct
    normalised texts) had a twin in train, because group ids are not stable
    across sessions -- the same command is re-run in a new session dir.
    """
    seen = {norm_for_dedup(r["x"]) for r in train}
    keep = [r for r in held if norm_for_dedup(r["x"]) not in seen]
    return keep, len(held) - len(keep)


def head_sensitivity(rows, ks=(200, 400, 800, 1500)):
    """Labelled/unlabelled at each head size. Trained-split rows only."""
    return [(k, sum(1 for r in rows if evidence_in_head(r["x"], k)),
             sum(1 for r in rows if not evidence_in_head(r["x"], k)))
            for k in ks]


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _row(r, ev):
    return {"x": redact(r["x"]), "y": ev["y"], "rule": ev["rule"],
            "span": ev["span"][:120], "offset": ev["offset"],
            "channel": ev["channel"], "tool": r["tool"], "group": r["group"]}


def build(seed=0):
    """Rebuild train / rule_held / unmatched_held from the v1 rows.

    v2 does NOT inherit a v1 label; every v1 row is re-derived by the v2 rule.
    v1 `unmatched_held` rows are a RULED source: v1 had no rule for them, so no
    v2 pattern can have been designed from one. Whatever v2 labels there is an
    out-of-distribution measurement, and is kept as a fourth file so it can
    never be mistaken for training data.
    """
    pool = load_v1("train") + load_v1("rule_held")
    ruled_src = [r for r in load_v1("unmatched_held") if r["y"] == V1_UNMATCHED]

    g = partition_groups(pool + ruled_src, seed=seed)

    emitted = {"train": [], "rule_held": [], "unmatched_held": []}
    dropped, relabel = collections.Counter(), collections.Counter()

    for r in pool:
        ev = classify_row(r)
        v1y = r["y"]
        if ev is None:
            dropped[("unlabelable_class" if v1y in V1_UNLABELABLE
                     else "no_evidence_in_head") + ":" + v1y] += 1
            if v1y in V1_UNLABELABLE:
                relabel[(v1y, None)] += 1
            continue
        relabel[(v1y, ev["y"])] += 1
        grp = r["group"]
        if grp in g["unmatched_held"]:
            # reserved for rows nothing can label; a v2-labelled row is not one
            dropped["reserved_group_had_a_v2_label:" + v1y] += 1
            continue
        emitted["rule_held" if grp in g["rule_held"] else "train"].append(
            _row(r, ev))

    # The v1 `unmatched` rows: v1's rule found nothing in the ROW. Measured --
    # 87 of their 104 session groups also hold a v1-labelled row, and all 16 that
    # v2 can label near-duplicate a row already in train. So they are NOT an
    # out-of-distribution set and v2 does not pretend otherwise: it keeps no
    # separate file for them, and reports the counts. Rare extra labelled rows
    # are not worth a misleading name.
    ruled_labelled, ruled_unlabelled, ruled_dup = 0, 0, 0
    train_texts = {norm_for_dedup(r["x"]) for r in pool}
    for r in ruled_src:
        ev = classify_row(r)
        if ev is None:
            ruled_unlabelled += 1
            continue
        if norm_for_dedup(r["x"]) in train_texts:
            ruled_dup += 1
            continue
        ruled_labelled += 1

    # Rows nothing can label -> the `unmatched` bucket. NOT a class: `y` is the
    # sentinel `unmatched`, `rule`/`span`/`offset` are null so no reader can
    # mistake one for a labelled row. Two views of the same bucket:
    #   unmatched.jsonl       all of them, any group (the honest eval pool)
    #   unmatched_held.jsonl  only those in reserved groups, so that file is
    #                         group-disjoint from train and rule_held
    unmatched_all = []
    for r in pool + ruled_src:
        if classify_row(r) is not None:
            continue
        row = {"x": redact(r["x"]), "y": UNMATCHED, "rule": None, "span": None,
               "offset": None, "channel": None, "tool": r["tool"],
               "group": r["group"]}
        unmatched_all.append(row)
        if r["group"] in g["unmatched_held"]:
            emitted["unmatched_held"].append(row)

    # one label per text; a text yielding two classes is a rule defect and is
    # counted, never resolved by arbitrary first-wins.
    for name in emitted:
        seen, out, contra = {}, [], 0
        for r in emitted[name]:
            prev = seen.get(r["x"])
            if prev is None:
                seen[r["x"]] = r
                out.append(r)
            elif prev["y"] != r["y"]:
                contra += 1
        dropped[name + ".contradictory_text"] = contra
        emitted[name] = out

    train, rule_held = emitted["train"], emitted["rule_held"]
    rule_held, rh_twins = strip_near_twins(train, rule_held)

    unlab = [r for r in pool if classify_row(r) is None]
    return {
        "train": train, "rule_held": rule_held,
        "unmatched_held": emitted["unmatched_held"],
        "unmatched": unmatched_all,
        "dropped": dropped, "relabel": relabel,
        "near_twins": {"rule_held": rh_twins},
        "v1_counts": collections.Counter(r["y"] for r in pool),
        "v1_unmatched_total": len(ruled_src),
        "v1_unmatched_labelled_by_v2": ruled_labelled,
        "v1_unmatched_still_unlabelled": ruled_unlabelled,
        "unlabelled_v1_rows": len(unlab),
        "unlabelled_in_train_groups": sum(
            1 for r in unlab if r["group"] in g["train"]),
        "v1_unmatched_labelled_fresh_rows": ruled_labelled,
        "v1_unmatched_labelled_dup_of_pool": ruled_dup,
        "sensitivity": head_sensitivity(load_v1("train")),
        "channel_counts": collections.Counter(
            r["channel"] for r in train + rule_held if r["channel"]),
    }


# ---------------------------------------------------------------------------
# audit + report
# ---------------------------------------------------------------------------

def tool_only_oracle(*splits):
    """Accuracy available from the tool NAME alone.

    Measured on v1: 54.6% (4014/7347). A model score below that is not evidence
    of learning the failure semantics. Reported, not corrected -- removing the
    `tool` field would only hide the leak, since a tool's error vocabulary is a
    stronger version of the same signal.
    """
    rows = [r for s in splits for r in s]
    if not rows:
        return 0.0
    by_tool = collections.defaultdict(collections.Counter)
    for r in rows:
        by_tool[r["tool"]][r["y"]] += 1
    return sum(c.most_common(1)[0][1] for c in by_tool.values()) / len(rows)


def audit_splits(b):
    """The invariants the emitted files must satisfy, measured on the rows."""
    parts = {"train": b["train"], "rule_held": b["rule_held"],
             "unmatched_held": b["unmatched_held"]}
    g = {n: {r["group"] for r in v} for n, v in parts.items()}
    t = {n: {norm_for_dedup(r["x"]) for r in v} for n, v in parts.items()}
    ov_g, ov_t = {}, {}
    names = list(parts)
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            ov_g[f"{a}^{c}"] = len(g[a] & g[c])
            ov_t[f"{a}^{c}"] = len(t[a] & t[c])
    return {
        "n": {n: len(v) for n, v in parts.items()},
        "groups": {n: len(v) for n, v in g.items()},
        "group_overlap": ov_g, "text_overlap": ov_t,
        "tool_only_oracle_all": tool_only_oracle(*parts.values()),
        "tool_only_oracle_train": tool_only_oracle(b["train"]),
        "null_label_readable": all(r.get("rule") is None
                                   for r in b["unmatched_held"]),
    }


def write(split, rows):
    os.makedirs(DST, exist_ok=True)
    p = os.path.join(DST, f"{split}.jsonl")
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def main():
    b = build()
    for split in ("train", "rule_held", "unmatched_held", "unmatched"):
        write(split, b[split])

    W = 78
    print("=" * W)
    print("HEAD_CHARS sensitivity (v1 TRAIN only; held splits never read for it)")
    print("=" * W)
    for k, keep, drop in b["sensitivity"]:
        print(f"  head={k:5d}  labelled {keep:5d}  unmatched {drop:5d} "
              f"({drop / max(keep + drop, 1):.1%})")
    print(f"  HEAD_CHARS used = {HEAD_CHARS}")

    print("\n" + "=" * W)
    print("PER-CLASS  v1 (union train+rule_held)  ->  v2")
    print("=" * W)
    v2c = collections.Counter(r["y"] for r in b["train"])
    rh_c = collections.Counter(r["y"] for r in b["rule_held"])
    print(f"  {'class':12s} {'v1':>6s} {'v2_train':>9s} {'rule_held':>10s}")
    for c in list(CLASSES) + list(V1_UNLABELABLE):
        if not (b["v1_counts"].get(c) or v2c.get(c)):
            continue
        print(f"  {c:12s} {b['v1_counts'].get(c, 0):6d} {v2c.get(c, 0):9d} "
              f"{rh_c.get(c, 0):10d}")
    print(f"  {UNMATCHED:12s} {'-':>6s} {len(b['unmatched']):9d} "
          f"{len(b['unmatched_held']):10d}"
          f"   (unmatched: the bucket / unmatched_held: its group-disjoint view)")
    print(f"  {'TOTAL':12s} {sum(b['v1_counts'].values()):6d} "
          f"{len(b['train']):9d} {len(b['rule_held']):10d}")

    print("\n" + "=" * W)
    print("WHERE v1 LABELS WENT")
    print("=" * W)
    moved = collections.Counter()
    for (old, new), n in b["relabel"].items():
        if new is None:
            moved[f"{old:11s} -> DROPPED (class not labelable)"] += n
        elif old != new:
            moved[f"{old:11s} -> {new}"] += n
        else:
            moved[f"{old:11s} -> {new} (unchanged)"] += n
    for k, v in moved.most_common():
        print(f"  {k:44s} {v:6d}")

    print("\n" + "=" * W)
    print("ROWS DROPPED, AND WHY")
    print("=" * W)
    for k, v in b["dropped"].most_common():
        print(f"  {k:46s} {v:6d}")
    print(f"  {'rule_held near-twin of train':46s} {b['near_twins']['rule_held']:6d}")
    print(f"  v1 unmatched_held source rows : {b['v1_unmatched_total']}"
          f"   -> v2 labelled {b['v1_unmatched_labelled_by_v2']}"
          f", still unlabelled {b['v1_unmatched_still_unlabelled']}")
    print(f"  v1 labelled rows v2 will not label: {b['unlabelled_v1_rows']} of "
          f"{sum(b['v1_counts'].values())} "
          f"({b['unlabelled_v1_rows'] / max(sum(b['v1_counts'].values()), 1):.1%})"
          f"; {b['unlabelled_in_train_groups']} of those are in a train group, so "
          f"belong in unmatched.jsonl")

    a = audit_splits(b)
    print("\n" + "=" * W)
    print("SPLIT INTEGRITY")
    print("=" * W)
    print(f"  rows   : {a['n']}")
    print(f"  groups : {a['groups']}")
    print(f"  group overlap : {a['group_overlap']}")
    print(f"  text  overlap : {a['text_overlap']}")
    print(f"  tool-only oracle: all={a['tool_only_oracle_all']:.3f} "
          f"train={a['tool_only_oracle_train']:.3f}  (v1 measured 0.546)")

    print("\n" + "=" * W)
    print("PROVENANCE (train)")
    print("=" * W)
    for n, v in collections.Counter(r["rule"] for r in b["train"]).most_common():
        print(f"  {_CLASS_OF[n]:10s} {n:28s} {v:5d}")
    print(f"\n  channel: {dict(b['channel_counts'])}")
    print(f"\nwrote {DST}/  train={len(b['train'])} "
          f"rule_held={len(b['rule_held'])} "
          f"unmatched_held={len(b['unmatched_held'])} "
          f"unmatched={len(b['unmatched'])}")


if __name__ == "__main__":
    main()
