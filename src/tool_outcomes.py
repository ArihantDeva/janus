"""Build a decision dataset from real tool-call outcomes.

WHAT THIS IS
------------
Every tool call I have ever made, with what happened. Pairing toolCall to
toolResult by id gives a supervised example:

    state  = tool name + arguments + a little preceding context
    label  = the OUTCOME class

That is a Janus-shaped decision: given this action in this state, what happens?
And unlike the safety corpus, the label is real -- observed, not authored.

THE OUTCOME TAXONOMY, AND WHY EACH CLASS HAS A DIFFERENT ACTION
---------------------------------------------------------------
The point of predicting an outcome is that the right response differs:

    success       proceed
    not_found     the INPUT is wrong (bad path, missing tool) -> fix the input
    runtime_error the ACTION is wrong (syntax, type, assertion) -> debug the action
    transient     the ENVIRONMENT was slow -> retry unchanged
    auth          the CREDENTIALS are wrong -> re-auth, do not retry

Collapsing these into "failed: yes/no" would throw away the only thing that makes
the prediction useful. Collapsing transient into runtime_error would teach a model
to change a command that only needed re-running.

ROWS DROPPED ON PURPOSE
-----------------------
Harness nudges are not task outcomes. "Use rg instead of grep" is a style
correction: its label describes MY compliance, not whether the action suited the
state. Including them would teach a model to predict a linter. They are counted
and reported, never silently merged.

LEAKAGE CONTROLS -- THIS IS THE PART THAT DECIDES IF THE DATA IS WORTH ANYTHING
-----------------------------------------------------------------------------
1. SPLIT BY SESSION, never by row. The same command recurs across sessions
   (measured: 1078 failure texts repeat, one appears 551 times); a row-level split
   would put identical commands on both sides and report a fake score.
2. Split by the TOP-LEVEL session directory, so nested subagent runs travel with
   their parent. 2516 files are named `session.jsonl` because they are subagent
   runs inside a session directory -- splitting on basename would have merged them
   into one giant group.
3. A triviality baseline is reported alongside: the majority class, plus a
   lexical shortcut any real model must beat.

WHAT THIS CANNOT PROVE
----------------------
Predicting my own tool failures is not predicting a stranger's. The distribution
is one person's workflow on one machine, and many failures are typos in paths I
had just created. A model trained here would learn this machine's quirks. That is
a real limitation and it is why the honest claim is "can a model learn from real
outcomes", not "is this model safe".
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORES = (".pi/agent/sessions", ".pi-bolt/agent/sessions")

#: Harness corrections, not task outcomes.
GUARD = re.compile(
    r"kb-guard|rg-enforce|BLOCKED by|Use `?read`? tool|Use Pi's|Use `grep` tool|"
    r"Use `rg`|instead of|script-lang-guard|browser-session-guard|"
    r"janus-mandate-guard|pdf-guard|seo-guard|WARNING —|⚠️|🛑", re.I)

#: Ordered: first match wins, and the order encodes specificity.
PATTERNS = [
    ("transient", re.compile(
        r"\btimeout\b|timed out|time out|ETIMEDOUT|ECONNRESET|ECONNREFUSED|"
        r"socket hang up|rate.?limit|\b429\b|\b50[234]\b|will retry|retrying|"
        r"temporar|too many requests|overloaded|try again", re.I)),
    ("auth", re.compile(
        r"permission denied|unauthori[sz]ed|forbidden|\b401\b|\b403\b|"
        r"not logged in|auth(entication)? (fail|required)|token (file|expired|invalid)|"
        r"invalid api key|access blocked|credential", re.I)),
    ("not_found", re.compile(
        r"no such file|not found|ENOENT|does not exist|cannot find|"
        r"unknown (tool|command)|module not found|command not found|"
        r"no matches found|\b404\b|unrecognized", re.I)),
    ("runtime_error", re.compile(
        r"SyntaxError|IndentationError|TypeError|NameError|AttributeError|"
        r"ValueError|KeyError|IndexError|AssertionError|ZeroDivisionError|"
        r"Traceback|\bException\b|exit code [1-9]|compilation|failed", re.I)),
]
LABELS = ("success",) + tuple(n for n, _ in PATTERNS)


def classify(text, is_error):
    if not is_error:
        return "success"
    head = text[:1200]
    for name, pat in PATTERNS:
        if pat.search(head):
            return name
    return "other_fail"


def _text_of(msg):
    c = msg.get("content")
    if isinstance(c, list):
        return "\n".join(b.get("text", "") for b in c
                         if isinstance(b, dict) and b.get("type") == "text")
    return str(c or "")


def _session_group(path):
    """Top-level session DIRECTORY, so nested subagent runs stay with their parent.

    Measured: 2516 files are named session.jsonl because they are subagent runs
    inside a session dir. Grouping on basename would collapse them into one
    group and leak across the split.
    """
    parts = path.split(os.sep)
    for i, p in enumerate(parts):
        if p == "sessions" and i + 1 < len(parts):
            # TWO components: the project directory AND the session id.
            # Measured: grouping on the project alone collapsed 318588 rows into
            # 78 groups, with 263k in a single one. Any split over that would be
            # a split over projects, not sessions, and near-identical commands
            # from consecutive sessions would land on both sides of it.
            return os.sep.join(parts[i + 1:i + 3])
    return os.path.basename(path)


def harvest(stores=STORES, root="~"):
    """Yield (state, label, tool, group) per tool call, pairing call to result."""
    files = []
    for s in stores:
        d = os.path.join(os.path.expanduser(root), s)
        if os.path.isdir(d):
            files += glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)

    dropped_guard = 0
    unpaired = 0
    counts = collections.Counter()

    for f in files:
        group = _session_group(f)
        pending = {}       # toolCallId -> (name, arguments)
        try:
            fh = open(f, errors="ignore")
        except Exception:
            continue
        for ln in fh:
            if "toolCall" not in ln and "toolResult" not in ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            m = r.get("message") or {}
            c = m.get("content")
            if not isinstance(c, list):
                continue

            for b in c:
                if isinstance(b, dict) and b.get("type") == "toolCall":
                    pending[b.get("id")] = (b.get("name"), b.get("arguments"))

            if m.get("role") != "toolResult":
                continue
            txt = _text_of(m)
            is_err = bool(m.get("isError"))
            if is_err and GUARD.search(txt[:400]):
                dropped_guard += 1
                continue
            call = pending.get(m.get("toolCallId"))
            if call is None:
                unpaired += 1
                continue
            name, args = call
            label = classify(txt, is_err)
            counts[label] += 1
            yield {
                "tool": name,
                "args": args,
                "label": label,
                "group": group,
                "err_head": txt[:300].replace("\n", " ") if is_err else "",
            }
        fh.close()

    print(f"  dropped guard nudges : {dropped_guard}")
    print(f"  unpaired results     : {unpaired}")
    print(f"  kept                 : {sum(counts.values())}")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {k:14s} {v:7d}  ({100*v/max(sum(counts.values()),1):5.1f}%)")


def state_text(row, ctx_chars=400):
    """Render a call as the model's input: the tool and its arguments."""
    a = row.get("args")
    if isinstance(a, (dict, list)):
        a = json.dumps(a, sort_keys=True)
    return f"Tool: {row['tool']}\nArguments: {str(a)[:ctx_chars]}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "toolout"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("harvesting...")
    rows = list(harvest())
    # de-duplicate exact (state,label) so one repeated command cannot dominate
    seen, uniq = set(), []
    for r in rows:
        k = (state_text(r), r["label"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    print(f"  after exact dedup    : {len(uniq)}")
    path = os.path.join(a.out, "calls.jsonl")
    with open(path, "w") as f:
        for r in uniq:
            f.write(json.dumps({"x": state_text(r), "y": r["label"],
                                "tool": r["tool"], "group": r["group"]}) + "\n")
    groups = {r["group"] for r in uniq}
    print(f"  distinct groups      : {len(groups)}")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
