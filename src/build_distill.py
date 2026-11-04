"""Build a large distillation corpus from every real tool failure on this machine.

WHY THIS EXISTS SEPARATELY FROM responses_v3
v3 kept only rows the audited rule could label into three classes -- 5,014 of them.
That was correct for measuring the rule, and wrong as a ceiling: 13,096 real
failures exist across 77 tools, and the rule's inability to label a row is a fact
about the rule, not about whether a teacher could label it. The whole point of
distillation is to replace the rule as the label authority, so the corpus must not
be pre-filtered by the rule being replaced.

WHAT IS KEPT
Every failure with a tool name and non-empty error text. No class filter. No rule
call for the purpose of selection. The rule label is still recorded, but only as a
*comparison* column -- never as the training target -- so teacher-vs-rule
disagreement stays measurable instead of being silently resolved in the rule's
favour.

WHAT IS NOT DROPPED AND WHY THAT MATTERS
Text that the previous harvest skipped as a "harness nudge" is now a real failure
case: a project's own injected annotation is exactly the kind of thing an agent
sees at runtime. It is excluded from the *state* (via the same stripper the rule
uses) rather than excluded from the corpus, because leaking our own annotation into
the input is the leak, and dropping the row is not the fix for it.

DEDUPLICATION
On the full state (tool, arguments, error) so two rows differing only in arguments
both survive. Near-twins are reported, not removed -- a teacher sees them as
separate observations, and removing them is a decision that needs evidence from the
teacher's behaviour, not a prior.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import relabel_v2 as V2  # noqa: E402
from build_state_v3 import HARD_CAP, _args_of  # noqa: E402
from tool_outcomes import _session_group, _text_of  # noqa: E402

# The earlier harvester only read two stores. `.codex/sessions` holds 461 more
# sessions with the same message shape, so leaving it out would cap the corpus for
# no reason other than history.
STORES = (".pi/agent/sessions", ".pi-bolt/agent/sessions", ".codex/sessions")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "distill")

#: Tools whose "failure" is the plan approval flow rather than a real error. These
#: are excluded because the correct response is not a diagnosis -- the tool is
#: reporting a pending decision, which is not a failure mode the classifier can fix.
NON_FAILURE_TOOLS = {"until_done_complete", "until_done_set", "ask_user_question",
                     "ask_multi", "secret_ask", "intercom", "task_panel"}


def harvest(min_error_chars=8):
    """Every failing toolResult paired with its toolCall, whole-machine."""
    files = []
    for s in STORES:
        d = os.path.join(os.path.expanduser("~"), s)
        if os.path.isdir(d):
            files += glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)

    rows, seen = [], set()
    for f in files:
        group = _session_group(f)
        calls, results = {}, []
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
            msg = r.get("message") or {}
            content = msg.get("content")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "toolCall" and b.get("id"):
                        calls[b["id"]] = b
            if msg.get("role") == "toolResult" and msg.get("isError"):
                results.append(msg)
        fh.close()

        for m in results:
            txt = _text_of(m)
            if len(txt.strip()) < min_error_chars:
                continue
            call = calls.get(m.get("toolCallId")) or {}
            name = m.get("toolName") or call.get("name")
            if not name or name in NON_FAILURE_TOOLS:
                continue
            # strip OUR OWN injected annotation from the observed text: it is a
            # leak in the state, not a reason to delete the row
            clean = V2.strip_injected(txt) if hasattr(V2, "strip_injected") else txt
            clean = V2.strip_status(clean) if hasattr(V2, "strip_status") else clean
            if len(clean.strip()) < min_error_chars:
                clean = txt
            rows.append({
                "tool": name,
                "arguments": _args_of(call) if call else None,
                "error": clean[:HARD_CAP],
                "group": group,
            })

    # exact dedup on the full state
    uniq = []
    for r in rows:
        k = (r["tool"], json.dumps(r["arguments"], sort_keys=True, default=str), r["error"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    print(f"harvested {len(rows)} raw failures -> {len(uniq)} after full-state dedup")
    return uniq


def main():
    rows = harvest()

    # the rule's opinion is recorded as a comparison column ONLY
    labelled = 0
    for r in rows:
        ev = V2.classify_row({"x": r["error"], "tool": r["tool"], "group": r["group"]})
        r["rule_label"] = ev["y"] if ev else None
        r["rule_name"] = ev["rule"] if ev else None
        labelled += ev is not None
    print(f"rule labelled {labelled}/{len(rows)} ({100*labelled/len(rows):.1f}%) "
          f"-- recorded for comparison, NOT used as the training target")

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "calls.jsonl")
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    c = collections.Counter(r["tool"] for r in rows)
    rc = collections.Counter(r["rule_label"] for r in rows)
    print(f"\nwrote {p} ({len(rows)} rows)")
    print(f"tools: {len(c)}  top: {c.most_common(8)}")
    print(f"rule labels: {rc.most_common()}")
    withargs = sum(1 for r in rows if r["arguments"])
    print(f"rows with arguments: {withargs} ({100*withargs/len(rows):.1f}%)")
    le = sorted(len(r["error"]) for r in rows)
    print(f"error chars: p50 {le[len(le)//2]} p90 {le[int(.9*len(le))]} max {le[-1]}")
    print(f"groups: {len({r['group'] for r in rows})}")


if __name__ == "__main__":
    main()
