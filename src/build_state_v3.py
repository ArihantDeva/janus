"""v3: full call STATE + question + choices -> response class.

WHAT CHANGED FROM v2, AND WHY
-----------------------------
v2 stored only `{error text, tool name}`. That threw away the arguments the call
was made with, which is the state a classifier actually needs: `cat missing.txt`
and `rm -rf missing/` are the same error text but different jobs, and `git push`
failing vs `pytest` failing differ only in the command.

It also inherited a 1500-CHARACTER cap applied at harvest time. Measured on 16,679
raw failures with the Llama-3.2 tokenizer, the longest is 6,010 tokens and p99 is
3,737, so that cap was discarding real content for no reason. v3 stores the error
and arguments uncapped (a generous 400k char safety valve only) and lets the
trainer choose its own truncation.

"CALL CONTEXT IS NOT LEAKAGE"
v2's report called tool-name predictiveness a defect (68.4% of rows predicted by
`tool` alone). That framing was wrong. If the input is state + question + choices,
then the tool and its arguments ARE the state -- legitimate input, exactly as a
Janus call passes the observed page. What must be reported is not "the tool name was
used" but "how much of the answer is available from a lookup table", so a model's
score can be compared against that floor. `tool_only_floor()` computes it.

LABELLING IS NOT RE-INVENTED HERE
Every label comes from `relabel_v2.classify_row`, the rule that survived the
adversarial audit (evidence must sit in the head of the message, injected
annotations stripped, status line excluded, no generic prose tokens). Writing a
second rule would be the "one decision in two places" bug this project keeps
hitting. What v3 adds is CONTEXT, not labels.

DROPPED CLASSES (per the audit, not for convenience)
    reauth  -- 35 rows, and 75 of the v1 158 were the bare word "credential" inside
               pytest test names. Not trainable and not evidence.
    abort   -- was never user interrupts: pytest collection errors and
               `EINTR: Interrupted system call`.
    retry   -- 306 rows after cleaning, 255 of them one pattern (status_timed_out).
               "retry" reduces to "it timed out"; there is no separate concept here.
    stop / inspect / resolve -- 32 / 11 / 22 rows, templated junk; 7 of 22 `resolve`
               rows were successful `git commit` output.

So v3 carries three classes: debug, fix_input, fix_call.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import relabel_v2 as V2  # noqa: E402
from tool_outcomes import GUARD, _session_group, _text_of  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "responses_v3")
STORES = (".pi/agent/sessions", ".pi-bolt/agent/sessions")

CLASSES = ("debug", "fix_input", "fix_call")

#: Safety valve only. The p99 of real failures is 3,737 tokens (~13k chars) and
#: the max observed is 6,010 tokens, so nothing real is cut at this width.
HARD_CAP = 400_000


def _args_of(block):
    """Arguments from a toolCall BLOCK.

    Measured block shape: {'type': 'toolCall', 'id': ..., 'name': ..., 'arguments': ...}
    where `arguments` is a PYTHON-REPR STRING, not JSON -- the samples use single
    quotes ("{'path': '/Users/...'}"), so json.loads fails on them. ast.literal_eval
    parses them; when even that fails the raw string is kept rather than dropped,
    because a call whose arguments will not parse is still a call worth
    classifying and its text is better than nothing.

    The first version of this looked for `toolInput`/`input`/`args` -- the field
    names a toolResult MESSAGE would use -- and so returned None for every row,
    producing a dataset where 0/4332 rows carried arguments. Silent and total,
    which is exactly the kind of bug that survives to a training run.
    """
    if not isinstance(block, dict):
        return None
    ti = block.get("arguments")
    if ti is None:
        ti = block.get("toolInput") or block.get("input")
    if isinstance(ti, str):
        try:
            return json.loads(ti)
        except Exception:
            pass
        try:
            import ast
            got = ast.literal_eval(ti)
            return got if isinstance(got, dict) else {"_raw": str(got)[:20000]}
        except Exception:
            return {"_raw": ti[:20000]}
    return ti if isinstance(ti, dict) else None


def harvest():
    """Pair each failing toolResult with the toolCall that produced it.

    The pairing is by `toolCallId` across the whole session file, which is why the
    file is read twice: toolResults and toolCalls are separate rows and either may
    come first.
    """
    files = []
    for s in STORES:
        d = os.path.join(os.path.expanduser("~"), s)
        if os.path.isdir(d):
            files += glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)

    rows = []
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
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "toolCall" and b.get("id"):
                        calls[b["id"]] = b
            if msg.get("role") == "toolResult" and msg.get("isError"):
                results.append(msg)
        fh.close()

        for m in results:
            txt = _text_of(m)
            if not txt.strip() or GUARD.search(txt[:400]):
                continue
            call = calls.get(m.get("toolCallId")) or {}
            name = m.get("toolName") or call.get("name")
            args = _args_of(call) if call else None
            rows.append({
                "tool": name,
                "arguments": args,
                "error": txt[:HARD_CAP],
                "group": group,
            })
    return rows


def label(row):
    """Reuse the audited v2 rule. Returns the evidence dict or None."""
    return V2.classify_row({"x": row["error"], "tool": row["tool"],
                            "group": row["group"]})


def build(seed=0):
    raw = harvest()
    print(f"harvested failing calls with context: {len(raw)}")

    kept, dropped_cls = [], collections.Counter()
    for r in raw:
        ev = label(r)
        if ev is None:
            dropped_cls["unlabelled"] += 1
            continue
        if ev["y"] not in CLASSES:
            dropped_cls[ev["y"]] += 1
            continue
        r = dict(r)
        r["y"] = ev["y"]
        r["provenance"] = {"rule": ev["rule"], "span": ev["span"][:120],
                           "offset": ev["offset"], "channel": ev["channel"]}
        kept.append(r)

    print(f"labelled into the three classes: {len(kept)}")
    for k, v in dropped_cls.most_common():
        print(f"  dropped {k:12s} {v}")

    # dedup on (tool, arguments, error) -- the full input, so two rows that differ
    # only in arguments are kept
    seen, uniq = set(), []
    for r in kept:
        k = (r["tool"], json.dumps(r["arguments"], sort_keys=True), r["error"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    print(f"after exact dedup on full state: {len(uniq)}")

    # group-disjoint split; groups are project+session so one session cannot
    # straddle the boundary
    import numpy as np
    rng = np.random.default_rng(seed)
    groups = np.array(sorted({r["group"] for r in uniq}))
    rng.shuffle(groups)
    cut = int(len(groups) * 0.75)
    tr_g = set(groups[:cut].tolist())
    train = [r for r in uniq if r["group"] in tr_g]
    held = [r for r in uniq if r["group"] not in tr_g]

    # near-twin removal on normalised error text, which session grouping misses
    train_norm = {V2.norm_for_dedup(r["error"]) for r in train}
    held, twins = [], 0
    for r in uniq:
        if r["group"] in tr_g:
            continue
        if V2.norm_for_dedup(r["error"]) in train_norm:
            twins += 1
            continue
        held.append(r)
    print(f"removed {twins} near-twins from held")
    return {"train": train, "rule_held": held, "twins_removed": twins}


def tool_only_floor(train, held):
    """How much of the answer a lookup table gets.

    The honest floor for any claimed score. A model that matches this has learned
    nothing beyond "which tool was called"; the margin above it is the only part
    attributable to reading the error.
    """
    table = collections.defaultdict(collections.Counter)
    for r in train:
        table[r["tool"]][r["y"]] += 1
    best = {t: c.most_common(1)[0][0] for t, c in table.items()}
    default = collections.Counter(r["y"] for r in train).most_common(1)[0][0]
    hit = sum(1 for r in held if best.get(r["tool"], default) == r["y"])
    return hit / max(1, len(held))


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


def main():
    b = build()
    os.makedirs(OUT, exist_ok=True)
    write(os.path.join(OUT, "train.jsonl"), b["train"])
    write(os.path.join(OUT, "rule_held.jsonl"), b["rule_held"])

    print("\nclass balance:")
    for name in ("train", "rule_held"):
        c = collections.Counter(r["y"] for r in b[name])
        tot = sum(c.values())
        print(f"  {name:10s} {tot:5d}  " +
              "  ".join(f"{k}={v}({100*v/tot:.0f}%)" for k, v in c.most_common()))

    fl = tool_only_floor(b["train"], b["rule_held"])
    print(f"\ntool-only floor on held: {100*fl:.1f}%  "
          f"<- any accuracy claim must beat this")
    tc = collections.Counter(r["tool"] for r in b["train"])
    print(f"tools: {dict(tc.most_common(8))}")

    # how much context does the state actually carry?
    import numpy as np
    la = np.array([len(json.dumps(r["arguments"] or {})) for r in b["train"]])
    le = np.array([len(r["error"]) for r in b["train"]])
    print(f"\nargument JSON chars: p50 {np.percentile(la,50):.0f} "
          f"p90 {np.percentile(la,90):.0f} max {la.max()}")
    print(f"error text chars  : p50 {np.percentile(le,50):.0f} "
          f"p90 {np.percentile(le,90):.0f} max {le.max()}")
    with_args = sum(1 for r in b["train"] if r["arguments"])
    print(f"rows carrying arguments: {with_args}/{len(b['train'])} "
          f"({100*with_args/max(1,len(b['train'])):.1f}%)")


if __name__ == "__main__":
    main()
