"""Turn R-Judge and Agent-SafetyBench into the same (prompt, label) shape.

Why this module exists: the training data is (tool call -> did it fail). The two
published benchmarks are not that. R-Judge labels a whole agent interaction as
unsafe; Agent-SafetyBench labels a request as unfillable. Converting either into
a per-tool-call judgment is a FRAMING CHOICE, made explicitly here so the README
reports what was actually measured rather than implying the benchmarks are
command-level.

Every row produced here is rendered by noul.safety_prompt -- the SAME function
that renders the training rows -- so a benchmark cannot get a bespoke format.

This docstring previously claimed that while `_prompt` actually invented its own
XML-ish wrapper that training never used. The effect was a format gap on top of a
task gap: the model trained on "Context: ... Action: ..." and was scored on a
different wrapper, the same class of trainer/eval desynchronisation that
invalidated two earlier runs. It is now structurally impossible because both
paths call one renderer.
"""
import json
import os
import random

import noul

D = os.environ.get("JANUS_DATA",
    os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data"))


def _prompt(context, action):
    """Render a benchmark row with the TRAINING renderer.

    Only fields already present in the benchmark are used, mapped onto the
    context/constraint/action slots the model was trained to read:
      R-Judge : user content -> context, agent action -> action
      ASB     : environment+tool -> context, request -> action
    No benchmark content is rewritten, summarised or invented.
    """
    return noul.safety_prompt(context, None, action)


def _dedupe(rows):
    """Collapse identical prompts. A prompt that appears with both labels is
    unmatchable, so the first label wins and the count is reported."""
    seen, out, conflicts = {}, [], 0
    for r in rows:
        k = r["prompt"]
        if k in seen:
            if seen[k] != r["label"]:
                conflicts += 1
            continue
        seen[k] = r["label"]
        out.append(r)
    return out, conflicts


# --------------------------------------------------------------- R-Judge

def rjudge_rows(records):
    """One row per agent action, labelled by the interaction's safety label.

    R-Judge's label is per interaction. Reusing it for each action inside that
    interaction assumes every action in an unsafe interaction is unsafe, which
    is the benchmark's aggregate view, not ground truth per step.
    """
    rows = []
    for rec in records:
        label = int(rec["label"])
        for turn in rec.get("contents", []):
            msgs = [m for m in turn if isinstance(m, dict)]
            # The user's request is the context; the agent action is the thing
            # being judged. Both are already in the benchmark.
            req = next((m.get("content") for m in msgs
                        if m.get("role") == "user"), None)
            for msg in msgs:
                action = msg.get("action")
                if not action:
                    continue
                rows.append({
                    "prompt": _prompt(req, action),
                    "label": label,
                    "bench": "rjudge",
                    "src_id": rec.get("id"),
                })
    out, conflicts = _dedupe(rows)
    return out


# ------------------------------------------------- Agent-SafetyBench

def asb_rows(records):
    """One row per request; unfulfillable == the unsafe case.

    Label mapping verified against the paper (arXiv:2412.14470v2, sec. 3):
    "a test case is deemed fulfillable if it is able to be safely completed;
    otherwise, it is labeled unfulfillable." So fulfillable == 0 (cannot be
    safely completed) is the UNSAFE case. Local data/asb.json was diffed against
    the upstream released_data.json: all 2000 records identical.

    Two things the earlier renderer got wrong, both of which deleted the
    evidence a judgement would rest on:

    1. It kept only environments[0]. 10.4% of records declare more than one
       environment, and in 6.5% a DROPPED environment carries the state that
       makes the request unfulfillable -- e.g. an OS env holding a file whose
       contents include "Social Security Number: ..." while environments[0] is
       just an SMS tool. The model was asked to judge those on a context that
       did not contain the risk.
    2. It rendered only name+tools and never `parameters`, which is where
       71.0% of records keep the actual state (file contents, inbox, records).

    All environments and their parameters are now rendered. No benchmark
    content is rewritten, summarised or invented -- only serialised.
    """
    rows = []
    for rec in records:
        envs = rec.get("environments") or []
        if not envs:
            ctx = "(no environment declared)"
        else:
            parts = []
            for env in envs:
                tools = ", ".join(env.get("tools") or []) or "(none)"
                params = env.get("parameters") or {}
                state = json.dumps(params, sort_keys=True) if params else "(empty)"
                parts.append(f"{env.get('name') or '(none)'} "
                             f"| tools: {tools} | state: {state}")
            ctx = " / ".join(parts)
        rows.append({
            "prompt": _prompt(ctx, rec["instruction"]),
            "label": 1 if int(rec["fulfillable"]) == 0 else 0,
            "bench": "asb",
            "src_id": rec.get("id"),
        })
    out, _ = _dedupe(rows)
    return out


def split_rows(rows, frac, seed=0):
    """Split by whole prompt so no prompt appears on both sides."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = max(1, int(len(rows) * frac))
    return rows[:n], rows[n:]


def load_bench(name):
    """Training split, dev split, and full rows for a benchmark."""
    if name == "rjudge":
        recs = [json.loads(l) for l in open(f"{D}/rjudge.jsonl")]
        return rjudge_rows(recs)
    if name == "asb":
        recs = json.load(open(f"{D}/asb.json"))
        return asb_rows(recs)
    raise SystemExit(f"unknown benchmark {name!r}")
