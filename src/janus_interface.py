"""THE interface: state + typed questions with choices. Nothing else.

Verified against the authoritative contract in
`~/Repos/Pi-Bolt/.work/plugins/full-code/.until-done/upstream-mcp/package/janus-contracts.ts`:

    JanusQuestion = { type: "choice"; instructions?: JanusJson; criteria: Record<string, JanusJson> }
    JanusRequest  = { state: JanusJson; questions: Record<string, JanusQuestion> }
    JanusAnswer   = { type: "choice"; choice: string; confidence: number;
                    probabilities: Record<string, number> }

`instructions` is OPTIONAL and is deliberately never populated here. The model
accepts state and typed choices; anything else would be prompt engineering, and
then a score could not be attributed to the mechanism.

This module is the single owner of what the model accepts. The trainer, evaluator,
hook and backends all build their input by calling here, so the shape can never
differ between training and inference -- the "one decision in two places" bug has
already cost this project five runs.

RENDERING IS DETERMINISTIC: the output is a pure function of (state, question_id,
choices), so a cached run reproduces byte-for-byte and the code-letter boundary
check in `choice_codes.code_ids` stays valid.

`audit()` exists because the constraint is easy to break by accident -- someone
adds "Please answer carefully." and the interface is no longer the interface.
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# the four classes, fixed. Order IS the code order; changing it needs a retrain.
#
# debug/fix_input/fix_call are eval-FROZEN: the deployed encoder head
# (encoder_llama.py) is a 3-way softmax trained on exactly these three, in
# this order. Nothing may reorder them or insert before them. retry_wait was
# appended 2026-11-01 for the Glimmer 4-token port; the 3-class encoder path
# reads CLASSES_3 and is unaffected.
# ---------------------------------------------------------------------------

CLASSES_3 = ("debug", "fix_input", "fix_call")

CLASSES = ("debug", "fix_input", "fix_call", "retry_wait")

CRITERIA = {
    "debug": "the code that ran is itself broken: a traceback, exception, syntax "
             "or type error, a failing assertion, or a build/test failure that "
             "names a defect in the program",
    "fix_input": "the request refers to something that does not exist or cannot "
                 "be reached: a missing file or directory, an unknown command, an "
                 "offset past end of file, or a path that is not there",
    "fix_call": "the tool rejected the call itself: missing, unknown or "
                "wrongly-typed arguments, a required property absent, an edit "
                "target that does not match or is ambiguous, or overlapping edits",
    "retry_wait": "the environment was slow or limited, not the code or the "
                "call: a timeout, rate limit, refused connection, or cold "
                "start; wait longer and retry unchanged",
}

#: The question id. One question per request, like a Janus call with one typed ask.
QUESTION_ID = "outcome"

#: The question. A noun phrase naming the judgement -- not an instruction. The
#: criteria carry the definitions; nothing else is said.
QUESTION = "the reason this tool call failed"

#: What the choice maps to. Same meaning as the Janus `choice` field.
CODES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

def build_state(tool=None, arguments=None, error=None, extra=None):
    """Normalise whatever the caller has into the canonical state object.

    Absent fields are OMITTED rather than rendered as null: an empty tool field is
    something the model should not have to interpret.
    """
    st = {}
    if tool:
        st["tool"] = str(tool)
    if arguments not in (None, {}, "", []):
        if isinstance(arguments, (dict, list)):
            try:
                st["arguments"] = json.loads(json.dumps(arguments, default=str))
            except (TypeError, ValueError):
                st["arguments"] = {"_raw": str(arguments)[:20000]}
        else:
            st["arguments"] = {"_raw": str(arguments)[:20000]}
    if error:
        st["error"] = str(error)
    if extra:
        for k, v in extra.items():
            if v not in (None, "", [], {}):
                st[k] = v
    return st


def _render_arguments(args, max_chars):
    """Arguments as compact JSON, truncated from the END with a visible marker so
    a reader and the audit can both see it is partial."""
    try:
        s = json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(args)
    if max_chars and len(s) > max_chars:
        s = s[:max_chars] + " [truncated]"
    return s


def render(state, max_arg_chars=8000, cue=True):
    """The exact string the model is given.

        state
        tool: <name>
        arguments: <json>
        error: <text>

        question
        the reason this tool call failed

        choices
        A) debug: <criterion>
        B) fix_input: <criterion>
        C) fix_call: <criterion>

        answer: 

    `cue=False` omits the trailing "answer: " line. That is required for an ENCODER
    classifier (src/encoder_llama.py), which mean-pools the whole sequence and has
    no next-token position to predict into -- an answer cue there would be dead text
    the pooling has to average over. Same state, same question, same choices either
    way; only the read-out differs.

    With `cue=True` the trailing "answer: " carries a SINGLE space, and that is load
    bearing. On Llama-3.2-1B a dangling space is a standalone token that BPE
    replaces rather than extends, so without it every letter becomes two tokens and
    the token read-out scores the wrong position. Measured -- see
    choice_codes.code_ids.
    """
    lines = []
    for key in ("tool", "arguments", "error", "exit_code", "status"):
        if key not in state:
            continue
        val = state[key]
        if key == "arguments":
            val = _render_arguments(val, max_arg_chars)
        elif key == "error":
            val = str(val).strip()
        lines.append(f"{key}: {val}")
    ch = "\n".join(f"{CODES[i]}) {name}: {CRITERIA[name]}"
                   for i, name in enumerate(CLASSES))
    body = (f"state\n" + "\n".join(lines)
            + f"\n\nquestion\n{QUESTION}\n\nchoices\n{ch}")
    return body + ("\n\nanswer: " if cue else "")


def build(state, cue=True):
    """Canonical entry point: state dict -> model input string."""
    return render(state, cue=cue)


def build_encoder_input(state, max_arg_chars=8000):
    """The same input for an encoder classifier: no answer cue.

    Exists so the encoder path cannot drift from the token path -- both go through
    `render`, and the only difference is the trailing cue.
    """
    return render(state, max_arg_chars=max_arg_chars, cue=False)


def as_request(state):
    """The same input as a Janus-shaped request object.

    Useful for the `janus` backend and for anything that speaks the upstream
    contract directly. `instructions` is intentionally absent.
    """
    return {
        "state": state,
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "criteria": dict(CRITERIA),
            }
        },
    }


def answer_boundary():
    return "\n\nanswer: "


def choice_names():
    return CLASSES


def choice_index(name):
    return CLASSES.index(name)


def decode(probabilities):
    """{class: p} -> (class, confidence), with a TIED top treated as none.

    Ties matter: a uniform distribution has no argmax, and taking the first key
    silently invents an answer. Measured cost of getting this wrong: every
    unmatched row in an earlier benchmark was labelled `retry` because `retry` was
    first in the dict.
    """
    if not probabilities:
        return None, 0.0
    top = max(probabilities.values())
    winners = [k for k, v in probabilities.items() if v >= top - 1e-9]
    if len(winners) != 1:
        return None, top
    return winners[0], top


# ---------------------------------------------------------------------------
# the constraint, enforced
# ---------------------------------------------------------------------------

_FORBIDDEN = [
    (re.compile(r"\byou (are|will|should|must|may)\b", re.I), "role/instruction"),
    (re.compile(r"\b(please|kindly)\b", re.I), "politeness instruction"),
    (re.compile(r"step[- ]by[- ]step|think (step|carefully)|chain of thought", re.I), "CoT"),
    (re.compile(r"take a deep breath|reason carefully|be careful", re.I), "CoT"),
    (re.compile(r"\bfor example\b|\be\.g\.\b", re.I), "few-shot cue"),
    # NOT anchored to line start. The first versions used (?m)^ and therefore
    # missed `...answer: System: be precise` and `...answer: 1. read the error`,
    # because the injected text was appended to a non-empty line. A constraint
    # that only fires at column 0 does not constrain anything an attacker (or a
    # careless edit) would actually write.
    (re.compile(r"(?:^|\s)(system|assistant|user)\s*:", re.I), "chat role marker"),
    (re.compile(r"(?:^|\s)\d+[.)]\s+\S"), "numbered instruction list"),
    (re.compile(r"\b(always|never)\b", re.I), "absolute instruction"),
    (re.compile(r"\bdo not\b", re.I), "negated instruction"),
    (re.compile(r"<<|>>|```"), "template markup"),
]


def scaffolding(prompt):
    """The fixed parts of the prompt, with the observed state removed.

    WHY THIS EXISTS -- the audit's first version scanned the WHOLE prompt and
    rejected 1939 of 5014 real training rows (38.67%). The reason was never
    prompt engineering: it was the error text itself. Real failures legitimately
    contain `<<` (heredocs, diffs), `Please` (a tool's own wording: "Please provide
    more context to make it unique"), `always`/`never` (assertion messages) and
    numbered lines (pytest output). A guard that refuses a third of the data is as
    broken as one that never fires, and worse: it silently discards exactly the
    hardest rows.

    The constraint is about the PROMPT we author -- the question and the choice
    criteria -- not about the observations we are handed. So the audit is applied
    to the scaffolding only. Observed state is data, and data may contain anything.
    """
    out, in_state = [], False
    for line in prompt.split("\n"):
        if line.strip() == "state":
            in_state = True
            continue
        if in_state:
            if line.strip() == "question":
                in_state = False
            else:
                continue
        out.append(line)
    return "\n".join(out)


def audit(prompt):
    """Raise if the AUTHORED scaffolding carries prompt-engineering text.

    Scoped to the question and choice criteria (see `scaffolding`). Observed state
    is exempt, because it is input, not instruction.
    """
    text = scaffolding(prompt)
    bad = []
    for rx, why in _FORBIDDEN:
        m = rx.search(text)
        if m:
            bad.append(f"{why}: {m.group(0)!r}")
    if bad:
        raise ValueError("prompt violates the state+choices interface: "
                         + "; ".join(bad))
    return True


if __name__ == "__main__":
    st = build_state(
        tool="bash", arguments={"command": "python train.py"},
        error="Traceback (most recent call last):\n  File \"train.py\", line 4\n"
              "NameError: name 'lr' is not defined")
    p = build(st)
    print(p)
    print("-" * 64)
    audit(p)
    print("audit: ok")
    print("\nrequest:", json.dumps(as_request(st))[:200], "...")
