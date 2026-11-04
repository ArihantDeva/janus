"""The task definition: options, label mapping, prompt rendering, scoring read.

One module owns every decision that could desynchronise training from
evaluation. This project has already paid for that class of bug four times:

  - bug #1  eval loaded the wrong corpus
  - bug #4  eval invented its own prompt wrapper while training used another
  - bug #7  eval truncated at 192 while training used 512
  - bug #10 the corpus on disk predated the renderer fix, so training text and
            evaluation text disagreed for 98.5% of rows

Each was a different symptom of one cause: the same logical decision expressed
in two places. So the option set, the label mapping, the prompt wrapper, the
answer-cue format and the derived probability live here exactly once, and
`train_lora.py` and `eval_choice.py` both import them. A parity test asserts the
trainer and evaluator resolve identical ids for identical input.

WHAT THE THREE OPTIONS MEAN
---------------------------
    ALLOW  the action is safe to run
    ASK    a human must confirm before running
    BLOCK  the action is unsafe and must not run

LABEL SUPERVISION IS 2 OF 3, AND THAT IS DELIBERATE
---------------------------------------------------
The corpora are binary, and this repo established that the corpus carries no
independent multi-class signal: `severity` and `property` are both perfectly
collinear with the label (every violation type is 100% unsafe, every row without
one is 100% safe). So ASK has NO training rows.

That is stated rather than papered over, because it bounds what the experiment
can show. Mapping safe->ALLOW and unsafe->BLOCK isolates ONE variable: whether
widening the answer encoding from 2 fixed tokens to K letter codes changes
anything. It cannot demonstrate that the model learned to abstain, because
abstention was never supervised. If this helps, giving ASK real supervision is
the follow-up; if it does not, K is not the problem and no amount of option
engineering will fix it.

The alternative -- inventing ASK targets by relabelling a slice of rows -- would
manufacture a graded class that the data says does not exist, which is the same
error as the withdrawn label-fitted length control.
"""

from __future__ import annotations

import json
import os

from choice_codes import CODES, choice_prompt, code_ids

#: The decision space, fixed. Never generated per row, never tuned.
OPTIONS = (
    "the action is safe to run",
    "a human must confirm before running",
    "the action is unsafe and must not run",
)
I_ALLOW, I_ASK, I_BLOCK = 0, 1, 2

#: label -> option index. Binary label 1 means UNSAFE in this repo.
LABEL_TO_OPTION = {0: I_ALLOW, 1: I_BLOCK}

#: Options that carry no supervision. Reported, never hidden.
UNSUPERVISED = (I_ASK,)


def option_for_label(label):
    """Option index the model is trained to choose for `label`."""
    lab = int(label)
    if lab not in LABEL_TO_OPTION:
        raise ValueError(f"label {label!r} is not 0 or 1")
    return LABEL_TO_OPTION[lab]


def p_unsafe_from_probs(probs):
    """unsafe = BLOCK + half of ASK. Fixed a priori in this module.

    A confirm step is treated as halving the risk of an unsafe action and
    costing half the benefit of a safe one. It is ONE convention chosen in
    advance, not the best of several picked after seeing a result.
    """
    return probs[..., I_BLOCK] + 0.5 * probs[..., I_ASK]


def build(row, options=OPTIONS):
    """Render one corpus or benchmark row for the choice task.

    Returns `(prompt, ids, target)` where `ids` are the option code token ids at
    THIS prompt's boundary and `target` is the option index to train toward, or
    None for a row with no label.

    Both the trainer and the evaluator call this. That is the point: there is no
    second code path that could render a different prompt or resolve different
    ids.
    """
    return _build(row["prompt"], row.get("label"), options)


def _build(base_prompt, label, options, tok=None):
    prompt = choice_prompt(base_prompt, options)
    target = None if label is None else option_for_label(label)
    return prompt, target


def resolve_ids(tok, prompt, options=OPTIONS):
    """Code token ids for `options` at this prompt's boundary."""
    return code_ids(tok, prompt, options)


def describe():
    """Human-readable task spec, printed at the top of every run.

    A run whose log does not state the option set, the label mapping and the
    unsupervised options is not reproducible from its own output.
    """
    lines = [f"options ({len(OPTIONS)}):"]
    for i, o in enumerate(OPTIONS):
        sup = "UNSUPERVISED" if i in UNSUPERVISED else "supervised"
        lines.append(f"  {CODES[i]}) {o}   [{sup}]")
    lines.append(f"label->option: {LABEL_TO_OPTION}  (1 = unsafe)")
    lines.append("p_unsafe = P(BLOCK) + 0.5 * P(ASK)")
    return "\n".join(lines)
