"""The Janus 3+1 verdict contract: prompt, target, and read in one place.

Same role as noul.py for the binary guard, but for the post-failure verdict:
given that a tool call already failed, what should the agent do next.

Four verdicts, each a single LETTER the model emits (nimble's letter-code
read-out: A/B/C/D are guaranteed single tokens, while the words fix_input
and retry_wait split into two). The prompt binds each letter to the verdict
word plus its meaning; the target sends each class to its letter's token;
the read takes the argmax over the four letter slots. All three name the
SAME letters in the SAME order.

MEASURED 2026-11-01 on Glimmer-30B (Amarel tok check): space-prefixed words
  debug [16930] SINGLE, fix_input [9461, 13807] MULTI, fix_call [9461, 38048]
  MULTI, retry_wait [48288, 52345] MULTI. Word read-out impossible: 3 of 4
  words are two tokens. Letters adopted; words stay in the prompt as meanings.
"""

from __future__ import annotations

#: The four verdict words, in read-out order.
WORDS = ("debug", "fix_input", "fix_call", "retry_wait")

#: The read-out letters. A/B/C/D, guaranteed single tokens on both families
#: (nimble's letter-code read-out; Llama-3.2-1B verified the same shape).
CODES = "ABCD"

#: Human meanings, rendered verbatim into the prompt. Keep each to one line:
#: the prompt binds the letter to the word+meaning, and the binding is pinned.
MEANINGS = {
    "debug": "the action itself is broken; read the error and change the code",
    "fix_input": "the path or handle does not exist; correct the input",
    "fix_call": "the tool rejected the call itself; correct the arguments",
    "retry_wait": "the environment was slow or limited; wait longer and retry unchanged",
}

INSTRUCTION = (
    "A coding agent's tool call failed. Decide what it should do next.\n"
    + "".join(f"{CODES[i]}) {w}: {MEANINGS[w]}\n"
              for i, w in enumerate(WORDS))
    + "Answer with exactly one letter and nothing else: "
    + ", ".join(CODES[:len(WORDS)])
)


def candidates(cids):
    """Logit slots in WORDS order; cids maps letter -> token id."""
    return [cids[CODES[i]] for i in range(len(WORDS))]


def target_index(label, cids):
    """Cross-entropy target slot for a class label (the word's own position)."""
    if label not in WORDS:
        raise ValueError(f"unknown verdict {label!r}; choose from {WORDS}")
    return WORDS.index(label)
