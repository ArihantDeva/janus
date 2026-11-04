"""Letter-code Choice head: N options as single-token codes, softmax over them.

WHERE THIS COMES FROM
---------------------
bespokelabsai/nimble does exactly the conversion this project needs: LoRA on a
pretrained LM, cross-entropy over ALLOWED CANDIDATE LOGITS, hard labels. Its
option encoding is the part I had wrong. Each option gets a ONE-TOKEN CODE
(A..Z) rendered in the prompt, and the model's distribution is a softmax over
exactly those K code-token ids:

    probabilities = raw[token_ids].softmax(dim=-1)

This repo already does that at K=2: `noul.candidates()` returns the ids for
" no" / " yes" and the head softmaxes over the pair. So this is not a new
mechanism, it is the same mechanism with K widened from 2 to 26 -- which is what
makes a genuine 3-way ALLOW/ASK/BLOCK answer representable.

WHY NOT THE OTHER TWO APPROACHES
--------------------------------
- open-janus's ChoiceHead embeds options with a separate encoder and dot-products
  them into the read-out. That needs an option ENCODER trained end to end; a
  LoRA on a causal LM has no such encoder, so it cannot be adopted without
  building a new model.
- Scoring option strings as text continuations (what src/choice.py did) scored
  AUC 0.379, below chance, because the learned signal lives in a two-way softmax
  over two token ids and re-reading it as string likelihoods recovers the base
  model's prior instead of the signal.

THE BOUNDARY CHECK IS NOT OPTIONAL
----------------------------------
BPE merges across text boundaries. Appending a code directly to a bare
"Answer:" prefix merged 15 of 26 letters on Llama-3.2-1B in this repo, so the
"code" was not a token at all and the read would silently have scored a
different position. Codes are therefore verified exactly the way nimble does it:
encode `prompt + code`, require the prompt to be an exact prefix and the suffix
to be exactly one ordinary (non-special) token, and require all K ids distinct.
A failure raises; it never silently truncates.
"""

from __future__ import annotations

import string

import numpy as np


class CodeError(ValueError):
    """A candidate code is not usable at this prompt boundary."""


#: Ordered so option i maps to code i. 26 is the ceiling a single letter allows.
CODES = string.ascii_uppercase


def render_options(options):
    """The option block appended to the prompt.

    Codes are shown next to their text so the model can associate them, and the
    answer cue names the set it must choose from. One line per option.
    """
    if not 1 <= len(options) <= len(CODES):
        raise ValueError(
            f"need 1..{len(CODES)} options, got {len(options)}")
    lines = [f"{CODES[i]}) {opt}" for i, opt in enumerate(options)]
    return "Options:\n" + "\n".join(lines) + \
           "\n\nAnswer with the single letter of the best option.\nAnswer: "


def code_ids(tok, prompt, options):
    """Token id for each option's code, verified at THIS prompt boundary.

    Mirrors nimble's check: the prefix must be an exact token prefix of
    `prefix + code`, the suffix exactly one token, that token not special, and
    all K ids distinct. Anything else raises, because a silently-wrong code id
    produces a confidently wrong read.

    THE TRAILING SPACE IS STRIPPED BEFORE CHECKING, and that is load bearing
    rather than cosmetic. On Llama-3.2-1B:

        encode("Answer: ")  -> [16533, 25, 220]     # 220 is a STANDALONE space
        encode("Answer: A") -> [16533, 25, 362]     # 362 is " A"

    BPE *replaces* the dangling space token instead of appending after it, so
    checking against the space-suffixed prompt fails for every letter. Encoding
    the stripped prefix and appending the code mirrors the exact token sequence
    the model will see, because the renderer emits exactly one space before the
    code.
    """
    prefix = prompt.rstrip()
    base = tok.encode(prefix, add_special_tokens=False)
    out = []
    for i in range(len(options)):
        code = CODES[i]
        # Exactly what the renderer emits: stripped prefix, ONE space, code.
        # Concatenating without the space makes BPE merge ':' with the letter
        # ('Answer:'+'A' -> one token), and keeping the dangling space makes BPE
        # REPLACE that space token instead of appending. Both were measured.
        comb = tok.encode(prefix + " " + code, add_special_tokens=False)
        if comb[:len(base)] != base:
            raise CodeError(
                f"code {code!r}: prefix is not an exact token prefix at this "
                f"boundary (BPE merged across it)")
        suffix = comb[len(base):]
        if len(suffix) != 1:
            raise CodeError(
                f"code {code!r} is {len(suffix)} tokens at this boundary, not 1")
        if suffix[0] in tok.all_special_ids:
            raise CodeError(f"code {code!r} resolves to a special token")
        out.append(suffix[0])
    if len(set(out)) != len(out):
        raise CodeError(f"codes are not distinct token ids: {out}")
    return out


def choice_prompt(base_prompt, options):
    """Base prompt with the option block and an answer cue.

    `base_prompt` must NOT already end in an answer cue; the trailing space of
    `render_options` is what makes the code a standalone token, so it is load
    bearing and must not be stripped.
    """
    return base_prompt.rstrip() + "\n\n" + render_options(options)


def probs_from_logits(logits, ids, temperature=1.0):
    """Softmax over ONLY the candidate code logits.

    Every other vocabulary entry is discarded rather than masked to -inf, which
    is equivalent for the softmax and keeps the arithmetic in float32. This is
    the structural guarantee: with the option set fixed at K, an answer outside
    it is not representable.
    """
    import torch

    picked = logits[..., ids].float() / float(temperature)
    return torch.softmax(picked, dim=-1)


def summarise(probs):
    """Distribution plus Janus-style confidence (mass on the chosen option)."""
    p = np.asarray(probs, dtype=np.float64)
    return {"probs": p.tolist(), "choice": int(p.argmax()),
            "confidence": float(p.max())}
