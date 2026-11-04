"""Choice primitive: N arbitrary candidate answers, probability for each.

WHY THIS EXISTS
---------------
Janus's Choice primitive takes a state and a list of candidate options supplied BY
THE CALLER, and returns a probability for every option plus a confidence. The
adapter in this repo cannot do that. It reads two HARDCODED single-token answers
out of the LM head:

    WORDS = ("yes", "no")
    cand_ids(tok)  -> asserts each word is exactly ONE token, else SystemExit

so the answer set is fixed at training time, every option must be one token, and
the caller cannot choose. That is a noul-shaped head (is this statement true?),
not a Choice-shaped one.

WHAT A CHOICE HEAD NEEDS
------------------------
1. N options, not 2, and N may vary per call.
2. Options are arbitrary STRINGS, not vocabulary tokens. "block and ask the
   user" is five tokens; under the current head it is unrepresentable.
3. One distribution over the options, normalised across them, so the values are
   comparable and sum to 1 (Janus returns a probability per option, not N
   independent booleans -- independent booleans cannot express mutual
   exclusivity and will not sum to 1).

THE MECHANISM
-------------
Score each option by the model's length-normalised log-likelihood of that option
as a continuation of the prompt:

    score(option) = (1/|tokens|) * sum_t log p(token_t | prompt, token_<t)

then softmax over the option scores with a temperature. This is the standard
approach for open-ended candidate scoring (it is what LM-eval-harness calls
"acc_norm"), and it is the only way to score options that are not single
tokens.

Length normalisation is NOT cosmetic here. Without it, a long option accumulates
more negative log-probability purely from being long, so softmax would be driven
by option verbosity -- which is the same length-confound that produced a false
"length control ties the adapter" result earlier in this project. Normalising is
what stops the head from being a length detector.

WHAT THIS IS NOT
----------------
It is not a new training objective. Training still uses the existing two-token
cross-entropy; this head re-reads the SAME adapter with a different decoding
rule. So it can be evaluated today without a retrain, and a gain cannot come
from extra supervision -- only from the decoding rule and the option set.
"""

from __future__ import annotations

import numpy as np
import torch


def _length_normalised(token_logps):
    """Mean token log-likelihood. Pulled out as a named function so there is
    exactly ONE place that decides whether to divide, and a test can pin it."""
    return (token_logps.sum() / token_logps.shape[0]).item()


def score_options(model, tok, prompt, options, maxlen=512, batch_size=None):
    """Length-normalised log-likelihood of each option as a continuation.

    Returns a float array of shape (len(options),) of SUMMED-per-token scores
    (higher = more likely). Raw scores; use `choice_probs` to normalise.

    Every option is scored in its own forward pass rather than concatenated,
    because a single padded batch would let the padding position leak into the
    length normalisation. Slower, but the alternative silently reintroduces the
    truncation/length class of bug this project has already paid for twice.
    """
    if not options:
        raise ValueError("options must be non-empty")

    # The prompt ends on "Answer:" with no trailing space. Adding exactly one
    # space means the option is scored as the continuation a model would write,
    # matching how the training targets were tokenised (" no", " yes").
    prefix = prompt.rstrip() + " "

    enc_p = tok(prefix, return_tensors="pt", add_special_tokens=False)
    p_ids = enc_p["input_ids"].to(model.device)
    n_prefix = p_ids.shape[1]

    scores = []
    with torch.no_grad():
        for opt in options:
            opt_ids = tok(opt, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(model.device)
            n_opt = opt_ids.shape[1]
            if n_opt == 0:
                scores.append(float("-inf"))
                continue
            full = torch.cat([p_ids, opt_ids], dim=1)
            # Left-truncation to maxlen must never cut the option itself, so if
            # the pair is too long, drop prefix tokens from the LEFT -- the same
            # direction train_util.keep_tail() uses.
            if full.shape[1] > maxlen:
                keep = maxlen - n_opt
                if keep < 1:
                    raise ValueError(
                        f"option {opt!r} is {n_opt} tokens, leaving no room for a "
                        f"prompt within maxlen={maxlen}")
                full = torch.cat([p_ids[:, -keep:], opt_ids], dim=1)
                n_prefix_eff = keep
            else:
                n_prefix_eff = n_prefix

            out = model(full)
            logits = out.logits.float()
            del out
            # log p(option tokens | prefix)
            lp = torch.log_softmax(logits[0, :-1], dim=-1)
            tgt = full[0, 1:]
            tok_lp = lp[torch.arange(tgt.shape[0]), tgt]
            # Positions belonging to the option are the last n_opt-1 targets.
            # The option's FIRST token is predicted at index n_prefix_eff-1.
            start = n_prefix_eff - 1
            opt_lp = tok_lp[start:start + n_opt]
            scores.append(_length_normalised(opt_lp))
    return np.asarray(scores, dtype=np.float64)


def choice_probs(raw, temperature=1.0):
    """Softmax over option scores -> a proper distribution summing to 1.

    T=1 returns the model's own relative preference. Temperature exists only for
    calibration and is reported, never tuned on a benchmark split.
    """
    raw = np.asarray(raw, dtype=np.float64)
    z = raw / float(temperature)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def confidence(probs):
    """Janus-style confidence: probability mass on the chosen option."""
    p = np.asarray(probs, dtype=np.float64)
    return float(p.max())


def choose(probs):
    """Index of the highest-probability option."""
    return int(np.argmax(np.asarray(probs)))
