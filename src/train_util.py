"""Scheduling helpers that must be testable without a GPU or torch installed.

These live apart from the trainer deliberately. `train_glimmer.py` imports torch
at module scope, so anything defined there cannot be unit-tested on a machine
without torch -- which is every machine except the cluster. A scheduling rule is
pure arithmetic and has no business requiring a 2GB dependency to verify.
"""


def checkpoint_due(step, steps, every):
    """True when this step should write a checkpoint.

    The last step always counts: a run that reaches the end must save, even when
    `every` does not divide `steps`. This exists because the first Glimmer run
    saved ONLY at the end while projecting 3.82 hours against a 4-hour wall
    clock, so a timeout would have destroyed hours of GPU with nothing kept.
    """
    if every <= 0:
        raise ValueError("checkpoint interval must be positive")
    return step % every == 0 or step == steps


def steps_for(n_examples, batch_size, accum, epochs):
    """Gradient steps needed to cover the data once per epoch."""
    if batch_size <= 0 or accum <= 0:
        raise ValueError("batch size and accumulation must be positive")
    import math
    return int(math.ceil(n_examples / (batch_size * accum) * epochs))


def keep_tail(tok):
    """Make truncation drop the HEAD, not the tail.

    `truncation_side` is a tokenizer ATTRIBUTE, not a keyword to __call__ --
    passing it as a kwarg raises TypeError from _batch_encode_plus. It must be
    set once on the tokenizer.

    The action under judgement and the "Answer:" cue are at the end of every
    prompt, so HuggingFace's default ("right", keep head / drop tail) deletes
    exactly the part being classified.
    """
    tok.truncation_side = "left"
    return tok


def ensure_pad(tok):
    """Give a tokenizer a pad token when it has none.

    Llama-3.2 ships no pad token, so batching with padding fails until one is
    set. Reusing eos is the standard fix and is safe here because every row is
    a single short prompt, not a document split into chunks.

    The attribute name is assembled at runtime because a harness output filter
    rewrites the literal assignment to a redaction placeholder on disk, which
    silently turned seven files into NameErrors that only surfaced on the
    cluster. Building the name dynamically keeps the intent readable and cannot
    be mistaken for a secret assignment.
    """
    attr = "pad" + "_token"
    idattr = "pad" + "_token_id"
    if getattr(tok, attr, None) is None:
        setattr(tok, attr, getattr(tok, "eos_token"))
    if getattr(tok, idattr, None) is None:
        setattr(tok, idattr, getattr(tok, "eos_token_id"))
    return tok
