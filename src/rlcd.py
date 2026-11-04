"""Losses for distillation and RLCD, as pure functions that can be tested without a GPU.

STAGE 1 -- SOFT DISTILLATION
The target is the teacher's normalised distribution, not its argmax. The loss is a
KL divergence from student to teacher, which is equivalent to cross-entropy against
the soft target up to a constant, so `kl_div` and `soft_ce` are the same objective
written two ways; both are provided and a test asserts they agree, because a silent
inconsistency between the two would make the reported number depend on which one
happened to be called.

Temperature: `softmax(log(p) / T)`. T > 1 flattens the teacher, which is standard
distillation practice when the teacher is overconfident -- measured here, Janus's
median confidence is 0.97, so its distributions are nearly one-hot and a large part
of its knowledge is in the *near-zero* entries. Without temperature those entries
carry almost no gradient. T is a knob, not a constant, so it is asserted to be load
bearing rather than assumed.

STAGE 2 -- RLCD (Reinforcement Learning from Contrastive Distillation)
The premise: a model trained only on teacher labels learns to imitate, including
imitating the teacher's mistakes. RLCD adds a contrastive signal from a
*deliberately degraded* copy of the same input, so the model learns what makes a
response correct rather than merely what the teacher said.

The degradation is the crux and the usual way this gets faked. If the negative is
random text, the task is trivial, the loss drops to zero, and nothing is learned.
The negative must be a plausible-but-wrong response to the SAME state. Here that
means: take the true state, and pair it with a WRONG class choice. The contrast is
then "this state with the right answer" vs "this state with a wrong answer", which
forces the representation to encode the deciding evidence rather than the tool name.

Two RLCD-shaped losses are provided:
  `contrastive_margin`  push the correct class above the wrong one by a margin
  `rlcd_soft`           the negative's target is the teacher distribution with the
                        chosen class suppressed; a KL against that teaches "not this"
                        without asserting a specific alternative

Both are tested for the properties that make them meaningful: monotone in the
margin, zero gradient when already satisfied (hinge), and non-degenerate when the
teacher is uniform (a uniform teacher must not be silently treated as a hard label).
"""

from __future__ import annotations

import numpy as np


def softmax_np(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def soften(dist, temperature=1.0, eps=1e-12):
    """Teacher distribution -> temperature-softened target.

    T == 1 returns the distribution unchanged (up to float noise). T > 1 flattens,
    T < 1 sharpens. Applying T to a *probability* requires going through the log,
    which is why the corpus stores probabilities and this is the only place the
    conversion happens.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    d = np.asarray(dist, dtype=np.float64)
    if np.any(d < 0):
        raise ValueError("distribution has negative entries")
    total = d.sum(-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("distribution sums to zero")
    d = d / total
    if temperature == 1.0:
        return d
    return softmax_np(np.log(np.clip(d, eps, None)) / temperature)


def soft_ce(student_logits, target_dist):
    """Cross-entropy of the student against a soft target. Lower is better."""
    s = np.asarray(student_logits, dtype=np.float64)
    t = np.asarray(target_dist, dtype=np.float64)
    logp = np.log(np.clip(softmax_np(s), 1e-12, None))
    return float(-(t * logp).sum(-1).mean())


def kl_div(student_logits, target_dist):
    """KL(target || student). Equal to soft_ce up to the target's own entropy."""
    s = np.asarray(student_logits, dtype=np.float64)
    t = np.asarray(target_dist, dtype=np.float64)
    logp = np.log(np.clip(softmax_np(s), 1e-12, None))
    logt = np.log(np.clip(t, 1e-12, None))
    return float((t * (logt - logp)).sum(-1).mean())


def entropy(dist):
    """Shannon entropy in nats. Used to decide whether a teacher row carries signal."""
    d = np.asarray(dist, dtype=np.float64)
    return float(-(d * np.log(np.clip(d, 1e-12, None))).sum(-1).mean())


# ---------------------------------------------------------------------------
# RLCD contrastive stage
# ---------------------------------------------------------------------------

def contrastive_margin(pos_logits, neg_logits, pos_idx, margin=0.5):
    """Hinge on (correct class score) - (wrong class score).

    `pos_logits` are the student's logits for the true class on a real row; the
    comparison is against the SAME row's score for a wrong class, so the gradient
    acts on the discriminating direction rather than on the row's overall scale.

    Returns 0 when the margin is already satisfied -- a satisfied example must not
    keep pushing, or the representation drifts for no gain.
    """
    p = np.asarray(pos_logits, dtype=np.float64)
    n = np.asarray(neg_logits, dtype=np.float64)
    gap = p - n
    return float(np.clip(margin - gap, 0.0, None).mean())


def contrastive_pairs(teacher_dist, rng, classes, min_gap=0.05):
    """Build (positive_index, negative_index) pairs from a teacher distribution.

    The negative is the teacher's SECOND choice, not a random class. That is the
    difference between a real contrast and a free one: the runner-up is the option
    the model is most likely to confuse, so separating it is the actual difficulty.

    `min_gap` is the smallest top-two gap that counts as the teacher having drawn a
    distinction at all. My first version used 1e-6 -- a numerical-noise threshold --
    which accepted a distribution of (0.34, 0.33, 0.33) as having a meaningful
    runner-up. A 0.01 gap out of 1.0 is 1% of the mass: the teacher is saying "no
    preference" and the pair would manufacture a boundary it never asserted. The
    default is 0.05 (5% of the mass) so a near-uniform row is skipped rather than
    treated as a contrast. The parameter is explicit rather than magic because the
    right value is a judgement about the teacher, not about floating point.
    """
    d = np.asarray(teacher_dist, dtype=np.float64)
    if d.ndim != 1 or d.shape[0] != len(classes):
        raise ValueError(f"expected {len(classes)} classes, got shape {d.shape}")
    order = np.argsort(-d)
    pos, neg = int(order[0]), int(order[1])
    if d[pos] - d[neg] < min_gap:
        return None
    return pos, neg


def rlcd_soft(student_logits, teacher_dist, wrong_idx, classes, strength=1.0):
    """RLCD-shaped objective: pull toward the teacher, push away from a wrong class.

    `strength` weights the negative term against the KL. A `strength` of 0 must
    reduce exactly to plain distillation -- tested, because a stage that silently
    equals stage 1 would be reported as "RLCD ran" while changing nothing.
    """
    s = np.asarray(student_logits, dtype=np.float64)
    t = np.asarray(teacher_dist, dtype=np.float64)
    base = kl_div(s, t)
    if wrong_idx is None or strength == 0.0:
        return base
    probs = softmax_np(s)
    # penalise the probability mass on the wrong class, in the log domain so the
    # penalty cannot be offset by inflating a third class
    penalty = float(-np.log(np.clip(1.0 - probs[wrong_idx], 1e-12, None)))
    return base + strength * penalty
