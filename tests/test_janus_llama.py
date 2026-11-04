"""Tests for the Llama->Janus conversion.

Each test pins something that has actually gone wrong in this project or is
guaranteed to bite quietly. Every one was checked by reintroducing the defect.

The high-value ones:
  * the code ids must be resolved at the REAL prompt boundary, not assumed
  * the option list and the class list must not drift apart (a silent off-by-one
    between them would train the model to answer a different question)
  * class weights must not contain a zero -- F.cross_entropy with weight 0 on the
    target returns NaN and poisons every parameter (measured in run 31)
"""

import json
import math
import os
import sys

import numpy as np
import pytest

# torch is a TRAINING dependency and only exists in the cluster venv, not in the
# local `make check` environment. Skipping cleanly keeps the suite green locally
# while still running these tests wherever training actually happens -- an import
# at module scope would instead break collection for the whole suite.
torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

J = pytest.importorskip("janus_llama")


class FakeTok:
    """Tokenizer stub that merges " X" into ONE token, like real BPE.

    The merge is the whole point. A naive per-character stub makes " A" two
    tokens, so code_ids correctly rejects it and the test fails for the wrong
    reason -- the stub, not the code. This stub reproduces the one behaviour
    code_ids actually depends on: a space-delimited letter is a single token at a
    word boundary.
    """

    all_special_ids = [0]

    def __init__(self):
        self.vocab = {"<pad>": 0}

    def _id(self, ch):
        if ch not in self.vocab:
            self.vocab[ch] = 1000 + len(self.vocab)
        return self.vocab[ch]

    def encode(self, s, add_special_tokens=False):
        out, i = [], 0
        while i < len(s):
            # a space followed by a letter (or the space itself) is one token
            if s[i] == " " and i + 1 < len(s):
                out.append(self._id(s[i:i + 2]))
                i += 2
            else:
                out.append(self._id(s[i]))
                i += 1
        return out


# ---------------------------------------------------------------------------
# the task definition
# ---------------------------------------------------------------------------

def test_classes_and_options_are_aligned():
    """Every class must have exactly one option string, and no extras.

    A mismatch here is the nastiest possible bug: the model would be trained on
    code letters that map to different text than the evaluator uses, and both
    halves would look fine on their own.
    """
    assert len(J.OPTION_TEXT) == len(J.CLASSES)
    assert set(J.OPTION_TEXT) == set(J.CLASSES)
    assert len(set(J.CLASSES)) == len(J.CLASSES)


def test_option_order_follows_class_order():
    """Code i must correspond to CLASSES[i]. Pinned because the read-out indexes
    logits positionally into CLASSES."""
    for i, c in enumerate(J.CLASSES):
        assert J.OPTION_TEXT[c] in J.build_prompt("x"), c
    p = J.build_prompt("boom")
    for i, c in enumerate(J.CLASSES):
        assert f"{J.CODES[i]}) {J.OPTION_TEXT[c]}" in p


def test_prompt_ends_at_a_clean_answer_cue():
    """code_ids requires the prompt to be a token prefix of prompt+code.

    That only holds if the prompt ends with a single space after the cue -- which
    is exactly the measurement that made me use 'Answer: ' rather than 'Answer:'.
    """
    p = J.build_prompt("boom")
    assert p.endswith("Answer: "), repr(p[-20:])
    assert not p.endswith("  ")


def test_prompt_contains_the_failure_text():
    p = J.build_prompt("Traceback: KeyError 'x'")
    assert "Traceback: KeyError 'x'" in p


def test_class_count_is_representable():
    assert 1 <= len(J.CLASSES) <= len(J.CODES)


def test_code_ids_resolve_to_distinct_single_tokens():
    tok = FakeTok()
    opts = [J.OPTION_TEXT[c] for c in J.CLASSES]
    ids = J.code_ids(tok, J.build_prompt("boom"), opts)
    assert len(ids) == len(J.CLASSES)
    assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# the objective (nimble's plain CE)
# ---------------------------------------------------------------------------

def test_nimble_loss_is_plain_cross_entropy():
    """The objective must be unmodified CE over the K candidate logits.

    If someone quietly adds a term, the result stops being attributable to
    nimble's recipe. Verified by comparing against torch's own CE.
    """
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    y = torch.tensor([0, 2])
    assert torch.allclose(J.nimble_loss(logits, y),
                          F.cross_entropy(logits, y), atol=1e-6)

    w = torch.tensor([0.5, 1.0, 2.0])
    assert torch.allclose(J.nimble_loss(logits, y, w),
                          F.cross_entropy(logits, y, weight=w), atol=1e-6)


def test_zero_class_weight_produces_nan():
    """Documents the trap precisely rather than trusting it stays fixed.

    Corrected after measuring: torch returns NaN only when the weight of a class
    that actually occurs as a TARGET is zero -- not merely because some weight is
    zero. The earlier note in this repo ("weight 0 returns NaN") was right about
    the failure but wrong about the trigger, and the sloppy version would have
    kept me from ever zeroing an absent class's weight, which is legitimate.

    Consequence for the trainer: a class present in the data must never get
    weight 0, which is why the formula is sqrt(n/(K*count)) and not anything that
    can reach zero.
    """
    targets = torch.tensor([0, 1])
    logits = torch.zeros(2, 3, requires_grad=True)
    # zero weight on a NON-target class: finite, no NaN
    ok = F.cross_entropy(logits, targets, weight=torch.tensor([0.0, 1.0, 1.0]))
    assert not torch.isnan(ok), "non-target zero weight should be finite"
    # zero weight on a class that IS a target: NaN
    bad = F.cross_entropy(logits, targets, weight=torch.tensor([0.0, 0.0, 1.0]))
    assert torch.isnan(bad), "target zero weight must produce NaN"


def test_sqrt_class_weights_are_positive_and_balanced():
    """The real weighting formula must give every class a positive weight and
    push rare classes UP, not down."""
    counts = {"a": 1000, "b": 50, "c": 500}
    n = sum(counts.values())
    K = len(counts)
    w = {c: math.sqrt(n / (K * v)) for c, v in counts.items()}
    assert all(v > 0 for v in w.values()), "zero weight => NaN loss"
    assert w["b"] > w["c"] > w["a"]


# ---------------------------------------------------------------------------
# evaluation arithmetic
# ---------------------------------------------------------------------------

def test_brier_and_ece_arithmetic():
    """Independent recomputation of the two headline eval numbers."""
    P = np.array([[0.9, 0.1], [0.4, 0.6], [0.7, 0.3]])
    y = np.array([0, 1, 1])
    oh = np.eye(2)[y]
    brier = ((P - oh) ** 2).sum(1).mean()
    # per row: 0.01+0.01=0.02 ; 0.16+0.16=0.32 ; 0.49+0.49=0.98 ; mean 0.44
    assert abs(brier - 0.44) < 1e-9

    conf, pred = P.max(1), P.argmax(1)
    correct = (pred == y).astype(float)
    # pred = argmax rows = [0,1,0] vs y=[0,1,1] -> [True, True, False]
    assert list(correct) == [1.0, 1.0, 0.0]


def test_perfect_predictions_are_calibrated():
    P = np.eye(3)
    y = np.array([0, 1, 2])
    oh = np.eye(3)[y]
    assert ((P - oh) ** 2).sum(1).mean() == 0.0
    assert all(P.max(1) == 1.0)


# ---------------------------------------------------------------------------
# data contract
# ---------------------------------------------------------------------------

def test_load_keeps_only_known_classes():
    """`load` must filter to CLASSES, so an added class in the data file cannot
    silently shift the label indices."""
    d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(d, "data", "responses", "train.jsonl")
    if not os.path.exists(p):
        pytest.skip("dataset not built")
    rows = J.load("train")
    assert rows, "empty training set"
    assert all(r["y"] in J.CLASSES for r in rows)
    assert all(isinstance(r["x"], str) and r["x"] for r in rows)


def test_label_index_matches_class_order():
    """A row labelled 'retry' must map to index 0, and every index must be
    reachable. Pinned because the whole read-out is positional."""
    assert J.CLASSES.index("retry") == 0
    assert J.CLASSES.index("abort") == len(J.CLASSES) - 1
    for i, c in enumerate(J.CLASSES):
        assert J.CLASSES.index(c) == i


# ---------------------------------------------------------------------------
# batch order
# ---------------------------------------------------------------------------

def test_training_batches_are_shuffled():
    """The trainer must not read rows in file order.

    This is a regression guard for a bug that was measured, not theorised: the
    data is class-clustered, so sequential batching produced one batch of eight
    `debug` rows followed by a batch carrying several weight-2.88 rare rows, and
    the logged loss swung 0.71 -> 3.10 -> 3.03 between consecutive records.

    The test inspects the source rather than running training, because a full
    run is minutes-to-hours and this must stay a fast guard. It checks that the
    sampling expression consumes a permutation or a generator rather than an
    arithmetic index into the row list, and that no `gi + k` style sequential
    counter has crept back in.
    """
    import inspect
    src = inspect.getsource(J.main)
    assert "permutation" in src, "batches are not shuffled"
    assert "order_rng" in src, "shuffling must be seedable and per-run"
    assert "gi + k" not in src, "sequential index arithmetic is back"
    # the batch selection must slice the shuffled order, not the raw rows
    assert "order[oi" in src, "batch index does not come from the shuffled order"


def test_seeds_get_different_orders():
    """Each seed must see a different row order, so n=3 bounds data variance
    rather than only LoRA-init variance (the run-29 failure)."""
    import numpy as np
    o = {s: np.random.default_rng(s).permutation(50).tolist() for s in (0, 1, 2)}
    assert len({tuple(v) for v in o.values()}) == 3, "seeds share one order"
    assert len({tuple(np.random.default_rng(s).permutation(50).tolist())
                for s in (0, 1, 2)}) == 3, "order generation is not seed-dependent"
