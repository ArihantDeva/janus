"""Tests for the Choice head.

The failure this suite is written against: the existing head CANNOT represent an
option that is not a single vocabulary token, so any "choice" result produced
through it would silently be a two-token boolean dressed up as a distribution.
These tests pin the three properties that make a Choice head a Choice head:
variable N, multi-token options, and a normalised distribution.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# The Choice head is torch code, but `make check` deliberately runs without
# torch so the data/label/leak tests stay fast and dependency-free. Skipping
# keeps the suite honest: these tests are reported as skipped, never as passed.
pytest.importorskip("torch", reason="Choice head needs torch; run with --with torch")

from choice import choice_probs, choose, confidence, score_options  # noqa: E402


class FakeTok:
    """Whitespace tokenizer. Enough to exercise multi-token options and to keep
    these tests runnable without downloading a real tokenizer."""

    def __init__(self):
        self.vocab = {}

    def _id(self, w):
        if w not in self.vocab:
            self.vocab[w] = len(self.vocab)
        return self.vocab[w]

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [self._id(w) for w in text.split()]
        import torch
        return {"input_ids": torch.tensor([ids])}


class FakeModel:
    """Scores continuations by a fixed per-word preference. Lets us assert the
    length normalisation and the distribution without a GPU."""

    def __init__(self, pref=None, n_vocab=200):
        self.pref = pref or {}
        self.n_vocab = n_vocab
        self.device = "cpu"

    def __call__(self, ids):
        import torch
        B, T = ids.shape
        # logits[i] predicts token at position i+1
        logits = torch.zeros(B, T, self.n_vocab)
        for b in range(B):
            for t in range(T):
                logits[b, t, :] = 0.0
        # encode preference: the model prefers words listed in self.pref
        inv = {v: k for k, v in _TOK.vocab.items()}
        for b in range(B):
            for t in range(T):
                for tid, w in inv.items():
                    logits[b, t, tid] = self.pref.get(w, 0.0)
        class O:
            pass
        o = O()
        o.logits = logits
        return o


_TOK = FakeTok()


def test_choice_probs_normalises():
    p = choice_probs([0.0, 1.0, 2.0])
    assert abs(p.sum() - 1.0) < 1e-12
    assert p[2] > p[1] > p[0]
    assert np.all(p >= 0)


def test_choice_probs_handles_many_options():
    # N=7, not 2. The old head structurally could not do this.
    p = choice_probs(np.linspace(-3, 3, 7))
    assert p.shape == (7,)
    assert abs(p.sum() - 1.0) < 1e-12


def test_temperature_one_preserves_order():
    raw = np.array([0.5, -1.0, 2.0])
    assert list(np.argsort(choice_probs(raw, 1.0))) == list(np.argsort(raw))


def test_confidence_is_max_probability():
    p = choice_probs([0.0, 5.0, 0.0])
    assert confidence(p) == pytest.approx(p.max())
    assert choose(p) == 1


def test_multitoken_options_are_representable():
    """The property the shipped head lacks: an option of >1 token."""
    multi = ["block and ask the user", "allow"]
    assert len(multi[0].split()) == 5
    # the old cand_ids() would SystemExit on this; the choice head scores it
    raw = score_options(FakeModel({"allow": 2.0}), _TOK,
                        "Context: x Action: y Answer:", multi)
    assert raw.shape == (2,)
    assert np.all(np.isfinite(raw))


def test_length_normalised_mean_is_token_invariant():
    """A sequence of k identical per-token log-probs must normalise to that same
    value for every k.

    This is the property that stops the head being a verbosity detector: if the
    division is missing, a 5-token option returns 5x the score of a 1-token
    option with identical per-token likelihood. Reintroducing the bug (dropping
    the division) makes this test fail with -5.0 vs -1.0.
    """
    import torch

    from choice import _length_normalised

    one = _length_normalised(torch.tensor([-1.0]))
    five = _length_normalised(torch.tensor([-1.0] * 5))
    assert one == pytest.approx(-1.0)
    assert five == pytest.approx(-1.0), (
        f"5-token score {five} != 1-token score {one} with identical per-token "
        f"likelihood -- length normalisation missing")
    assert one - five == pytest.approx(0.0, abs=1e-9)


def test_option_longer_than_maxlen_raises_rather_than_truncating():
    """A silently truncated option would be scored as a different option."""
    _TOK.vocab.clear()
    long_opt = " ".join(["w"] * 20)
    with pytest.raises(ValueError):
        score_options(FakeModel(), _TOK, "Answer:", [long_opt], maxlen=10)
