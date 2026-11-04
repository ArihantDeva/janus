"""Explicit filesystem / corpus integration tests; no live services."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import train_distill as T


def test_save_checkpoint_writes_encoder_tokenizer_and_head(tmp_path):
    """The nn.Module wrapper has no save_pretrained; save its components."""
    pytest.importorskip("torch")
    from encoder_fixture import save_tiny_checkpoint

    model, tok = save_tiny_checkpoint(tmp_path / "source")
    out = tmp_path / "saved"
    T.save_checkpoint(model, tok, str(out))
    assert (out / "head.pt").exists()
    assert (out / "config.json").exists()
    assert (out / "tokenizer.json").exists()


def test_tfidf_ceiling_beats_floor_on_the_same_real_eval_rows():
    """Optional private-corpus integration; model-boundary tests never skip."""
    pytest.importorskip("sklearn")
    if not os.path.isfile(T.CORPUS):
        pytest.skip("private real-failure corpus not supplied")
    tr_all, ev, _ = T.split_by_group(T.load_corpus(), seed=0)
    tr, _, _ = T.split_by_group(tr_all, frac=0.12, seed=500)
    got = T.tfidf_ceiling(tr[:3000], ev[:800])
    assert got is not None
    assert got > T.tool_only_floor(tr[:3000], ev[:800])
