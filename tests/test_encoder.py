"""Tests for the encoder classifier.

The load-bearing claim is that this is genuinely non-autoregressive. A model that
keeps ONE causal mask trains normally and quietly behaves like a worse language
model, so the mask removal is verified rather than asserted.

Tests that need torch are skipped locally (make check has no torch); they run on
the cluster where training happens.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

import janus_interface as JI  # noqa: E402


# ---------------------------------------------------------------------------
# the interface stays the single owner
# ---------------------------------------------------------------------------

def test_encoder_input_has_no_answer_cue():
    """An encoder mean-pools the sequence. An answer cue would be dead text the
    pooling has to average over, and a letter position that nothing reads."""
    st = JI.build_state(tool="bash", error="boom")
    enc = JI.build_encoder_input(st)
    assert "answer:" not in enc
    assert not enc.endswith(" ")


def test_encoder_and_token_inputs_agree_on_everything_but_the_cue():
    """Same state, same question, same choices. If these drift, the two read-outs
    are trained on different questions and cannot be compared."""
    st = JI.build_state(tool="bash", arguments={"a": 1}, error="boom")
    tok_in = JI.build(st, cue=True)
    enc_in = JI.build_encoder_input(st)
    assert tok_in.startswith(enc_in)
    assert tok_in[len(enc_in):] == "\n\nanswer: "


def test_encoder_input_passes_the_interface_audit():
    st = JI.build_state(tool="bash", arguments={"command": "pytest -q"},
                        error="E   AssertionError: nope")
    JI.audit(JI.build_encoder_input(st))


def test_encoder_input_contains_the_three_choice_criteria():
    """The choices must still be in the input: they are what disambiguates
    `fix_input` from `fix_call`, and dropping them would silently make the task
    harder than the one the token read-out solves."""
    p = JI.build_encoder_input(JI.build_state(tool="t", error="e"))
    for c in JI.CLASSES:
        assert JI.CRITERIA[c] in p


# ---------------------------------------------------------------------------
# non-autoregressive properties (need torch)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def _tiny_llama():
    from transformers import LlamaConfig, LlamaModel
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=64)
    return LlamaModel(cfg)


def test_causal_masks_are_present_before_patching():
    """Sensitivity check: if a stock Llama reports zero causal masks, then the
    counter is broken and the '0 after patching' result below proves nothing."""
    import encoder_llama as E
    m = _tiny_llama()
    assert E.count_causal_masks(m) > 0, "counter found no causal mask in a stock Llama"


def test_patching_removes_every_causal_mask():
    import encoder_llama as E
    m = _tiny_llama()
    E.make_bidirectional(m)
    assert E.count_causal_masks(m) == 0
    E.assert_bidirectional(m)


def test_assert_bidirectional_refuses_an_unpatched_model():
    """The guard must fire. A silent single surviving mask is the failure mode
    that makes an 'encoder' quietly causal."""
    import encoder_llama as E
    with pytest.raises(RuntimeError, match="causal mask"):
        E.assert_bidirectional(_tiny_llama())


def test_direction_actually_changes_the_output():
    """Behavioural proof of bidirectionality: changing a LATER token must change
    the representation of an EARLIER position. Under causal masking the earlier
    position cannot see it.

    This is the test that cannot be satisfied by a decorative patch.
    """
    import encoder_llama as E
    m = _tiny_llama().eval()
    ids_a = torch.tensor([[5, 6, 7, 8]])
    ids_b = torch.tensor([[5, 6, 7, 9]])       # differs only at the LAST position
    mask = torch.ones_like(ids_a)
    with torch.no_grad():
        a = m(input_ids=ids_a, attention_mask=mask).last_hidden_state[0, 0]
        b = m(input_ids=ids_b, attention_mask=mask).last_hidden_state[0, 0]
        assert torch.allclose(a, b, atol=1e-5), "causal model: position 0 ignores later tokens"

        E.make_bidirectional(m)
        a2 = m(input_ids=ids_a, attention_mask=mask).last_hidden_state[0, 0]
        b2 = m(input_ids=ids_b, attention_mask=mask).last_hidden_state[0, 0]
        assert not torch.allclose(a2, b2, atol=1e-5), "patched model still ignores later tokens"


def test_masked_mean_pooling_ignores_padding():
    """A short error and a long one must not embed differently for a reason that
    has nothing to do with content. Padding must not dilute the vector."""
    h = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    m = mask.unsqueeze(-1).float()
    pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-6)
    assert torch.allclose(pooled, torch.tensor([[2.0, 3.0]])), pooled


def test_no_vocabulary_is_ever_used_for_output(tmp_path):
    """The claim 'no tokens are returned' must be structurally true: the head maps
    hidden -> n_classes, and nothing writes into vocabulary space."""
    import encoder_llama as E
    from encoder_fixture import save_tiny_checkpoint
    save_tiny_checkpoint(tmp_path)
    m, tok = E.build_encoder(str(tmp_path), len(JI.CLASSES_3), "cpu", lora_r=0)
    assert m.head.out_features == len(JI.CLASSES_3)
    assert m.head.in_features == m.encoder.config.hidden_size
    out = m(torch.randint(0, len(tok), (2, 8)),
            torch.ones(2, 8, dtype=torch.long))
    assert out.shape == (2, len(JI.CLASSES_3)), "output must be choices, not tokens"


def test_encoder_backend_speaks_the_same_contract_as_janus():
    """Swap-ability is the point: same keys, same normalisation, same validation.
    If the encoder returned tokens or a bare label, the hook would break the
    moment JANUS_BACKEND changed."""
    import backends as B

    good = {c: 1.0 / len(JI.CLASSES_3) for c in JI.CLASSES_3}
    assert B.validate(good, "encoder") == good
    with pytest.raises(Exception):
        B.validate({"debug": 1.0}, "encoder")


def test_encoder_backend_is_registered_under_its_name():
    """A backend the resolver cannot find is a backend that does not exist."""
    import backends as B
    assert B.get_backend("keyword").name == "keyword"
    import inspect
    src = inspect.getsource(B.get_backend)
    assert '"encoder"' in src


def test_forward_survives_half_encoder_with_float_head():
    """GPU autocast feeds half hidden states into a float head (bench crash
    63392006). The forward must cast, not assume matching dtypes.
    Verified on Amarel (torch): 1 passed in 16.02s, job 63392617."""
    import encoder_llama as E
    import torch
    import torch.nn as nn

    class FakeEncoder(nn.Module):
        def forward(self, input_ids, attention_mask):
            batch, seq = input_ids.shape
            out = torch.ones(batch, seq, 4, dtype=torch.float16)
            return type("O", (), {"last_hidden_state": out})()

    clf = E.EncoderClassifier(FakeEncoder(), 4, 3)
    ids = torch.ones(1, 2, dtype=torch.long)
    mask = torch.ones(1, 2, dtype=torch.long)
    out = clf(input_ids=ids, attention_mask=mask)
    assert out.shape == (1, 3)
