"""Checkpoint fidelity at the public distribution boundary, fully offline."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import backends as B
import janus_interface as JI
from encoder_fixture import save_tiny_checkpoint

TEXT = JI.build_encoder_input(JI.build_state(tool="bash", error="missing file"))


def reference(model, tokenizer):
    inputs = tokenizer([TEXT], padding=True, return_tensors="pt")
    with torch.no_grad():
        probs = model.eval()(**inputs).float().softmax(-1)[0].tolist()
    return dict(zip(JI.CLASSES_3, probs))


def restored(path, head=None):
    backend = B.get_backend("encoder", encoder=str(path),
                            head=str(head or path / "head.pt"))
    return backend(TEXT)


def test_full_checkpoint_restores_saved_predictions_not_random_base(tmp_path):
    """Replacing from_pretrained with LlamaModel(config) must fail this."""
    model, tokenizer = save_tiny_checkpoint(tmp_path)
    expected = reference(model, tokenizer)
    assert restored(tmp_path) == pytest.approx(expected, abs=1e-5)
    assert restored(tmp_path) == pytest.approx(expected, abs=1e-5)


def save_adapter_checkpoint(path):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel
    import encoder_llama as E
    import train_distill as T

    model, tok = save_tiny_checkpoint(path / "base")
    base = AutoModel.from_pretrained(str(path / "base"), local_files_only=True)
    E.make_bidirectional(base)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model.encoder = get_peft_model(base, LoraConfig(
            r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], lora_dropout=0))
    adapter_weights = [p for name, p in model.encoder.named_parameters() if "lora_B" in name]
    with torch.no_grad():
        for value in adapter_weights:
            value.fill_(0.2)
    T.save_checkpoint(model.eval(), tok, str(path / "adapter"))
    return model, tok


def test_adapter_checkpoint_restores_base_adapter_and_head(tmp_path):
    """Dropping the adapter or reinitializing the base changes probabilities."""
    model, tok = save_adapter_checkpoint(tmp_path)
    assert restored(tmp_path / "adapter") == pytest.approx(reference(model, tok), abs=1e-5)


def test_wrapped_head_rejects_changed_class_order(tmp_path):
    """Wrong class order is a different classifier, even with the same shape."""
    model, _ = save_tiny_checkpoint(tmp_path)
    head = tmp_path / "reordered.pt"
    torch.save({"head": model.head.state_dict(),
                "classes": ["fix_call", "fix_input", "debug"]}, head)
    with pytest.raises(ValueError, match="class"):
        restored(tmp_path, head)
