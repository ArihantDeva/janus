"""Restore encoder weights, optional LoRA and the fixed three-class head."""

import os

import janus_interface as JI


def load_head(path):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if "classes" in state and tuple(state["classes"]) != JI.CLASSES_3:
        raise ValueError("checkpoint class order differs from the encoder interface")
    return state["head"] if "head" in state else state


def load_encoder(path):
    import torch
    from transformers import AutoModel

    if not os.path.isfile(os.path.join(path, "adapter_config.json")):
        return AutoModel.from_pretrained(path, torch_dtype=torch.float32,
                                         local_files_only=True)
    from peft import PeftConfig, PeftModel

    config = PeftConfig.from_pretrained(path, local_files_only=True)
    base = AutoModel.from_pretrained(config.base_model_name_or_path,
                                     torch_dtype=torch.float32, local_files_only=True)
    return PeftModel.from_pretrained(base, path, local_files_only=True)


def restore(path, head):
    import encoder_llama as E

    if not path or not head:
        raise ValueError("encoder backend needs JANUS_ENCODER and JANUS_ENCODER_HEAD")
    state = load_head(head)
    encoder = load_encoder(path)
    encoder.config.use_cache = False
    E.make_bidirectional(encoder)
    E.assert_bidirectional(encoder)
    model, tokenizer = E.wrap_classifier(encoder, path, encoder.config.hidden_size,
                                         len(JI.CLASSES_3))
    model.head.load_state_dict(state)
    return model.eval(), tokenizer


def predict(model, tokenizer, text, max_chars, max_len, temperature=None):
    import torch
    from backends import validate

    text = text.rstrip()
    if text.endswith("answer:"):
        text = text[:-len("answer:")].rstrip()
    encoded = tokenizer([text[:max_chars]], return_tensors="pt", padding=True,
                         truncation=True, max_length=max_len)
    inputs = {k: encoded[k] for k in ("input_ids", "attention_mask")}
    with torch.inference_mode():
        logits = model(**inputs).float()[0]
        probs = (logits / temperature if temperature else logits).softmax(-1).tolist()
    return validate(dict(zip(JI.CLASSES_3, probs)), "janus")


def load_temperature(path):
    """Read a saved calibration temperature; absent calibration means none."""
    if not path:
        return None
    import json

    value = json.load(open(path))
    if isinstance(value, dict):
        if "classes" in value and tuple(value["classes"]) != JI.CLASSES_3:
            raise ValueError("calibration class order differs from the encoder interface")
        value = value.get("T")
    value = float(value)
    if not value > 0:
        raise ValueError("temperature must be positive")
    return value


class JanusBackend:
    """CPU inference boundary. Checkpoint, not a hardcoded model, owns the weights."""

    name = "janus"
    uses_encoder_input = True

    def __init__(self, encoder=None, head=None, base=None, max_chars=None,
                 max_len=None, temperature=None):
        self._model, self._tok = restore(encoder or os.environ.get("JANUS_ENCODER"),
                                        head or os.environ.get("JANUS_ENCODER_HEAD"))
        self.max_chars = int(max_chars or os.environ.get("JANUS_ENCODER_MAX_CHARS", 8192))
        self.max_len = int(max_len or os.environ.get("JANUS_ENCODER_MAX_LEN", 2048))
        self.temperature = (float(temperature) if temperature else
                            load_temperature(os.environ.get("JANUS_ENCODER_TEMPERATURE")))

    def __call__(self, text):
        return predict(self._model, self._tok, text, self.max_chars, self.max_len,
                       self.temperature)
