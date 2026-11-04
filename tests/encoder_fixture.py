"""Offline, seeded model fixtures for checkpoint boundary tests."""

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaModel, PreTrainedTokenizerFast

import encoder_llama as E


def save_tiny_checkpoint(path):
    path.mkdir(parents=True, exist_ok=True)
    vocab = {word: i for i, word in enumerate(
        ["[PAD]", "[UNK]", "state", "tool", "bash", "error", "boom", "question",
         "choices", "debug", "fix_input", "fix_call", "missing", "file", "test"])}
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]",
                                 unk_token="[UNK]",
                                 model_input_names=["input_ids", "attention_mask"])
    tok.save_pretrained(str(path))
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=512)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        enc = LlamaModel(cfg)
        E.make_bidirectional(enc)
        model, _ = E.wrap_classifier(enc, str(path), 32, 3)
    enc.save_pretrained(str(path))
    torch.save(model.head.state_dict(), str(path / "head.pt"))
    return model.eval(), tok
