"""Glimmer frozen-encoder classification head (mean-pool + linear)."""
from __future__ import annotations


class GlimmerClf(__import__("torch").nn.Module):
    """Frozen Glimmer language_model + mean-pool + linear head."""

    def __init__(self, encoder, hidden, n_cls):
        super().__init__()
        self.encoder = encoder
        self.head = __import__("torch").nn.Linear(hidden, n_cls)

    def forward(self, input_ids, attention_mask):
        o = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = o.last_hidden_state.float()
        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-6)
        return self.head(pooled.to(self.head.weight.dtype))
