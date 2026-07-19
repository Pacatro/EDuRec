from dataclasses import dataclass

import torch
from torch import nn

from ... import settings


@dataclass
class FusionConfig:
    emb_dim: int = settings.EMB_DIM
    n_heads: int = settings.NUM_HEADS
    dropout: float = settings.DROPOUT


class CrossAttention(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.emb_dim = cfg.emb_dim
        self.n_heads = cfg.n_heads
        self.norm = nn.LayerNorm(cfg.emb_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.emb_dim,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )

    def forward(self, sources: list[torch.Tensor]) -> torch.Tensor:
        x = torch.stack(sources, dim=1)
        q = x[:, 0:1]
        k = x
        v = x

        out, _ = self.attn(q, k, v, need_weights=False)
        out = self.norm(out.squeeze(1))

        return out
