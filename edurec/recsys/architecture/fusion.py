from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FusionConfig:
    emb_dim: int
    n_heads: int
    num_module_types: int
    dropout: float = 0.1


class SelfAttentionFusion(nn.Module):
    """Fuse module outputs using a learned token and self-attention."""

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.fusion_token = nn.Parameter(torch.empty(1, 1, cfg.emb_dim))
        self.module_type_embeddings = nn.Embedding(cfg.num_module_types, cfg.emb_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=cfg.emb_dim,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        nn.init.normal_(self.fusion_token, std=0.02)
        nn.init.normal_(self.module_type_embeddings.weight, std=0.02)

    def forward(
        self,
        modules: torch.Tensor,
        module_types: torch.Tensor,
        active_modules: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = modules.size(0)
        typed_modules = modules + self.module_type_embeddings(module_types)
        fusion_token = self.fusion_token.expand(batch_size, -1, -1)
        tokens = torch.cat([fusion_token, typed_modules], dim=1)

        fusion_is_active = active_modules.new_ones(1)
        active_tokens = torch.cat([fusion_is_active, active_modules])
        key_padding_mask = ~active_tokens.expand(batch_size, -1)

        attended_tokens, _ = self.attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return attended_tokens[:, 0]
