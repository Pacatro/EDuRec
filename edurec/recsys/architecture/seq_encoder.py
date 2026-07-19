from dataclasses import dataclass

import torch
from torch import nn

from ... import settings


@dataclass
class SeqEncoderConfig:
    emb_dim: int
    n_heads: int
    n_blocks: int
    ff_dim: int
    dropout: float = 0.1
    norm_first: bool = True
    max_history_len: int = settings.MAX_HISTORY_LEN
    num_ctx_feats: int = 0


class SeqEncoder(nn.Module):
    def __init__(self, cfg: SeqEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.pos_emb = nn.Embedding(cfg.max_history_len, cfg.emb_dim)
        self.ctx_proj = (
            nn.Linear(cfg.num_ctx_feats, cfg.emb_dim, bias=False)
            if cfg.num_ctx_feats > 0
            else None
        )
        self.input_norm = nn.LayerNorm(cfg.emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.emb_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=cfg.norm_first,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_blocks,
            norm=nn.LayerNorm(cfg.emb_dim),
            enable_nested_tensor=not cfg.norm_first,
        )

    def forward(
        self,
        history_emb: torch.Tensor,
        history_mask: torch.Tensor,
        history_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, history_len, _ = history_emb.shape
        safe_mask = history_mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        safe_mask[empty_rows, 0] = True

        positions = torch.arange(history_len, device=history_emb.device).unsqueeze(0)
        tokens = history_emb + self.pos_emb(positions)

        if self.ctx_proj is not None and history_ctx is not None:
            tokens = tokens + self.ctx_proj(history_ctx)

        tokens = self.input_norm(tokens)
        tokens = tokens * safe_mask.unsqueeze(-1).float()

        causal_mask = torch.triu(
            torch.ones(history_len, history_len, device=history_emb.device),
            diagonal=1,
        ).bool()

        encoded = self.transformer(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=~safe_mask,
        )

        valid_counts = history_mask.long().sum(dim=1)
        last_indices = (valid_counts - 1).clamp(min=0)
        gather_index = last_indices.view(batch_size, 1, 1).expand(
            -1, 1, self.cfg.emb_dim
        )
        seq_user_emb = encoded.gather(dim=1, index=gather_index).squeeze(1)

        has_history = valid_counts > 0
        return seq_user_emb * has_history.unsqueeze(-1)
