from dataclasses import dataclass

import torch
from torch import nn

from edurec import config


@dataclass
class RankerConfig:
    embed_dim: int
    n_heads: int
    n_blocks: int
    ff_dim: int
    num_scores: int = 1
    dropout: float = 0.1
    norm_first: bool = True
    max_histoy_len: int = config.MAX_HISTORY_LEN


class Ranker(nn.Module):
    def __init__(self, cfg: RankerConfig):
        super().__init__()
        self.cfg = cfg

        self.user_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.history_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.candidate_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)

        encoder_layer = nn.TransformerEncoderLayer(
            cfg.embed_dim,
            cfg.n_heads,
            cfg.ff_dim,
            cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=cfg.norm_first,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_blocks,
            norm=nn.LayerNorm(cfg.embed_dim),
            enable_nested_tensor=not cfg.norm_first,
        )

        self.pos_emb = nn.Embedding(cfg.max_histoy_len, cfg.embed_dim)

        self.scorer = Scorer(cfg.embed_dim, cfg.num_scores, cfg.dropout)

    def forward(
        self,
        user_emb: torch.Tensor,  # [B, D]
        history_emb: torch.Tensor,  # [B, H, D]
        candidate_emb: torch.Tensor,  # [B, C, D]
        history_mask: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:
        B, H, _ = history_emb.shape
        C = candidate_emb.shape[1]

        user_token = self.user_proj(user_emb).unsqueeze(1)  # [B, 1, D]
        history_tokens = self.history_proj(history_emb)  # [B, H, D]
        candidate_tokens = self.candidate_proj(candidate_emb)  # [B, C, D]

        # Positional Ecoding
        pos_emb = self.pos_emb(
            torch.arange(H, device=history_emb.device).unsqueeze(0).repeat(B, 1)
        )  # [B, H, D]

        history_tokens = history_tokens + pos_emb

        seq = torch.cat(
            [user_token, history_tokens, candidate_tokens], dim=1
        )  # [B, T, D]

        T = seq.shape[1]
        candidate_start_offset = 1 + H

        attn_mask = self._make_attn_mask(T, candidate_start_offset, seq.device)
        padding_mask = torch.zeros((B, 1 + H + C), dtype=torch.bool, device=seq.device)
        padding_mask[:, 1 : 1 + H] = history_mask

        out = self.transformer(seq, mask=attn_mask, src_key_padding_mask=padding_mask)

        candidate_out = out[:, candidate_start_offset:, :]  # [B, C, D]
        scores = self.scorer(candidate_out)  # [B, C, num_scores]

        return scores

    def _make_attn_mask(
        self, seq_len: int, candidate_start_offset: int, device: torch.device
    ) -> torch.Tensor:
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)) == 0
        mask[candidate_start_offset:, candidate_start_offset:] = True
        candidate_indices = torch.arange(candidate_start_offset, seq_len, device=device)
        mask[candidate_indices, candidate_indices] = False
        return mask


class Scorer(nn.Module):
    def __init__(self, emb_dim: int, num_scores: int, dropout: float):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, num_scores),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scorer(x)
