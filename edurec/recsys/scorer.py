from dataclasses import dataclass

import torch
from torch import nn

from edurec import config


@dataclass
class ScorerConfig:
    emb_dim: int
    n_heads: int
    n_blocks: int
    ff_dim: int
    num_scores: int = 1
    dropout: float = 0.1
    norm_first: bool = True
    max_histoy_len: int = config.MAX_HISTORY_LEN


class Scorer(nn.Module):
    def __init__(self, cfg: ScorerConfig):
        super().__init__()
        self.cfg = cfg

        self.user_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=False)
        self.history_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=False)
        self.candidate_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=False)

        encoder_layer = nn.TransformerEncoderLayer(
            cfg.emb_dim,
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
            norm=nn.LayerNorm(cfg.emb_dim),
            enable_nested_tensor=not cfg.norm_first,
        )

        self.pos_emb = nn.Embedding(cfg.max_histoy_len, cfg.emb_dim)

        self.scorer = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.emb_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.emb_dim // 2, cfg.num_scores),
        )

    def forward(
        self,
        user_emb: torch.Tensor,  # [B, D]
        history_emb: torch.Tensor,  # [B, H, D]
        candidate_emb: torch.Tensor,  # [B, C, D]
        history_mask: torch.Tensor | None,  # [B, H]
    ) -> torch.Tensor:
        single_candidate = candidate_emb.ndim == 2
        if single_candidate:
            candidate_emb = candidate_emb.unsqueeze(1)

        B, H, _ = history_emb.shape
        C = candidate_emb.shape[1]

        user_token = self.user_proj(user_emb).unsqueeze(1)  # [B, 1, D]
        history_tokens = self.history_proj(history_emb)  # [B, H, D]
        candidate_tokens = self.candidate_proj(candidate_emb)  # [B, C, D]

        if history_mask is None:
            history_mask = torch.ones(
                (B, H), dtype=torch.bool, device=history_emb.device
            )

        # Positional Ecoding
        pos_emb = self.pos_emb(
            torch.arange(H, device=history_emb.device).unsqueeze(0).repeat(B, 1)
        )  # [B, H, D]

        history_tokens = (history_tokens + pos_emb) * history_mask.unsqueeze(-1).float()

        seq = torch.cat(
            [user_token, history_tokens, candidate_tokens], dim=1
        )  # [B, T, D]

        T = seq.shape[1]
        candidate_start_offset = 1 + H

        attn_mask = self._make_attn_mask(T, candidate_start_offset, seq.device)
        padding_mask = torch.zeros((B, 1 + H + C), dtype=torch.bool, device=seq.device)
        padding_mask[:, 1 : 1 + H] = ~history_mask

        out = self.transformer(seq, mask=attn_mask, src_key_padding_mask=padding_mask)

        candidate_out = out[:, candidate_start_offset:, :]  # [B, C, D]
        scores = self.scorer(candidate_out)  # [B, C, num_scores]

        if single_candidate:
            return scores.squeeze(1).squeeze(-1)

        return scores

    def _make_attn_mask(
        self, seq_len: int, candidate_start_offset: int, device: torch.device
    ) -> torch.Tensor:
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

        causal_mask = torch.triu(
            torch.ones(candidate_start_offset, candidate_start_offset, device=device),
            diagonal=1,
        ).bool()
        mask[:candidate_start_offset, :candidate_start_offset] = causal_mask

        mask[candidate_start_offset:, :candidate_start_offset] = False

        if candidate_start_offset < seq_len:
            candidate_indices = torch.arange(
                candidate_start_offset, seq_len, device=device
            )
            mask[candidate_indices, candidate_indices] = False

        return mask
