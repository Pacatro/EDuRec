from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RankerConfig:
    dim_model: int
    n_heads: int
    n_blocks: int
    ff_dim: int
    dropout: float
    max_history_len: int


class Scorer(nn.Module):
    def __init__(self, dim_model: int, dropout: float):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scorer(x)


class GhostRanker(nn.Module):
    def __init__(self, cfg: RankerConfig):
        super().__init__()
        self.cfg = cfg

        max_len = cfg.max_history_len + 2  # user token + item tokens + candidate tokens
        self.pos_enc = nn.Embedding(max_len, cfg.dim_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=cfg.dim_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_enc = nn.TransformerEncoder(
            encoder_layers,
            num_layers=cfg.n_blocks,
        )

        self.final_norm = nn.LayerNorm(cfg.dim_model)

        self.scorer = Scorer(cfg.dim_model, cfg.dropout)

    def forward(
        self,
        token_u: torch.Tensor,  # [B, D]
        tokens_i: torch.Tensor,  # [B, L, D]
        tokens_c: torch.Tensor,  # [B, K, D]
        hist_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, d_u = token_u.shape
        b, l, d_i = tokens_i.shape
        _, k, d_c = tokens_c.shape

        assert d_u == d_i == d_c, "All token dimensions must match d_model."

        token_u = token_u.unsqueeze(1).expand(b, k, d_i)  # [B, K, D]
        tokens_i = tokens_i.unsqueeze(1).expand(b, k, l, d_i)  # [B, K, L, D]

        # Sequence: [token_u, token_i_1, token_i_2, ..., token_i_L, token_c]
        seq = torch.cat(
            [
                token_u.unsqueeze(2),  # [B, K, 1, D]
                tokens_i,  # [B, K, L, D]
                tokens_c.unsqueeze(1),  # [B, 1, K, D]
            ],
            dim=2,
        )  # [B, K, L+2, D]

        seq = seq.reshape(b * k, l + 2, d_i)  # [B*K, L+2, D]
        seq = self.pos_enc(seq)

        if hist_mask is not None:
            user_valid = torch.ones(b, 1, dtype=torch.bool, device=hist_mask.device)
            cand_valid = torch.ones(b, 1, dtype=torch.bool, device=hist_mask.device)

            valid_mask = torch.cat(
                [user_valid, hist_mask, cand_valid], dim=1
            )  # [B, L+2]

            valid_mask = valid_mask.unsqueeze(1).expand(b, k, l + 2)  # [B, K, L+2]
            key_padding_mask = ~valid_mask.reshape(b * k, l + 2)
        else:
            key_padding_mask = None

        h = self.transformer_enc(
            seq, src_key_padding_mask=key_padding_mask
        )  # [B*K, T, D]

        h = self.final_norm(h)
        h_candidate = h[:, -1, :]  # [B*K, D]

        scores = self.scorer(h_candidate)  # [B*K, 1]

        h_candidate = h_candidate.view(b, k, d_i)
        scores = scores.view(b, k)
        return scores, h_candidate
