from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RankerConfig:
    """
    Configuration parameters for the Ranker module.

    Attributes:
        dim_model (int): The number of expected features in the input and output
            of the transformer and scorer.
        n_heads (int): Number of heads in the multi-head attention mechanism.
        n_blocks (int): Number of sub-encoder-layers in the transformer encoder.
        ff_dim (int): Dimension of the feedforward network model.
        dropout (float): Dropout value applied to the transformer and scorer layers.
        max_history_len (int): Maximum length of the user's interaction history.
    """

    dim_model: int
    n_heads: int
    n_blocks: int
    ff_dim: int
    dropout: float
    max_history_len: int


# TODO: Implement candidate isolation
# En lugar de seq = [token_u, token_i, token_c] usar seq = [token_u, token_i, token_c1, token_c2, ..., token_cK]
class Ranker(nn.Module):
    """
    Transformer-based Ranker for candidate item scoring.

    This module processes a sequence consisting of a user embedding, historical
    item embeddings, and a candidate item embedding. It uses a Transformer
    Encoder to model the interactions within this sequence and outputs a score
    representing the likelihood of interaction with the candidate item.

    The input sequence layout is typically: [User, Hist_1, ..., Hist_L, Candidate].
    """

    def __init__(self, cfg: RankerConfig):
        super().__init__()
        self.cfg = cfg

        # user token + hist_token + item tokens + candidate tokens
        max_len = cfg.max_history_len + 2
        self.pos_enc = nn.Embedding(max_len, cfg.dim_model)

        self.in_dropout = nn.Dropout(cfg.dropout)
        self.layer_norm = nn.LayerNorm(cfg.dim_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.dim_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.transformer_enc = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_blocks,
        )

        self.final_norm = nn.LayerNorm(cfg.dim_model)
        self.scorer = Scorer(cfg.dim_model, cfg.dropout)

    def forward(
        self,
        token_u: torch.Tensor,  # [B, 1, D]
        tokens_i: torch.Tensor,  # [B, L, D]
        tokens_c: torch.Tensor,  # [B, K, D]
        hist_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, D = tokens_i.shape

        if tokens_c.ndim == 2:
            tokens_c = tokens_c.unsqueeze(1)

        _, K, D = tokens_c.shape

        # Repeat context tokend for each candidate: [B, K, ...] --> [B*K, ...]
        # Now, for each example in the batch, we generate K different sequences
        # [u, h1, ..., hL, c1]
        # [u, h1, ..., hL, c2]
        # ...
        # [u, h1, ..., hL, cK]
        token_u = token_u.unsqueeze(1).expand(-1, K, -1, -1)
        tokens_i = tokens_i.unsqueeze(1).expand(-1, K, -1, -1)
        tokens_c = tokens_c.unsqueeze(2)

        seq = torch.cat([token_u, tokens_i, tokens_c], dim=2)
        # L + 2 because we add the user token and the candidate token
        seq = seq.reshape(B * K, L + 2, D)  # [B*K, L+2, D]

        positions = torch.arange(
            L + 2,
            device=seq.device,
            dtype=torch.long,
        ).unsqueeze(0)

        seq = self.in_dropout(seq + self.pos_enc(positions))  # [B*K, L+2, D]

        src_key_padding_mask = None

        if hist_mask is not None:
            hist_mask = ~hist_mask
            special_tokens = torch.zeros(
                B, 2, dtype=torch.bool, device=hist_mask.device
            )
            src_key_padding_mask = torch.cat(
                [special_tokens[:, :1], hist_mask, special_tokens[:, 1:]], dim=1
            )
            src_key_padding_mask = (
                src_key_padding_mask.unsqueeze(1)
                .expand(-1, K, -1)
                .reshape(B * K, L + 2)
            )

        seq = self.layer_norm(seq)
        hidden = self.transformer_enc(seq, src_key_padding_mask=src_key_padding_mask)
        hidden = self.final_norm(hidden)

        h_candidate = hidden[:, -1, :]  # [B*K, D]
        scores = self.scorer(h_candidate).reshape(B, K)  # [B, K]

        return scores.squeeze(1) if K == 1 else scores


class Scorer(nn.Module):
    def __init__(self, dim_model: int, dropout: float):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model, 1),
        )
        # self.scorer = nn.Linear(dim_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scorer(x)
