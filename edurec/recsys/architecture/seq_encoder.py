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


class SeqEncoder(nn.Module):
    def __init__(self, cfg: SeqEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.pos_emb = nn.Embedding(cfg.max_history_len, cfg.emb_dim)
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
    ) -> torch.Tensor:
        """Encode a user interaction history.

        Args:
            history_emb: Item representations with shape
                ``[batch_size, history_len, emb_dim]``.
            history_mask: Boolean mask with shape
                ``[batch_size, history_len]``. True values indicate valid
                interactions.
        Returns:
            Sequential user state with shape ``[batch_size, emb_dim]``.
            Users without history receive a zero vector.
        """
        if history_emb.ndim != 3:
            raise ValueError(
                "history_emb must have shape [batch_size, history_len, emb_dim]."
            )

        batch_size, history_len, emb_dim = history_emb.shape

        if emb_dim != self.cfg.emb_dim:
            raise ValueError(
                f"Expected embedding dimension {self.cfg.emb_dim}, got {emb_dim}."
            )

        if history_len > self.cfg.max_history_len:
            raise ValueError(
                f"History length {history_len} exceeds "
                f"max_history_len={self.cfg.max_history_len}."
            )

        if history_mask.shape != (batch_size, history_len):
            raise ValueError(
                f"history_mask must have shape [{batch_size}, {history_len}]."
            )

        history_mask = history_mask.bool()
        has_history = history_mask.any(dim=1)

        # PyTorch attention cannot process a row where every token is masked.
        safe_mask = history_mask.clone()
        safe_mask[~has_history, 0] = True

        # Positions 0, 1, ..., L - 1 for valid interactions, independent
        # of whether padding is placed on the left or right.
        position_ids = history_mask.long().cumsum(dim=1) - 1
        position_ids = position_ids.clamp(min=0, max=self.cfg.max_history_len - 1)

        tokens = history_emb + self.pos_emb(position_ids)

        tokens = self.input_norm(tokens)
        tokens = tokens.masked_fill(~safe_mask.unsqueeze(-1), 0.0)

        causal_mask = torch.triu(
            torch.ones(
                history_len,
                history_len,
                dtype=torch.bool,
                device=history_emb.device,
            ),
            diagonal=1,
        )

        encoded = self.transformer(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=~safe_mask,
        )

        # Find the actual final valid position instead of assuming right padding.
        sequence_positions = torch.arange(history_len, device=history_emb.device)
        sequence_positions = sequence_positions.unsqueeze(0).expand(batch_size, -1)

        last_indices = sequence_positions.masked_fill(~history_mask, -1).amax(dim=1)
        last_indices = last_indices.clamp_min(0)

        seq_user_emb = encoded[
            torch.arange(batch_size, device=encoded.device),
            last_indices,
        ]

        # Zero is appropriate here as long as the fusion layer also receives
        # an explicit availability mask for the sequence source.
        return seq_user_emb * has_history.unsqueeze(-1)
