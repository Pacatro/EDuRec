from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ... import settings


@dataclass
class SeqEncoderConfig:
    emb_dim: int
    hidden_dim: int = settings.GRU_HIDDEN_DIM
    num_layers: int = settings.GRU_LAYERS
    dropout: float = 0.1
    max_history_len: int = settings.MAX_HISTORY_LEN


class SeqEncoder(nn.Module):
    """GRU encoder over a user's chronological item history."""

    def __init__(self, cfg: SeqEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.gru = nn.GRU(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(cfg.hidden_dim, cfg.emb_dim)
        self.norm = nn.LayerNorm(cfg.emb_dim)

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
        lengths = history_mask.sum(dim=1)
        has_history = lengths > 0

        # pack_padded_sequence requires at least one step per row, so empty
        # histories are encoded with a length of one and masked out afterwards.
        safe_lengths = lengths.clamp(min=1)
        sorted_lengths, order = safe_lengths.sort(descending=True)

        packed = pack_padded_sequence(
            history_emb[order],
            sorted_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        _, hidden = self.gru(packed)

        # `hidden` follows the sorted batch order, so restore the input order.
        sorted_last = hidden[-1]
        last = sorted_last.new_empty(sorted_last.shape)
        last[order] = sorted_last

        seq_user_emb = self.norm(self.proj(last))
        return seq_user_emb * has_history.unsqueeze(-1)
