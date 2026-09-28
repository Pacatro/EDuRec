from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .... import settings


@dataclass
class SeqEncoderConfig:
    emb_dim: int
    hidden_dim: int = settings.GRU_HIDDEN_DIM
    num_layers: int = settings.GRU_LAYERS
    dropout: float = 0.1
    cell_type: Literal["gru", "lstm"] = settings.SEQ_CELL


class SeqEncoder(nn.Module):
    """Recurrent encoder over a user's chronological item history.

    Supports GRU and LSTM cells, selected via ``cfg.cell_type``. The recurrent
    network keeps one state per interaction; the last state ``h_T`` is combined
    with an attention-pooled summary ``h_att`` so relevant earlier interactions
    remain reachable instead of being condensed into a single state. The static
    user profile is deliberately not consumed here: it is fused afterwards.
    """

    def __init__(self, cfg: SeqEncoderConfig):
        super().__init__()
        self.cell_type = cfg.cell_type
        self.hidden_dim = cfg.hidden_dim
        self.num_layers = cfg.num_layers

        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[cfg.cell_type]
        self.rnn = rnn_cls(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.proj = nn.Linear(cfg.hidden_dim * 2, cfg.emb_dim)
        self.norm = nn.LayerNorm(cfg.emb_dim)

    def _pool(
        self,
        outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attention-pool valid states: ``sum_t alpha_t h_t``."""
        logits = self.attention(outputs).squeeze(-1)
        # A large negative (not ``-inf``) keeps softmax finite for empty rows.
        logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=1)
        return torch.einsum("bl,blh->bh", weights, outputs)

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
        outputs, hidden = self.rnn(packed)

        # LSTM returns ``(h_n, c_n)`` while GRU returns ``h_n`` directly.
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        # ``hidden[-1]`` is the last valid state of every row (sorted order).
        last = hidden[-1]
        padded_outputs, _ = pad_packed_sequence(
            outputs,
            batch_first=True,
            total_length=history_emb.size(1),
        )
        pooled = self._pool(padded_outputs, history_mask[order])

        combined = torch.cat([last, pooled], dim=-1)
        seq_user_emb = self.proj(combined)

        # Restore the original (unsorted) batch order.
        restored = seq_user_emb.new_empty(seq_user_emb.shape)
        restored[order] = seq_user_emb
        seq_user_emb = self.norm(restored)

        return seq_user_emb * has_history.unsqueeze(-1)
