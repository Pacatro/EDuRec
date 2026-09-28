from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .... import settings


@dataclass
class SeqEncoderConfig:
    emb_dim: int
    hidden_dim: int = settings.GRU_HIDDEN_DIM
    num_layers: int = settings.GRU_LAYERS
    dropout: float = 0.1
    cell_type: Literal["gru", "lstm"] = settings.SEQ_CELL
    condition_dim: int = 0


class SeqEncoder(nn.Module):
    """Recurrent encoder over a user's chronological item history.

    Supports GRU and LSTM cells, selected via ``cfg.cell_type``.
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
        self.state_proj = (
            nn.Linear(cfg.condition_dim, cfg.hidden_dim * cfg.num_layers)
            if cfg.condition_dim > 0
            else None
        )
        self.proj = nn.Linear(cfg.hidden_dim, cfg.emb_dim)
        self.norm = nn.LayerNorm(cfg.emb_dim)

    def _initial_hidden(
        self,
        condition: torch.Tensor,
        order: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.state_proj is not None
        state = self.state_proj(condition)[order]
        state = state.view(-1, self.num_layers, self.hidden_dim).permute(1, 0, 2)
        state = state.contiguous()
        if self.cell_type == "lstm":
            return state, torch.zeros_like(state)
        return state

    def forward(
        self,
        history_emb: torch.Tensor,
        history_mask: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a user interaction history.

        Args:
            history_emb: Item representations with shape
                ``[batch_size, history_len, emb_dim]``.
            history_mask: Boolean mask with shape
                ``[batch_size, history_len]``. True values indicate valid
                interactions.
            condition: Optional ``[batch_size, condition_dim]`` profile used to
                initialise the recurrent hidden state.
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
        if self.state_proj is not None and condition is not None:
            _, hidden = self.rnn(packed, self._initial_hidden(condition, order))
        else:
            _, hidden = self.rnn(packed)

        # LSTM returns ``(h_n, c_n)`` while GRU returns ``h_n`` directly.
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        # `hidden` follows the sorted batch order, so restore the input order.
        sorted_last = hidden[-1]
        last = sorted_last.new_empty(sorted_last.shape)
        last[order] = sorted_last

        seq_user_emb = self.norm(self.proj(last))
        return seq_user_emb * has_history.unsqueeze(-1)
