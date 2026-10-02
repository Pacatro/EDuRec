from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from edurec import settings
from edurec.recsys.archs.modules.attn_pooling import (
    AttentionPooling,
    masked_mean_pool,
)


@dataclass
class SeqEncoderConfig:
    emb_dim: int
    hidden_dim: int = settings.GRU_HIDDEN_DIM
    num_layers: int = settings.GRU_LAYERS
    dropout: float = 0.1
    cell_type: Literal["gru", "lstm"] = settings.SEQ_CELL
    condition_dim: int = 0
    use_attention_pooling: bool = True


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
        self.pool = (
            AttentionPooling(cfg.hidden_dim) if cfg.use_attention_pooling else None
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
        return_sequence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
            packed_out, _ = self.rnn(packed, self._initial_hidden(condition, order))
        else:
            packed_out, _ = self.rnn(packed)

        outputs, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=history_emb.size(1)
        )

        # ``outputs`` follows the sorted batch order. Pool every recurrent
        # state while masking out the padding steps.
        step_mask = torch.arange(outputs.size(1), device=outputs.device)[
            None, :
        ] < sorted_lengths.unsqueeze(1)
        sorted_pooled = (
            self.pool(outputs, step_mask)
            if self.pool is not None
            else masked_mean_pool(outputs, step_mask)
        )

        # Restore the input batch order.
        pooled = sorted_pooled.new_empty(sorted_pooled.shape)
        pooled[order] = sorted_pooled

        seq_user_emb = self.norm(self.proj(pooled))
        seq_user_emb = seq_user_emb * has_history.unsqueeze(-1)
        if not return_sequence:
            return seq_user_emb

        sequence = outputs.new_empty(
            (outputs.size(0), outputs.size(1), outputs.size(2))
        )
        sequence[order] = outputs
        sequence = self.norm(self.proj(sequence))
        return seq_user_emb, sequence


@dataclass
class TransformerSeqEncoderConfig:
    emb_dim: int
    hidden_dim: int = settings.TRANSFORMER_HIDDEN_DIM
    num_layers: int = settings.TRANSFORMER_LAYERS
    num_heads: int = settings.TRANSFORMER_HEADS
    dropout: float = 0.1
    condition_dim: int = 0
    max_history_len: int = settings.MAX_HISTORY_LEN
    use_attention_pooling: bool = True


class TransformerSeqEncoder(nn.Module):
    """Transformer encoder over a user's chronological item history.

    A learned positional embedding marks the interaction order and a stack of
    pre-norm Transformer blocks mixes items through self-attention. The
    resulting states are attention-pooled into a single user vector.
    """

    def __init__(self, cfg: TransformerSeqEncoderConfig):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.max_history_len = cfg.max_history_len

        self.input_proj = nn.Linear(cfg.emb_dim, cfg.hidden_dim)
        self.pos_emb = nn.Embedding(cfg.max_history_len, cfg.hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=cfg.num_layers,
            norm=nn.LayerNorm(cfg.hidden_dim),
            enable_nested_tensor=False,
        )
        self.condition_proj = (
            nn.Linear(cfg.condition_dim, cfg.hidden_dim)
            if cfg.condition_dim > 0
            else None
        )
        self.pool = (
            AttentionPooling(cfg.hidden_dim) if cfg.use_attention_pooling else None
        )
        self.proj = nn.Linear(cfg.hidden_dim, cfg.emb_dim)
        self.norm = nn.LayerNorm(cfg.emb_dim)

    def forward(
        self,
        history_emb: torch.Tensor,
        history_mask: torch.Tensor,
        condition: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a user interaction history.

        Args:
            history_emb: Item representations with shape
                ``[batch_size, history_len, emb_dim]``.
            history_mask: Boolean mask with shape
                ``[batch_size, history_len]``. True values indicate valid
                interactions.
            condition: Optional ``[batch_size, condition_dim]`` profile added to
                every step before self-attention.
        Returns:
            Sequential user state with shape ``[batch_size, emb_dim]``.
            Users without history receive a zero vector.
        """
        history_mask = history_mask.bool()
        seq_len = history_emb.size(1)

        x = self.input_proj(history_emb)
        positions = torch.arange(seq_len, device=x.device).clamp(
            max=self.max_history_len - 1
        )
        x = x + self.pos_emb(positions)
        if self.condition_proj is not None and condition is not None:
            x = x + self.condition_proj(condition).unsqueeze(1)

        # Transformer layers treat True entries as padding and skip them.
        padding_mask = ~history_mask
        # Fully padded rows would produce NaN attention, so allow them to attend
        # to their placeholder steps and mask the pooled vector out afterwards.
        empty = ~history_mask.any(dim=1)
        padding_mask = padding_mask.masked_fill(empty.unsqueeze(1), False)

        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        pooled = (
            self.pool(encoded, history_mask | empty.unsqueeze(1))
            if self.pool is not None
            else masked_mean_pool(encoded, history_mask)
        )

        seq_user_emb = self.norm(self.proj(pooled))
        seq_user_emb = seq_user_emb * (~empty).unsqueeze(-1)
        if return_sequence:
            sequence = self.norm(self.proj(encoded))
            return seq_user_emb, sequence
        return seq_user_emb
