from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

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
    use_attention_pooling: bool = False


class SeqEncoder(nn.Module):
    """Recurrent encoder over a user's chronological item history.

    Supports GRU and LSTM cells, selected via ``cfg.cell_type``. The history is
    run through the recurrent network over the full padded sequence and the
    last valid step is read off as the user state.
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
        self.pool = (
            AttentionPooling(cfg.hidden_dim) if cfg.use_attention_pooling else None
        )
        self.proj = nn.Linear(cfg.hidden_dim, cfg.emb_dim)
        self.norm = nn.LayerNorm(cfg.emb_dim)

    def forward(
        self,
        event_emb: torch.Tensor,
        history_mask: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a user interaction history.

        Args:
            event_emb: Item representations with shape
                ``[batch_size, history_len, emb_dim]``.
            history_mask: Boolean mask with shape
                ``[batch_size, history_len]``. True values indicate valid
                interactions; right padding is ignored.
            initial_state: Optional ``[num_layers, batch_size, hidden_dim]``
                recurrent state used to start the sequence.
        Returns:
            Sequential user state with shape ``[batch_size, emb_dim]``.
            Users without history receive a zero vector.
        """
        mask = history_mask.bool()
        lengths = mask.sum(dim=1)
        has_history = lengths > 0

        # Zero the right padding before the RNN so the placeholder steps do not
        # leak into the running state of later valid steps.
        event_emb = event_emb * mask.unsqueeze(-1).to(event_emb.dtype)

        batch_size = event_emb.size(0)
        if initial_state is None:
            hidden = event_emb.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        else:
            hidden = initial_state

        # Right padding does not affect earlier valid steps, so the full padded
        # sequence can be run without packing.
        if self.cell_type == "lstm":
            cell = event_emb.new_zeros(self.num_layers, batch_size, self.hidden_dim)
            outputs, _ = self.rnn(event_emb, (hidden, cell))
        else:
            outputs, _ = self.rnn(event_emb, hidden)

        if self.pool is not None:
            pooled = self.pool(outputs, mask)
        else:
            # Read the last valid hidden state of each row.
            last = (lengths - 1).clamp(min=0)
            pooled = outputs[torch.arange(batch_size, device=outputs.device), last]

        user = self.norm(self.proj(pooled)) * has_history.unsqueeze(-1)
        return user


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
    causal: bool = False
    use_last_state: bool = False


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
        self.causal = cfg.causal
        self.use_last_state = cfg.use_last_state

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

        if self.causal and seq_len > self.max_history_len:
            # Dataset histories are right-padded: keep each row's latest valid
            # courses, rather than slicing off short histories with the padding.
            start = (history_mask.sum(1) - self.max_history_len).clamp(min=0)
            indices = start[:, None] + torch.arange(
                self.max_history_len, device=history_emb.device
            )
            history_emb = history_emb.gather(
                1, indices.unsqueeze(-1).expand(-1, -1, history_emb.size(-1))
            )
            history_mask = history_mask.gather(1, indices)
            seq_len = history_emb.size(1)
        if seq_len == 0:
            state = history_emb.new_zeros(history_emb.size(0), self.proj.out_features)
            return (state, state.unsqueeze(1)[:, :0]) if return_sequence else state

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

        causal_mask = (
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(1)
            if self.causal
            else None
        )
        encoded = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        if self.use_last_state:
            positions = torch.arange(seq_len, device=x.device)
            last = (
                positions.expand_as(history_mask).masked_fill(~history_mask, 0).amax(1)
            )
            pooled = encoded[torch.arange(x.size(0), device=x.device), last]
        else:
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
