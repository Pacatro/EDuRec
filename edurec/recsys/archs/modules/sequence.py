"""Sequential encoders over a user's chronological interaction history."""

from typing import Literal

import torch
from torch import nn

from edurec import settings


def masked_mean_pool(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Uniform mean over valid steps with shape ``[batch, dim]``.

    Args:
        sequence: Hidden states with shape ``[batch, seq_len, dim]``.
        mask: Boolean mask with shape ``[batch, seq_len]``. True values indicate
            valid steps; padded steps are ignored. Rows with no valid step
            receive a zero vector.
    """
    weights = mask.bool().unsqueeze(-1).to(sequence.dtype)
    return (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class SeqEncoder(nn.Module):
    """Recurrent encoder over a user's chronological item history.

    Supports GRU and LSTM cells, selected via ``cell_type``. The history is run
    through the recurrent network over the full padded sequence and the last
    valid step is read off as the user state.
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = settings.GRU_HIDDEN_DIM,
        num_layers: int = settings.GRU_LAYERS,
        dropout: float = 0.1,
        cell_type: Literal["gru", "lstm"] = settings.SEQ_CELL,
    ):
        super().__init__()
        self.cell_type = cell_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[cell_type]
        self.rnn = rnn_cls(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

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

        # Read the last valid hidden state of each row.
        last = (lengths - 1).clamp(min=0)
        pooled = outputs[torch.arange(batch_size, device=outputs.device), last]

        user = self.norm(self.proj(pooled)) * has_history.unsqueeze(-1)
        return user


class TransformerSeqEncoder(nn.Module):
    """Transformer encoder over a user's chronological item history.

    A learned positional embedding marks the interaction order and a stack of
    pre-norm Transformer blocks mixes items through self-attention. The
    resulting states are pooled into a single user vector.
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = settings.TRANSFORMER_HIDDEN_DIM,
        num_layers: int = settings.TRANSFORMER_LAYERS,
        num_heads: int = settings.TRANSFORMER_HEADS,
        dropout: float = 0.1,
        max_history_len: int = settings.MAX_HISTORY_LEN,
        causal: bool = False,
        use_last_state: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_history_len = max_history_len
        self.causal = causal
        self.use_last_state = use_last_state

        self.input_proj = nn.Linear(emb_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_history_len, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )
        self.proj = nn.Linear(hidden_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

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
            return history_emb.new_zeros(history_emb.size(0), self.proj.out_features)

        x = self.input_proj(history_emb)
        positions = torch.arange(seq_len, device=x.device).clamp(
            max=self.max_history_len - 1
        )
        x = x + self.pos_emb(positions)

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
            pooled = masked_mean_pool(encoded, history_mask)

        seq_user_emb = self.norm(self.proj(pooled))
        return seq_user_emb * (~empty).unsqueeze(-1)
