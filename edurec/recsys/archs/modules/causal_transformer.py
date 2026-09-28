import math
from dataclasses import dataclass

import torch
from torch import nn

from .... import settings


@dataclass
class CausalTransformerConfig:
    emb_dim: int
    num_layers: int = settings.TRANSFORMER_LAYERS
    num_heads: int = settings.TRANSFORMER_HEADS
    ffn_dim: int = settings.TRANSFORMER_FFN_DIM
    dropout: float = settings.DROPOUT
    max_len: int = settings.MAX_HISTORY_LEN
    use_temporal: bool = True
    num_time_buckets: int = settings.TIME_EMBED_BUCKETS


class TemporalEmbedding(nn.Module):
    """Learned embedding over log-spaced buckets of the elapsed time.

    Time deltas span several orders of magnitude (seconds to years), so the raw
    value is compressed with ``log1p`` and discretised into ``num_buckets``
    log-spaced bins. Padding slots carry a zero delta and are never attended to.
    """

    def __init__(
        self,
        emb_dim: int,
        num_buckets: int = settings.TIME_EMBED_BUCKETS,
        max_seconds: float = 1e10,
    ):
        super().__init__()
        boundaries = torch.linspace(
            0.0,
            math.log1p(max_seconds),
            num_buckets - 1,
        )
        self.register_buffer("boundaries", boundaries)
        self.embedding = nn.Embedding(num_buckets, emb_dim)

    def forward(self, deltas: torch.Tensor) -> torch.Tensor:
        compressed = torch.log1p(deltas.clamp(min=0.0))
        buckets = torch.bucketize(compressed, self.boundaries)
        return self.embedding(buckets)


class CausalTransformer(nn.Module):
    """Causal transformer over a user's chronological item history.

    Item embeddings are summed with a learned positional embedding and, when
    available, a temporal embedding of the time elapsed since each interaction.
    The stack uses a causal attention mask, so each position only sees its past,
    and returns the contextual hidden states ``h1..hT``.
    """

    def __init__(self, cfg: CausalTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.max_len = cfg.max_len

        self.position_emb = nn.Embedding(cfg.max_len, cfg.emb_dim)
        self.temporal_emb = (
            TemporalEmbedding(cfg.emb_dim, cfg.num_time_buckets)
            if cfg.use_temporal
            else None
        )
        self.input_norm = nn.LayerNorm(cfg.emb_dim)
        self.input_dropout = nn.Dropout(cfg.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.emb_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=cfg.num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(cfg.emb_dim)

    def forward(
        self,
        history_emb: torch.Tensor,
        history_mask: torch.Tensor,
        history_times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a history batch.

        Args:
            history_emb: Item representations ``[batch, seq_len, emb_dim]``.
            history_mask: Boolean mask ``[batch, seq_len]``, True for valid
                interactions.
            history_times: Optional elapsed-time deltas ``[batch, seq_len]``.
        Returns:
            Contextual hidden states ``[batch, seq_len, emb_dim]``; padded
            positions are zero.
        """
        history_mask = history_mask.bool()
        seq_len = history_emb.size(1)

        positions = torch.arange(seq_len, device=history_emb.device)
        positions = positions.clamp(max=self.max_len - 1)
        x = history_emb + self.position_emb(positions).unsqueeze(0)

        if self.temporal_emb is not None and history_times is not None:
            x = x + self.temporal_emb(history_times.to(history_emb.dtype))

        x = self.input_dropout(self.input_norm(x))
        padding_mask = ~history_mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )

        hidden = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        hidden = self.output_norm(hidden)

        return hidden * history_mask.unsqueeze(-1).to(hidden.dtype)
