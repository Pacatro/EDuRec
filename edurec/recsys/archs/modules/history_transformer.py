import math
from dataclasses import dataclass

import torch
from torch import nn

from .... import settings


@dataclass
class HistoryTransformerConfig:
    emb_dim: int
    num_heads: int = settings.TRANSFORMER_HEADS
    num_layers: int = settings.TRANSFORMER_LAYERS
    dim_feedforward: int = settings.TRANSFORMER_FF_DIM
    dropout: float = settings.DROPOUT
    max_len: int = settings.MAX_HISTORY_LEN


def safe_valid_mask(mask: torch.Tensor) -> torch.Tensor:
    """Guarantee at least one valid position per row.

    Fully masked rows would otherwise make the attention softmax produce NaNs.
    The padded position is kept only to stabilise the forward pass; callers must
    still zero the output of rows without real history.
    """
    valid = mask.any(dim=1, keepdim=True)
    return mask | ~valid


class HistoryTransformer(nn.Module):
    """Transformer encoder over a user's chronological item history.

    Item embeddings are summed with a learned positional embedding and encoded
    by a stack of self-attention blocks. The valid positions are attention-pooled
    into a global user representation, while the full encoded sequence is kept as
    memory for candidate attention.
    """

    def __init__(self, cfg: HistoryTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.pos_emb = nn.Embedding(cfg.max_len, cfg.emb_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.emb_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
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
        self.norm = nn.LayerNorm(cfg.emb_dim)
        self.pool_query = nn.Parameter(torch.zeros(cfg.emb_dim))

    def forward(
        self,
        history_emb: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a history batch.

        Args:
            history_emb: Item representations with shape
                ``[batch_size, history_len, emb_dim]``.
            history_mask: Boolean mask with shape
                ``[batch_size, history_len]``. True values indicate valid
                interactions.
        Returns:
            ``(global_user, memory)`` with shapes ``[batch_size, emb_dim]`` and
            ``[batch_size, history_len, emb_dim]``. Users without history
            receive a zero global representation.
        """
        mask = history_mask.bool()
        batch_size, length, emb_dim = history_emb.shape
        if length == 0:
            empty = history_emb.new_zeros(batch_size, emb_dim)
            return empty, history_emb.new_zeros(batch_size, 0, emb_dim)

        has_history = mask.any(dim=1)
        safe_mask = safe_valid_mask(mask)

        positions = torch.arange(length, device=history_emb.device)
        positions = positions.clamp(max=self.cfg.max_len - 1)
        x = history_emb + self.pos_emb(positions).unsqueeze(0)

        memory = self.encoder(x, src_key_padding_mask=~safe_mask)

        pool_logits = memory @ self.pool_query / math.sqrt(emb_dim)
        pool_logits = pool_logits.masked_fill(
            ~safe_mask, torch.finfo(pool_logits.dtype).min
        )
        weights = torch.softmax(pool_logits, dim=1)
        pooled = (weights.unsqueeze(-1) * memory).sum(dim=1)

        global_user = self.norm(pooled) * has_history.unsqueeze(-1)
        return global_user, memory
