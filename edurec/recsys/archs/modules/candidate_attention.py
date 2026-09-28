from dataclasses import dataclass

import torch
from torch import nn

from .... import settings
from .history_transformer import safe_valid_mask


@dataclass
class CandidateAttentionConfig:
    emb_dim: int
    num_heads: int = settings.TRANSFORMER_HEADS
    dropout: float = settings.DROPOUT


class CandidateAttention(nn.Module):
    """Attend candidate items to the encoded user history.

    Each candidate acts as a query over the history memory, so its refined
    representation carries the parts of the user's past that are relevant to it.
    The output is a residual connection around a multi-head cross-attention
    block followed by layer normalisation.
    """

    def __init__(self, cfg: CandidateAttentionConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            cfg.emb_dim,
            cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cfg.emb_dim)

    def forward(
        self,
        candidate_emb: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Refine candidates with history context.

        Args:
            candidate_emb: Candidate representations ``[batch, candidates, dim]``.
            memory: Encoded history ``[batch, history_len, dim]``.
            memory_mask: Boolean valid mask ``[batch, history_len]``.
        Returns:
            Context-aware candidate representations with the same shape as
            ``candidate_emb``.
        """
        if candidate_emb.size(1) == 0 or memory.size(1) == 0:
            return candidate_emb

        safe_mask = safe_valid_mask(memory_mask.bool())
        attended, _ = self.attn(
            candidate_emb,
            memory,
            memory,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        return self.norm(candidate_emb + attended)
