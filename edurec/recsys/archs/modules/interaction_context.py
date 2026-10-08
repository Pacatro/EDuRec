from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn


@dataclass
class ContextConfig:
    emb_dim: int
    dense_dim: int = 0
    cat_cardinalities: list[int] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        """Whether there is any interaction context signal to encode."""
        return self.dense_dim > 0 or len(self.cat_cardinalities) > 0


class ContextEncoder(nn.Module):
    """Encode per-interaction context features into a sequence representation.

    Dense context columns are projected with a linear layer and every
    categorical column gets its own embedding table. The ``OrdinalEncoder``
    codes are shifted by one so that the unknown value ``-1`` maps to the
    padding row. All contributions are summed and normalised, mirroring
    :class:`~edurec.recsys.archs.modules.user_state.UserProfileEncoder`.
    """

    def __init__(self, cfg: ContextConfig):
        super().__init__()
        self.cfg = cfg

        self.dense_proj = (
            nn.Linear(cfg.dense_dim, cfg.emb_dim) if cfg.dense_dim > 0 else None
        )
        self.cat_embs = nn.ModuleList(
            nn.Embedding(cardinality, cfg.emb_dim, padding_idx=0)
            for cardinality in cfg.cat_cardinalities
        )
        self.norm = nn.LayerNorm(cfg.emb_dim)

    def forward(
        self,
        dense: torch.Tensor,
        cat: torch.Tensor,
    ) -> torch.Tensor | None:
        """Build ``[batch, seq_len, emb_dim]`` context representations.

        Args:
            dense: Dense context values with shape ``[batch, seq_len, dense_dim]``.
            cat: Ordinal categorical codes with shape ``[batch, seq_len, C]``,
                where ``-1`` marks an unknown value.
        Returns:
            The summed, normalised context with shape
            ``[batch, seq_len, emb_dim]``, or ``None`` when no context feature is
            configured so callers can skip the path entirely.
        """
        if self.dense_proj is None and len(self.cat_embs) == 0:
            return None

        parts: list[torch.Tensor] = []
        if self.dense_proj is not None:
            parts.append(self.dense_proj(dense))

        for idx, module in enumerate(self.cat_embs):
            embedding = cast(nn.Embedding, module)
            codes = (cat[..., idx] + 1).clamp(min=0, max=embedding.num_embeddings - 1)
            parts.append(embedding(codes))

        return self.norm(torch.stack(parts, dim=0).sum(dim=0))
