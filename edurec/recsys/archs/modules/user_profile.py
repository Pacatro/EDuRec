from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn


@dataclass
class UserProfileConfig:
    emb_dim: int
    num_dense_feats: int = 0
    cat_cardinalities: list[int] = field(default_factory=list)
    num_users: int = 0
    use_id_embedding: bool = False

    @property
    def is_active(self) -> bool:
        """Whether there is any user signal to encode."""
        return (
            self.num_dense_feats > 0
            or len(self.cat_cardinalities) > 0
            or self.use_id_embedding
        )


class UserProfileEncoder(nn.Module):
    """Encode a static user profile into a single representation.

    Every categorical user field gets its own embedding table. The
    ``OrdinalEncoder`` codes are shifted by one so that the unknown value ``-1``
    maps to the padding row. The numeric/text block is projected with a linear
    layer, and an optional user-identifier embedding captures collaborative
    signal. All contributions are summed and normalised.
    """

    def __init__(self, cfg: UserProfileConfig):
        super().__init__()
        self.cfg = cfg

        self.cat_embs = nn.ModuleList(
            nn.Embedding(cardinality, cfg.emb_dim, padding_idx=0)
            for cardinality in cfg.cat_cardinalities
        )
        self.dense_proj = (
            nn.Linear(cfg.num_dense_feats, cfg.emb_dim)
            if cfg.num_dense_feats > 0
            else None
        )
        self.id_emb = (
            nn.Embedding(cfg.num_users, cfg.emb_dim)
            if cfg.use_id_embedding
            else None
        )
        self.norm = nn.LayerNorm(cfg.emb_dim)

    def forward(
        self,
        cat_feats: torch.Tensor,
        dense_feats: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Build ``[batch, emb_dim]`` user profile representations.

        The feature tables hold one row per user and are gathered with
        ``user_ids``. Returns ``None`` when no user feature is configured, so
        callers can skip the profile path entirely.
        """
        if (
            len(self.cat_embs) == 0
            and self.dense_proj is None
            and self.id_emb is None
        ):
            return None

        batch_ids = user_ids.clamp(min=0)
        if self.cfg.num_users > 0:
            batch_ids = batch_ids.clamp(max=self.cfg.num_users - 1)

        parts: list[torch.Tensor] = []
        if len(self.cat_embs) > 0:
            batch_cat = cat_feats[batch_ids]
            for idx, module in enumerate(self.cat_embs):
                embedding = cast(nn.Embedding, module)
                codes = (batch_cat[:, idx] + 1).clamp(
                    min=0, max=embedding.num_embeddings - 1
                )
                parts.append(embedding(codes))

        if self.dense_proj is not None:
            parts.append(self.dense_proj(dense_feats[batch_ids]))

        if self.id_emb is not None:
            parts.append(self.id_emb(batch_ids))

        return self.norm(torch.stack(parts, dim=0).sum(dim=0))
