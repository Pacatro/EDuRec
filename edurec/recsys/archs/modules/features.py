"""Tabular feature encoders shared by user profiles and interaction context."""

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn


def clamp_user_ids(user_ids: torch.Tensor, num_users: int) -> torch.Tensor:
    """Clamp user identifiers to valid rows of the per-user feature tables."""
    ids = user_ids.clamp(min=0)
    return ids.clamp(max=max(num_users - 1, 0))


class TabularEncoder(nn.Module):
    """Encode tabular features by summing one embedding per column.

    Dense columns go through a linear projection and every categorical column
    gets its own embedding table. ``OrdinalEncoder`` codes are shifted by one so
    that the unknown value ``-1`` maps to the padding row. An optional
    identifier embedding adds collaborative signal.

    The block works on any input whose last dimension is the feature dimension,
    so the same module encodes per-user profiles ``[batch, features]`` and
    per-event context ``[batch, seq_len, features]``.
    """

    def __init__(
        self,
        emb_dim: int,
        dense_dim: int = 0,
        cat_cardinalities: Sequence[int] = (),
        num_ids: int = 0,
    ):
        super().__init__()
        self.dense_proj = nn.Linear(dense_dim, emb_dim) if dense_dim > 0 else None
        self.cat_embs = nn.ModuleList(
            nn.Embedding(cardinality, emb_dim, padding_idx=0)
            for cardinality in cat_cardinalities
        )
        self.id_emb = nn.Embedding(num_ids, emb_dim) if num_ids > 0 else None
        self.norm = nn.LayerNorm(emb_dim)

    @property
    def is_active(self) -> bool:
        """Whether there is any feature block to encode."""
        return (
            self.dense_proj is not None
            or len(self.cat_embs) > 0
            or self.id_emb is not None
        )

    def forward(
        self,
        dense: torch.Tensor | None,
        cat: torch.Tensor | None,
        ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return representations with shape ``[..., emb_dim]``.

        Args:
            dense: Dense columns with shape ``[..., dense_dim]``.
            cat: Ordinal categorical codes with shape ``[..., num_cat]``, where
                ``-1`` marks an unknown value.
            ids: Optional ``[...,]`` identifiers for the identifier embedding.
        """
        if not self.is_active:
            raise RuntimeError("TabularEncoder has no feature block to encode.")

        parts: list[torch.Tensor] = []
        if self.dense_proj is not None:
            if dense is None:
                raise ValueError("Missing dense features for the tabular encoder.")
            parts.append(self.dense_proj(dense))

        for idx, module in enumerate(self.cat_embs):
            if cat is None:
                raise ValueError("Missing categorical features for the tabular encoder.")
            embedding = cast(nn.Embedding, module)
            codes = (cat[..., idx] + 1).clamp(min=0, max=embedding.num_embeddings - 1)
            parts.append(embedding(codes))

        if self.id_emb is not None:
            if ids is None:
                raise ValueError("Missing identifiers for the tabular encoder.")
            parts.append(self.id_emb(ids))

        return self.norm(torch.stack(parts, dim=0).sum(dim=0))


class UserStateEncoder(nn.Module):
    """Build the initial recurrent state for a user from their static profile.

    The static profile is encoded by :class:`TabularEncoder` and projected into
    ``num_layers`` recurrent states. When no profile feature is available a
    learned per-layer state is broadcast to the batch as a fallback.
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        num_dense_feats: int = 0,
        cat_cardinalities: Sequence[int] = (),
        num_users: int = 0,
        use_id_embedding: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_users = num_users

        self.profile = (
            TabularEncoder(
                emb_dim=emb_dim,
                dense_dim=num_dense_feats,
                cat_cardinalities=cat_cardinalities,
                num_ids=num_users if use_id_embedding else 0,
            )
            if num_dense_feats > 0
            or len(cat_cardinalities) > 0
            or use_id_embedding
            else None
        )
        self.to_state = (
            nn.Linear(emb_dim, hidden_dim * num_layers)
            if self.profile is not None
            else None
        )
        self.learned_state = nn.Parameter(torch.zeros(num_layers, hidden_dim))

    def forward(
        self,
        cat_feats: torch.Tensor,
        dense_feats: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return the ``[num_layers, batch_size, hidden_dim]`` initial state."""
        batch_size = user_ids.size(0)
        if self.profile is None or self.to_state is None:
            return (
                self.learned_state.unsqueeze(1).expand(-1, batch_size, -1).contiguous()
            )

        ids = clamp_user_ids(user_ids, self.num_users)
        profile = self.profile(dense_feats[ids], cat_feats[ids], ids=ids)
        state = self.to_state(profile).view(
            batch_size, self.num_layers, self.hidden_dim
        )
        return state.permute(1, 0, 2).contiguous()
