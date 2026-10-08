from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn


@dataclass
class UserStateConfig:
    emb_dim: int
    hidden_dim: int
    num_layers: int = 1
    num_dense_feats: int = 0
    cat_cardinalities: list[int] = field(default_factory=list)
    num_users: int = 0
    use_id_embedding: bool = False

    @property
    def is_active(self) -> bool:
        """Whether there is any static user signal to condition on."""
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

    def __init__(self, cfg: UserStateConfig):
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
            nn.Embedding(cfg.num_users, cfg.emb_dim) if cfg.use_id_embedding else None
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
        if len(self.cat_embs) == 0 and self.dense_proj is None and self.id_emb is None:
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


class UserStateEncoder(nn.Module):
    """Build the initial recurrent state for a user from their static profile.

    The static profile is encoded by :class:`UserProfileEncoder` and projected
    into ``num_layers`` recurrent states. When no profile feature is available a
    learned per-layer state is broadcast to the batch as a fallback.
    """

    def __init__(self, cfg: UserStateConfig):
        super().__init__()
        self.cfg = cfg

        self.profile: UserProfileEncoder | None = None
        self.to_state: nn.Linear | None = None
        if cfg.is_active:
            self.profile = UserProfileEncoder(cfg)
            self.to_state = nn.Linear(cfg.emb_dim, cfg.hidden_dim * cfg.num_layers)

        self.learned_state = nn.Parameter(torch.zeros(cfg.num_layers, cfg.hidden_dim))

    def forward(
        self,
        cat_feats: torch.Tensor,
        dense_feats: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return the ``[num_layers, batch_size, hidden_dim]`` initial state."""
        batch_size = user_ids.size(0)
        if self.profile is None:
            return (
                self.learned_state.unsqueeze(1).expand(-1, batch_size, -1).contiguous()
            )

        profile = self.profile(cat_feats, dense_feats, user_ids)
        assert profile is not None
        assert self.to_state is not None
        state = self.to_state(profile).view(
            batch_size, self.cfg.num_layers, self.cfg.hidden_dim
        )
        return state.permute(1, 0, 2).contiguous()
