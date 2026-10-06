from dataclasses import dataclass, field

import torch
from torch import nn

from edurec.recsys.archs.modules.user_profile import (
    UserProfileConfig,
    UserProfileEncoder,
)


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


class UserStateEncoder(nn.Module):
    """Build the initial recurrent state for a user from their static profile.

    The static profile is encoded by :class:`UserProfileEncoder` and projected
    into ``num_layers`` recurrent states. When no profile feature is available a
    learned per-layer state is broadcast to the batch as a fallback.
    """

    def __init__(self, cfg: UserStateConfig):
        super().__init__()
        self.cfg = cfg

        profile_cfg = UserProfileConfig(
            emb_dim=cfg.emb_dim,
            num_dense_feats=cfg.num_dense_feats,
            cat_cardinalities=cfg.cat_cardinalities,
            num_users=cfg.num_users,
            use_id_embedding=cfg.use_id_embedding,
        )
        self.profile: UserProfileEncoder | None = None
        self.to_state: nn.Linear | None = None
        if profile_cfg.is_active:
            self.profile = UserProfileEncoder(profile_cfg)
            self.to_state = nn.Linear(
                cfg.emb_dim, cfg.hidden_dim * cfg.num_layers
            )

        self.learned_state = nn.Parameter(
            torch.zeros(cfg.num_layers, cfg.hidden_dim)
        )

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
                self.learned_state.unsqueeze(1)
                .expand(-1, batch_size, -1)
                .contiguous()
            )

        profile = self.profile(cat_feats, dense_feats, user_ids)
        assert profile is not None
        assert self.to_state is not None
        state = self.to_state(profile).view(
            batch_size, self.cfg.num_layers, self.cfg.hidden_dim
        )
        return state.permute(1, 0, 2).contiguous()
