from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseRecArch(nn.Module, ABC):
    """Common interface for the recommendation architectures.

    Maps a history batch to item scores. Architectures may use graph encoders
    and static user profiles or encode only the sequence of course embeddings.
    """

    @abstractmethod
    def forward(
        self,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        edge_index: dict[tuple[str, str, str], torch.Tensor],
        i_static_feats: torch.Tensor,
        u_static_feats: torch.Tensor,
        u_cat_feats: torch.Tensor,
        user_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
