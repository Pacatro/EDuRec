from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseRecArch(nn.Module, ABC):
    """Common interface for the recommendation architectures.

    Maps a history batch to item scores. Architectures may use an item
    knowledge graph and static user profiles, or encode only the sequence of
    course embeddings. The interaction context (``h_dense``/``h_cat``) and the
    temporal gaps (``h_delta``) describe the historical events only; they are
    never derived from the target interaction.
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
        h_dense: torch.Tensor | None = None,
        h_cat: torch.Tensor | None = None,
        h_delta: torch.Tensor | None = None,
        candidate_item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
