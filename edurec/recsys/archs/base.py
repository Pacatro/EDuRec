from abc import ABC, abstractmethod

import torch
from torch import nn

from .modules.kg_encoder import GraphEncoder


class BaseRecArch(nn.Module, ABC):
    """Common interface for the recommendation architectures.

    Every architecture owns a knowledge-graph encoder (``kg``) over item nodes
    and maps a history batch to item scores. Users are represented solely by the
    output of the sequential history encoder.
    """

    kg: GraphEncoder

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
