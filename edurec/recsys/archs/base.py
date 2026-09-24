from abc import ABC, abstractmethod

import torch
from torch import nn

from .modules.kg_encoder import KGEncoder


class BaseRecArch(nn.Module, ABC):
    """Common interface for the recommendation architectures.

    Every architecture owns a knowledge-graph encoder (``kg``) and maps a query
    batch to item scores. The concrete heads differ in how the user
    representation is built (parallel graph+sequence fusion vs. a serial
    graph-to-sequence pipeline).
    """

    kg: KGEncoder

    @abstractmethod
    def forward(
        self,
        u_ids: torch.Tensor,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        edge_index: dict[tuple[str, str, str], torch.Tensor],
        u_static_feats: torch.Tensor,
        i_static_feats: torch.Tensor,
        candidate_item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
