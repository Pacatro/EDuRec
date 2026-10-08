from abc import ABC, abstractmethod
from typing import Self, cast

import torch
from torch import nn

from edurec.recsys.archs.modules.kg_encoder import EdgeType


class BaseRecArch(nn.Module, ABC):
    """Common interface for the recommendation architectures.

    Maps a history batch to item scores. Architectures may use an item
    knowledge graph and static user profiles, or encode only the sequence of
    course embeddings. The interaction context (``h_dense``/``h_cat``) and the
    temporal gaps (``h_delta``) describe the historical events only; they are
    never derived from the target interaction.

    Batch-invariant tensors (graph structure and feature tables) are attached
    once with :meth:`register_static` and kept as buffers, so :meth:`forward`
    only receives per-batch data.
    """

    # Buffers created by ``register_static``.
    i_static_feats: torch.Tensor
    u_static_feats: torch.Tensor
    u_cat_feats: torch.Tensor

    _edge_types: tuple[EdgeType, ...] = ()
    _item_emb_cache: torch.Tensor | None = None

    def register_static(
        self,
        edge_index: dict[EdgeType, torch.Tensor],
        i_static_feats: torch.Tensor,
        u_static_feats: torch.Tensor,
        u_cat_feats: torch.Tensor,
    ) -> None:
        """Attach the batch-invariant graph and feature tables as buffers."""
        self._edge_types = tuple(edge_index)
        for idx, edge_type in enumerate(self._edge_types):
            self.register_buffer(
                f"edge_index_{idx}",
                edge_index[edge_type],
                persistent=False,
            )
        self.register_buffer("i_static_feats", i_static_feats, persistent=False)
        self.register_buffer("u_static_feats", u_static_feats, persistent=False)
        self.register_buffer("u_cat_feats", u_cat_feats, persistent=False)
        self._item_emb_cache = None

    @property
    def edge_index(self) -> dict[EdgeType, torch.Tensor]:
        return {
            edge_type: cast(torch.Tensor, self.get_buffer(f"edge_index_{idx}"))
            for idx, edge_type in enumerate(self._edge_types)
        }

    def item_embeddings(self) -> torch.Tensor:
        """Return the shared ``[num_items, emb_dim]`` item table.

        The table depends on the learned parameters, so it is recomputed every
        training step and only cached across calls while evaluating.
        """
        if not self.training and self._item_emb_cache is not None:
            return self._item_emb_cache

        embeddings = self._compute_item_embeddings()
        if not self.training:
            self._item_emb_cache = embeddings
        return embeddings

    @abstractmethod
    def _compute_item_embeddings(self) -> torch.Tensor: ...

    def train(self, mode: bool = True) -> Self:
        self._item_emb_cache = None
        return super().train(mode)

    def auxiliary_loss(
        self,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return an optional training regularizer; targets never enter history."""
        return None

    @abstractmethod
    def forward(
        self,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        user_ids: torch.Tensor,
        h_dense: torch.Tensor | None = None,
        h_cat: torch.Tensor | None = None,
        h_delta: torch.Tensor | None = None,
        candidate_item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
