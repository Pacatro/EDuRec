"""Relational item-knowledge-graph encoder and its contrastive helpers."""

from collections.abc import Mapping, Sequence
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HGTConv

from edurec import settings

EdgeType = tuple[str, str, str]


def _valid_heads(emb_dim: int, requested: int) -> int:
    """Largest divisor of ``emb_dim`` that is ``<= requested`` and ``>= 1``."""
    for candidate in range(min(requested, emb_dim), 0, -1):
        if emb_dim % candidate == 0:
            return candidate
    return 1


class GraphEncoder(nn.Module):
    """Relational item-knowledge-graph encoder built on :class:`HGTConv`.

    Item nodes start from a learned identifier embedding plus a projection of
    their dense features. Categorical and list-valued item metadata become
    attribute nodes with their own embeddings. Available attribute text features
    are projected and added to these embeddings before message passing.
    When ``graph_mode == "kg"``,
    stacked heterogeneous graph transformer convolutions propagate information
    over the typed item/attribute relations; every layer adds an explicit
    residual followed by a per-layer LayerNorm. When ``graph_mode == "id"`` no
    message passing is performed and the input item embeddings are returned.

    Only item representations are exposed. User-item interactions never enter
    the encoder.
    """

    def __init__(
        self,
        num_items: int,
        emb_dim: int,
        item_feat_dim: int = 0,
        num_layers: int = settings.GNN_LAYERS,
        heads: int = settings.GNN_HEADS,
        dropout: float = settings.DROPOUT,
        node_counts: Mapping[str, int] | None = None,
        node_feat_dims: Mapping[str, int] | None = None,
        edge_types: Sequence[EdgeType] = (),
        graph_mode: Literal["kg", "id"] = "kg",
    ):
        super().__init__()
        node_counts = dict(node_counts or {})
        node_feat_dims = dict(node_feat_dims or {})
        self.graph_mode = graph_mode

        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.item_proj = (
            nn.Linear(item_feat_dim, emb_dim) if item_feat_dim > 0 else None
        )

        self.input_norm = nn.LayerNorm(emb_dim)
        self.attr_embs = nn.ModuleDict(
            {
                node_type: nn.Embedding(count, emb_dim)
                for node_type, count in node_counts.items()
            }
        )
        self.attr_projs = nn.ModuleDict(
            {
                node_type: nn.Linear(dim, emb_dim, bias=False)
                for node_type, dim in node_feat_dims.items()
            }
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        if graph_mode == "kg" and edge_types:
            heads = _valid_heads(emb_dim, heads)
            node_types = [*node_counts.keys(), "item"]
            self.convs = nn.ModuleList(
                HGTConv(
                    in_channels=emb_dim,
                    out_channels=emb_dim,
                    metadata=(node_types, list(edge_types)),
                    heads=heads,
                )
                for _ in range(num_layers)
            )
            self.norms = nn.ModuleList(
                nn.LayerNorm(emb_dim) for _ in range(num_layers)
            )

    def forward(
        self,
        edge_index: dict[EdgeType, torch.Tensor],
        item_feats: torch.Tensor,
        node_feats: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        item = self.item_emb.weight
        if self.item_proj is not None:
            item = item + self.item_proj(item_feats)

        x: dict[str, torch.Tensor] = {"item": item}
        for node_type, embedding in self.attr_embs.items():
            x[node_type] = cast(nn.Embedding, embedding).weight
            if node_type in self.attr_projs:
                if node_feats is None or node_type not in node_feats:
                    raise ValueError(f"Missing graph features for {node_type}")
                x[node_type] = x[node_type] + self.attr_projs[node_type](
                    node_feats[node_type]
                )

        x = {node_type: self.input_norm(value) for node_type, value in x.items()}

        if self.graph_mode == "kg" and len(self.convs) > 0:
            for conv, norm in zip(self.convs, self.norms):
                out = conv(x, edge_index)
                for node_type in list(x.keys()):
                    message = out.get(node_type)
                    if message is None:
                        continue
                    x[node_type] = norm(x[node_type] + self.dropout(message))

        return x["item"]


def drop_relation_edges(
    edge_index: dict[EdgeType, torch.Tensor], dropout: float
) -> dict[EdgeType, torch.Tensor]:
    """Drop edges together with their explicit ``rev_`` counterparts.

    Keep node IDs and all relation keys, including empty relations. Reverse
    edges are rebuilt from sampled forward edges: coalescing the original
    relations can give the two directions different column orders. Inputs
    are never modified. Sampling uses the device's seeded PyTorch RNG.
    """
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("Edge dropout must be between 0 and 1.")
    if dropout == 0.0:
        return dict(edge_index)

    view: dict[EdgeType, torch.Tensor] = {}
    for edge_type, edges in edge_index.items():
        source, relation, target = edge_type
        if relation.startswith("rev_"):
            forward_type = (target, relation.removeprefix("rev_"), source)
            if forward_type in edge_index:
                continue

        keep = torch.rand(edges.size(1), device=edges.device) >= dropout
        selected = edges[:, keep]
        view[edge_type] = selected
        reverse_type = (target, f"rev_{relation}", source)
        if reverse_type in edge_index:
            view[reverse_type] = selected.flip(0)

    return view


def item_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    item_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric cross-view InfoNCE for unique item IDs.

    ``first``/``second`` have shape ``[num_items, emb_dim]``; ``item_ids``
    selects the same items in both views. Other selected items are negatives.
    Compute normalized logits in float32 even under mixed precision.
    """
    if not temperature > 0.0:
        raise ValueError("Contrastive temperature must be positive.")
    ids = item_ids.unique()
    if ids.numel() < 2:
        return (first.sum() + second.sum()) * 0.0

    with torch.autocast(device_type=first.device.type, enabled=False):
        z1 = F.normalize(first[ids].float(), dim=-1)
        z2 = F.normalize(second[ids].float(), dim=-1)
        logits = z1 @ z2.T / temperature
        labels = torch.arange(ids.numel(), device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )
