from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn
from torch_geometric.nn import HGTConv

from edurec import settings

EdgeType = tuple[str, str, str]


@dataclass
class GraphEncoderConfig:
    num_items: int
    emb_dim: int
    item_feat_dim: int = 0
    num_layers: int = settings.GNN_LAYERS
    heads: int = 4
    dropout: float = settings.DROPOUT
    node_counts: dict[str, int] = field(default_factory=dict)
    edge_types: list[EdgeType] = field(default_factory=list)
    graph_mode: str = "kg"


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
    attribute nodes with their own embeddings. When ``graph_mode == "kg"``,
    stacked heterogeneous graph transformer convolutions propagate information
    over the typed item/attribute relations; every layer adds an explicit
    residual followed by a per-layer LayerNorm. When ``graph_mode == "id"`` no
    message passing is performed and the input item embeddings are returned.

    Only item representations are exposed. User-item interactions never enter
    the encoder.
    """

    def __init__(self, cfg: GraphEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.item_emb = nn.Embedding(cfg.num_items, cfg.emb_dim)
        self.item_proj = (
            nn.Linear(cfg.item_feat_dim, cfg.emb_dim) if cfg.item_feat_dim > 0 else None
        )

        self.input_norm = nn.LayerNorm(cfg.emb_dim)
        self.attr_embs = nn.ModuleDict(
            {
                node_type: nn.Embedding(count, cfg.emb_dim)
                for node_type, count in cfg.node_counts.items()
            }
        )

        self.convs: nn.ModuleList = nn.ModuleList()
        self.norms: nn.ModuleList = nn.ModuleList()
        self.dropout = nn.Dropout(cfg.dropout)

        if cfg.graph_mode == "kg" and cfg.edge_types:
            heads = _valid_heads(cfg.emb_dim, cfg.heads)
            node_types = [*cfg.node_counts.keys(), "item"]
            self.convs = nn.ModuleList(
                HGTConv(
                    in_channels=cfg.emb_dim,
                    out_channels=cfg.emb_dim,
                    metadata=(node_types, list(cfg.edge_types)),
                    heads=heads,
                )
                for _ in range(cfg.num_layers)
            )
            self.norms = nn.ModuleList(
                nn.LayerNorm(cfg.emb_dim) for _ in range(cfg.num_layers)
            )

    def forward(
        self,
        edge_index: dict[EdgeType, torch.Tensor],
        item_feats: torch.Tensor,
    ) -> torch.Tensor:
        item = self.item_emb.weight
        if self.item_proj is not None:
            item = item + self.item_proj(item_feats)

        x: dict[str, torch.Tensor] = {"item": item}
        for node_type, embedding in self.attr_embs.items():
            x[node_type] = cast(torch.Tensor, embedding.weight)

        x = {node_type: self.input_norm(value) for node_type, value in x.items()}

        if self.cfg.graph_mode == "kg" and len(self.convs) > 0:
            for conv, norm in zip(self.convs, self.norms):
                out = conv(x, edge_index)
                for node_type in list(x.keys()):
                    message = out.get(node_type)
                    if message is None:
                        continue
                    x[node_type] = norm(x[node_type] + self.dropout(message))

        return x["item"]
