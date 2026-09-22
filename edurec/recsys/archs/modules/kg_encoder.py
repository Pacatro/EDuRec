from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from .... import settings

EdgeType = tuple[str, str, str]


@dataclass
class KGEncoderConfig:
    num_users: int
    num_items: int
    emb_dim: int
    user_feat_dim: int = 0
    item_feat_dim: int = 0
    num_layers: int = settings.GNN_LAYERS
    node_counts: dict[str, int] = field(default_factory=dict)
    edge_types: list[EdgeType] = field(default_factory=list)
    graph_mode: str = "kg"


class KGEncoder(nn.Module):
    """Knowledge-graph encoder over users, items and metadata attributes.

    User and item nodes start from a learned identifier embedding plus a
    projection of their numeric/text features. Categorical and list-valued
    metadata become attribute nodes with their own embeddings. Stacked
    heterogeneous convolutions propagate information across every typed edge.
    """

    def __init__(self, cfg: KGEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.num_users = cfg.num_users
        self.num_items = cfg.num_items

        self.user_emb = nn.Embedding(cfg.num_users, cfg.emb_dim)
        self.item_emb = nn.Embedding(cfg.num_items, cfg.emb_dim)
        self.user_proj = (
            nn.Linear(cfg.user_feat_dim, cfg.emb_dim) if cfg.user_feat_dim > 0 else None
        )
        self.item_proj = (
            nn.Linear(cfg.item_feat_dim, cfg.emb_dim) if cfg.item_feat_dim > 0 else None
        )

        self.input_norm = nn.LayerNorm(cfg.emb_dim)
        self.output_norm = nn.LayerNorm(cfg.emb_dim)

        if cfg.graph_mode == "kg":
            self.attr_embs = nn.ModuleDict(
                {
                    node_type: nn.Embedding(count, cfg.emb_dim)
                    for node_type, count in cfg.node_counts.items()
                }
            )
            self.convs = nn.ModuleList(
                HeteroConv(
                    {
                        edge_type: SAGEConv(cfg.emb_dim, cfg.emb_dim)
                        for edge_type in cfg.edge_types
                    },
                    aggr="sum",
                )
                for _ in range(cfg.num_layers)
            )
        else:
            self.attr_embs = nn.ModuleDict()
            self.convs = None

    def forward(
        self,
        edge_index: dict[EdgeType, torch.Tensor],
        user_feats: torch.Tensor,
        item_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = {
            "user": self.user_emb.weight,
            "item": self.item_emb.weight,
        }
        for node_type, projection, features in (
            ("user", self.user_proj, user_feats),
            ("item", self.item_proj, item_feats),
        ):
            if projection is not None:
                x[node_type] = x[node_type] + projection(features)
        for node_type, embedding in self.attr_embs.items():
            x[node_type] = cast(torch.Tensor, embedding.weight)

        x = {node_type: self.input_norm(value) for node_type, value in x.items()}

        if self.convs is not None:
            for conv in self.convs:
                out = conv(x, edge_index)
                x = {
                    node_type: self.output_norm(
                        value + out.get(node_type, torch.zeros_like(value))
                    )
                    for node_type, value in x.items()
                }

        return x["user"], x["item"]
