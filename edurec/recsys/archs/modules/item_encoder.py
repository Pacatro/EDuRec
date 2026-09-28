from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from .... import settings
from .kg_encoder import EdgeType


@dataclass
class ItemEncoderConfig:
    num_items: int
    emb_dim: int
    meta_dim: int = 0
    text_dim: int = 0
    num_layers: int = settings.GNN_LAYERS
    node_counts: dict[str, int] = field(default_factory=dict)
    edge_types: list[EdgeType] = field(default_factory=list)
    graph_mode: str = "kg"
    dropout: float = 0.0


class ItemEncoder(nn.Module):
    """Item encoder from the multi-interest architecture.

    Three per-item signals are fused before message passing: a learned course
    identifier embedding, an encoder over numeric metadata and a projection of
    the multilingual text embedding. The fused item nodes, together with the
    categorical attribute nodes, are refined by relation-aware GraphSAGE layers
    whose edge types come from the dataset schema, and the resulting item
    embeddings are returned.
    """

    def __init__(self, cfg: ItemEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.id_emb = nn.Embedding(cfg.num_items, cfg.emb_dim)
        self.meta_encoder = (
            nn.Sequential(
                nn.Linear(cfg.meta_dim, cfg.emb_dim),
                nn.GELU(),
                nn.Linear(cfg.emb_dim, cfg.emb_dim),
            )
            if cfg.meta_dim > 0
            else None
        )
        self.text_projection = (
            nn.Linear(cfg.text_dim, cfg.emb_dim) if cfg.text_dim > 0 else None
        )

        self.input_norm = nn.LayerNorm(cfg.emb_dim)
        self.output_norm = nn.LayerNorm(cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.dropout)

        self.attr_embs = nn.ModuleDict(
            {
                node_type: nn.Embedding(count, cfg.emb_dim)
                for node_type, count in cfg.node_counts.items()
            }
        )
        self.convs = (
            nn.ModuleList(
                HeteroConv(
                    {
                        edge_type: SAGEConv(cfg.emb_dim, cfg.emb_dim)
                        for edge_type in cfg.edge_types
                    },
                    aggr="sum",
                )
                for _ in range(cfg.num_layers)
            )
            if cfg.graph_mode == "kg" and cfg.edge_types
            else nn.ModuleList()
        )

    def forward(
        self,
        edge_index: dict[EdgeType, torch.Tensor],
        item_feats: torch.Tensor,
    ) -> torch.Tensor:
        meta_dim = self.cfg.meta_dim
        text_dim = self.cfg.text_dim

        item = self.id_emb.weight
        if self.meta_encoder is not None:
            item = item + self.meta_encoder(item_feats[:, :meta_dim])
        if self.text_projection is not None:
            item = item + self.text_projection(
                item_feats[:, meta_dim : meta_dim + text_dim]
            )

        x: dict[str, torch.Tensor] = {"item": self.dropout(self.input_norm(item))}

        for node_type, embedding in self.attr_embs.items():
            x[node_type] = self.input_norm(cast(torch.Tensor, embedding.weight))

        for conv in self.convs:
            out = conv(x, edge_index)
            x = {
                node_type: self.output_norm(
                    value + out.get(node_type, torch.zeros_like(value))
                )
                for node_type, value in x.items()
            }

        return x["item"]
