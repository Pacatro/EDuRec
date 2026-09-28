from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from .... import settings

EdgeType = tuple[str, str, str]


class RelationAwareGraphSAGE(nn.Module):
    """One GraphSAGE layer with softmax-weighted relation aggregation.

    Every edge type keeps its own ``SAGEConv``; the resulting messages are
    combined with a learned softmax over relations, so the encoder decides how
    much each relation (``hasSkill``, ``hasTopic``, references, co-occurrence)
    contributes instead of treating all of them uniformly. Item-item and
    item-attribute edges are both supported because the reverse directions are
    already added when the graph is built.
    """

    def __init__(self, edge_types: list[EdgeType], emb_dim: int):
        super().__init__()
        self.edge_types = list(edge_types)
        self.convs = nn.ModuleList(
            SAGEConv(emb_dim, emb_dim) for _ in self.edge_types
        )
        self.relation_logits = nn.Parameter(torch.zeros(len(self.edge_types)))

    def forward(
        self,
        x: dict[str, torch.Tensor],
        edge_index: dict[EdgeType, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not self.edge_types:
            return {}

        weights = torch.softmax(self.relation_logits, dim=0)
        out: dict[str, torch.Tensor] = {}

        for weight, edge_type, conv in zip(weights, self.edge_types, self.convs):
            edges = edge_index.get(edge_type)
            if edges is None or edges.numel() == 0:
                continue

            src, _, dst = edge_type
            message = conv((x[src], x[dst]), edges) * weight
            out[dst] = out[dst] + message if dst in out else message

        return out


@dataclass
class GraphEncoderConfig:
    num_items: int
    emb_dim: int
    item_feat_dim: int = 0
    num_layers: int = settings.GNN_LAYERS
    node_counts: dict[str, int] = field(default_factory=dict)
    edge_types: list[EdgeType] = field(default_factory=list)
    graph_mode: str = "kg"


class GraphEncoder(nn.Module):
    """Item knowledge-graph encoder.

    Item nodes start from a learned identifier embedding plus a projection of
    their numeric/text features. Categorical and list-valued metadata become
    attribute nodes with their own embeddings. Stacked relation-aware
    heterogeneous convolutions propagate information across every typed edge,
    weighting each relation with a learned softmax, so items that share an
    attribute value become neighbours with an importance the model can tune.
    Only items feed the graph; user representations are built by the sequential
    encoder.
    """

    def __init__(self, cfg: GraphEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.item_emb = nn.Embedding(cfg.num_items, cfg.emb_dim)
        self.item_proj = (
            nn.Linear(cfg.item_feat_dim, cfg.emb_dim) if cfg.item_feat_dim > 0 else None
        )

        self.input_norm = nn.LayerNorm(cfg.emb_dim)
        self.output_norm = nn.LayerNorm(cfg.emb_dim)

        self.attr_embs = nn.ModuleDict(
            {
                node_type: nn.Embedding(count, cfg.emb_dim)
                for node_type, count in cfg.node_counts.items()
            }
        )
        self.convs = nn.ModuleList(
            RelationAwareGraphSAGE(cfg.edge_types, cfg.emb_dim)
            for _ in range(cfg.num_layers)
        )

    def forward(
        self,
        edge_index: dict[EdgeType, torch.Tensor],
        item_feats: torch.Tensor,
    ) -> torch.Tensor:
        item = self.item_emb.weight
        if self.item_proj is not None:
            item = item + self.item_proj(item_feats)

        x = {"item": item}

        for node_type, embedding in self.attr_embs.items():
            x[node_type] = cast(torch.Tensor, embedding.weight)

        x = {node_type: self.input_norm(value) for node_type, value in x.items()}

        if self.cfg.graph_mode == "kg":
            for conv in self.convs:
                out = conv(x, edge_index)
                x = {
                    node_type: self.output_norm(
                        value + out.get(node_type, torch.zeros_like(value))
                    )
                    for node_type, value in x.items()
                }

        return x["item"]
