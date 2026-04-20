from dataclasses import dataclass, field

import torch
from torch import nn
from torch_geometric.nn import LGConv

from .losses import LossReduction

from .. import settings
from .static_feats_encoder import StaticFeatureEncoder


@dataclass
class GraphEncoderConfig:
    num_users: int
    num_items: int
    emb_dim: int
    drop_edges_p: float = settings.DROP_EDGES_P
    tau: float = settings.TAU
    loss_reduc: LossReduction = LossReduction(settings.LOSS_REDUCTION)
    num_layers: int = settings.GNN_LAYERS
    num_user_dense_feats: int = 0
    num_item_dense_feats: int = 0
    user_cat_cardinalities: list[int] = field(default_factory=list)
    item_cat_cardinalities: list[int] = field(default_factory=list)


class GraphEncoder(nn.Module):
    def __init__(self, cfg: GraphEncoderConfig):
        super().__init__()
        self.num_users = cfg.num_users
        self.num_items = cfg.num_items
        self.drop_edges_p = cfg.drop_edges_p

        self.user_emb = nn.Embedding(self.num_users, cfg.emb_dim)
        self.item_emb = nn.Embedding(self.num_items, cfg.emb_dim)

        self.user_static_encoder = StaticFeatureEncoder(
            cfg.num_user_dense_feats, cfg.user_cat_cardinalities, cfg.emb_dim
        )
        self.item_static_encoder = StaticFeatureEncoder(
            cfg.num_item_dense_feats, cfg.item_cat_cardinalities, cfg.emb_dim
        )

        self.convs = nn.ModuleList(LGConv() for _ in range(cfg.num_layers))
        self.norm = nn.LayerNorm(cfg.emb_dim)

    def forward(
        self,
        edge_index: torch.Tensor,
        user_static_feats: torch.Tensor,
        item_static_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u_x = self.user_emb.weight + self.user_static_encoder(user_static_feats)
        i_x = self.item_emb.weight + self.item_static_encoder(item_static_feats)

        x = torch.cat([u_x, i_x], dim=0)
        x = self.norm(x)

        layer_embeddings = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            layer_embeddings.append(x)

        all_embs = torch.mean(torch.stack(layer_embeddings, dim=0), dim=0)

        user_final = all_embs[: self.num_users]
        item_final = all_embs[self.num_users :]

        return user_final, item_final
