from dataclasses import dataclass
from enum import StrEnum

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import LGConv

from ... import config


class LossReduction(StrEnum):
    MEAN = "mean"
    SUM = "sum"


@dataclass
class GnnEncoderConfig:
    num_users: int
    num_items: int
    emb_dim: int
    drop_edges_p: float = config.DROP_EDGES_P
    tau: float = config.TAU
    loss_reduc: LossReduction = LossReduction(config.LOSS_REDUCTION)
    num_layers: int = config.GNN_LAYERS


class GnnEncoder(nn.Module):
    def __init__(self, cfg: GnnEncoderConfig):
        super().__init__()
        self.num_users = cfg.num_users
        self.num_items = cfg.num_items
        self.drop_edges_p = cfg.drop_edges_p

        self.user_emb = nn.Embedding(self.num_users, cfg.emb_dim)
        self.item_emb = nn.Embedding(self.num_items, cfg.emb_dim)

        self.convs = nn.ModuleList(LGConv() for _ in range(cfg.num_layers))

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        assert data.edge_index is not None

        edge_index = data.edge_index

        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        layer_embeddings = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            layer_embeddings.append(x)

        all_embs = torch.mean(torch.stack(layer_embeddings, dim=0), dim=0)

        user_final = all_embs[: self.num_users]
        item_final = all_embs[self.num_users :]

        return user_final, item_final


class InfoNCELoss(nn.Module):
    def __init__(self, tau: float = 0.1, reduction: LossReduction = LossReduction.MEAN):
        super().__init__()
        self.tau = tau
        self.reduction = reduction

    def forward(
        self,
        u_emb1: torch.Tensor,
        i_emb1: torch.Tensor,
        u_emb2: torch.Tensor,
        i_emb2: torch.Tensor,
    ) -> torch.Tensor:
        loss_u = self._loss(u_emb1, u_emb2)
        loss_i = self._loss(i_emb1, i_emb2)

        if self.reduction == LossReduction.SUM:
            return loss_u + loss_i

        return 0.5 * (loss_u + loss_i)

    def _loss(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)

        logits_12 = h1 @ h2.T / self.tau
        logits_21 = h2 @ h1.T / self.tau

        labels = torch.arange(h1.size(0), device=h1.device)

        loss_12 = F.cross_entropy(logits_12, labels)
        loss_21 = F.cross_entropy(logits_21, labels)

        return 0.5 * (loss_12 + loss_21)
