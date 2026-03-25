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
    embed_dim: int
    drop_edges_p: float = config.DROP_EDGES_P
    tau: float = config.TAU
    max_samples_u: int = config.MAX_SAMPLES_U
    max_samples_i: int = config.MAX_SAMPLES_I
    loss_reduc: LossReduction = LossReduction(config.LOSS_REDUCTION)
    num_layers: int = config.GNN_LAYERS


class GnnEncoder(nn.Module):
    def __init__(self, cfg: GnnEncoderConfig):
        super().__init__()
        self.num_users = cfg.num_users
        self.num_items = cfg.num_items
        self.drop_edges_p = cfg.drop_edges_p

        self.user_emb = nn.Embedding(self.num_users, cfg.embed_dim)
        self.item_emb = nn.Embedding(self.num_items, cfg.embed_dim)

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
    def __init__(
        self,
        tau: float = 0.1,
        max_samples_u: int = 2048,
        max_samples_i: int = 2048,
        reduction: LossReduction = LossReduction.MEAN,
    ):
        super().__init__()
        self.tau = tau
        self.max_samples_u = max_samples_u
        self.max_samples_i = max_samples_i
        self.reduction = reduction

    def forward(
        self,
        u_emb1: torch.Tensor,
        i_emb1: torch.Tensor,
        u_emb2: torch.Tensor,
        i_emb2: torch.Tensor,
    ) -> torch.Tensor:
        loss_u = self._compute_loss(u_emb1, u_emb2, self.max_samples_u)
        loss_i = self._compute_loss(i_emb1, i_emb2, self.max_samples_i)

        if self.reduction == LossReduction.SUM:
            return loss_u + loss_i

        return 0.5 * (loss_u + loss_i)

    def _compute_loss(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        max_samples: int = 0,
    ) -> torch.Tensor:
        h1, h2 = self._sample_nodes(h1, h2, max_samples)
        return 0.5 * (self._directional_loss(h1, h2) + self._directional_loss(h2, h1))

    def _directional_loss(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        logits = self._similarity(h1, h2) / self.tau
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def _sample_nodes(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        max_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = h1.shape[0]

        if max_samples <= 0 or n <= max_samples:
            return h1, h2

        idx = torch.randperm(n, device=h1.device)[:max_samples]
        return h1[idx], h2[idx]

    def _similarity(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)
        return h1 @ h2.t()
