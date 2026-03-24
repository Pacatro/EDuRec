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

        all_embs = torch.cat(
            [self.user_emb.weight, self.item_emb.weight], dim=0
        )  # [num_users + num_items, embed_dim]

        layer_embs = [all_embs]

        for conv in self.convs:
            all_embs = conv(all_embs, data.edge_index)
            layer_embs.append(all_embs)

        final_embs = torch.stack(layer_embs, dim=1).mean(dim=1)

        user_embs = final_embs[: self.num_users]
        course_embs = final_embs[self.num_users :]

        return user_embs, course_embs


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for node representations learned from a
    homogeneous graph that represents a bipartite user–item graph.

    The unified node space is assumed to follow the convention:

        - User nodes: [0, ..., num_users - 1]
        - Item nodes: [num_users, ..., num_users + num_items - 1]

    The loss is computed independently for users and items using two aligned
    embedding sets (e.g. obtained from two graph augmentations in Graph
    Contrastive Learning). The final loss is obtained by combining the user
    and item losses according to the selected reduction strategy.

    To keep memory usage manageable on large graphs, the loss optionally
    samples a subset of nodes before computing the pairwise similarity
    matrix.

    This implementation is inspired by the InfoNCE formulation used in the
    PyGCL library (https://github.com/PyGCL/PyGCL/blob/main/GCL/losses/infonce.py).

    Args:
        tau (float): Temperature parameter used to scale similarity scores
            before the softmax. Lower values make the distribution sharper.
        max_samples_u (int): Maximum number of user nodes used to compute
            the contrastive loss. If the number of users exceeds this value,
            a random subset is sampled.
        max_samples_i (int): Maximum number of item nodes used to compute
            the contrastive loss. If the number of items exceeds this value,
            a random subset is sampled.
        reduction (LossReduction): Strategy used to combine the user and
            item losses. Supported options are MEAN and SUM.
    """

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
        self, h1: torch.Tensor, h2: torch.Tensor, num_users: int, num_items: int
    ) -> torch.Tensor:
        h1_u = h1[:num_users]
        h2_u = h2[:num_users]

        h1_i = h1[num_users : num_users + num_items]
        h2_i = h2[num_users : num_users + num_items]

        loss_u = self._compute_loss(h1_u, h2_u, self.max_samples_u)
        loss_i = self._compute_loss(h1_i, h2_i, self.max_samples_i)

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
