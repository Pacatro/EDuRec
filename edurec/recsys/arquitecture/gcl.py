from dataclasses import dataclass
from enum import StrEnum

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import LGConv
from torch_geometric.utils import dropout_edge

from ... import config


class LossReduction(StrEnum):
    MEAN = "mean"
    SUM = "sum"


@dataclass
class GCLConfig:
    """
    Configuration container for the GHOST graph contrastive learning module.

    Attributes:
        dim_user (int): Dimensionality of the input user feature vectors.
        dim_item (int): Dimensionality of the input item feature vectors.
        dim_hidden (int): Dimensionality of the latent structural embeddings
            produced after projection and graph propagation.
        drop_edges_p (float): Probability of dropping edges when generating
            stochastic graph views for contrastive learning.
        tau (float): Temperature parameter used in the InfoNCE loss.
        max_samples_u (int): Maximum number of user nodes used to compute.
        max_samples_i (int): Maximum number of item nodes used to compute.
        loss_reduc (LossReduction): Strategy used to combine the user and
            item losses.
    """

    dim_user: int
    dim_item: int
    dim_hidden: int
    drop_edges_p: float = config.DROP_EDGES_P
    tau: float = config.TAU
    max_samples_u: int = config.MAX_SAMPLES_U
    max_samples_i: int = config.MAX_SAMPLES_I
    loss_reduc: LossReduction = LossReduction(config.LOSS_REDUCTION)


class GhostGCL(nn.Module):
    """
    Graph Contrastive Learning encoder for the GHOST recommendation architecture.

    This module learns structural user and item embeddings from a bipartite
    user-item interaction graph represented as a homogeneous PyG `Data` object
    with a unified node index space.

    The encoder operates in three main stages:

        1. User and item raw features are independently projected into a shared
        latent space of dimension `dim_hidden`.
        2. Two stochastic graph views are created from the original interaction
        graph by applying edge dropout.
        3. Each view is encoded with a Light Graph Convolution (LGConv) layer, and
        the resulting node representations are optimized with an InfoNCE-based
        contrastive objective.

    The final structural representations are obtained by averaging the node
    embeddings from both augmented views, then splitting them back into user
    and item embeddings according to the unified node layout.

    Expected graph layout:
        - Nodes [0, ..., num_users - 1] correspond to users.
        - Nodes [num_users, ..., num_users + num_items - 1] correspond to items.

    Expected input `Data` object attributes:
        - `u_x` (torch.Tensor): User feature matrix of shape
          [num_users, dim_user].
        - `i_x` (torch.Tensor): Item feature matrix of shape
          [num_items, dim_item].
        - `edge_index` (torch.Tensor): Bipartite interaction graph in COO
          format with unified node indexing and shape [2, num_edges].
        - `num_nodes` (int): Total number of nodes in the graph.
        - optionally `num_users` and `num_items`, although this implementation
          derives them from `u_x` and `i_x`.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing:
                - `u_struct`: Structural user embeddings of shape
                  [num_users, dim_hidden].
                - `i_struct`: Structural item embeddings of shape
                  [num_items, dim_hidden].
                - `loss`: Scalar InfoNCE contrastive loss computed between the
                  two augmented graph views.
    """

    def __init__(self, cfg: GCLConfig):
        super().__init__()
        self.drop_edges_p = cfg.drop_edges_p

        self.u_proj = nn.Linear(cfg.dim_user, cfg.dim_hidden)
        self.i_proj = nn.Linear(cfg.dim_item, cfg.dim_hidden)

        self.gnn = LGConv()
        self.contrast_loss = InfoNCELoss(
            tau=cfg.tau,
            max_samples_u=cfg.max_samples_u,
            max_samples_i=cfg.max_samples_i,
            reduction=cfg.loss_reduc,
        )

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert data.edge_index is not None

        # View 1
        edge_index_1, _ = dropout_edge(
            data.edge_index, p=self.drop_edges_p, force_undirected=True
        )

        # View 2
        edge_index_2, _ = dropout_edge(
            data.edge_index, p=self.drop_edges_p, force_undirected=True
        )

        z1 = self.encode(data, edge_index_1)
        z2 = self.encode(data, edge_index_2)

        num_users = data.u_x.shape[0]
        num_items = data.i_x.shape[0]

        loss = self.contrast_loss(z1, z2, num_users, num_items)

        z = 0.5 * (z1 + z2)
        u_struct = z[:num_users]
        i_struct = z[num_users:]

        return u_struct, i_struct, loss

    def encode(self, data: Data, edge_index: torch.Tensor) -> torch.Tensor:
        u = self.u_proj(data.u_x)
        i = self.i_proj(data.i_x)
        x = torch.cat([u, i], dim=0)
        x = self.gnn(x, edge_index)
        return x


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for node representations learned from a
    homogeneous graph that represents a bipartite user–item graph.

    The unified node space is assumed to follow the convention:

        - [0, ..., num_users - 1]                 -> user nodes
        - [num_users, ..., num_users + num_items - 1] -> item nodes

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
        reduction (InfoNCEReduction): Strategy used to combine the user and
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
        sim = torch.exp(self._similarity(h1, h2) / self.tau)
        pos = sim.diag()
        neg = sim.sum(dim=1) - pos
        loss = -torch.log(pos / (pos + neg)).mean()
        return loss

    def _sample_nodes(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        max_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = h1.shape[0]

        if n <= max_samples:
            return h1, h2

        idx = torch.randperm(n, device=h1.device)[:max_samples]
        return h1[idx], h2[idx]

    def _similarity(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)
        return h1 @ h2.t()
