from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import LGConv
from torch_geometric.utils import dropout_edge


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
    """

    dim_user: int
    dim_item: int
    dim_hidden: int
    drop_edges_p: float = 0.2
    tau: float = 0.1


class GhostGCL(nn.Module):
    """
    Graph contrastive learning module for the GHOST recommendation architecture.

    This module learns structural user and item representations from a bipartite
    interaction graph. First, user and item input features are projected into a
    shared latent space. Then, two stochastic graph views are generated through
    edge dropout and passed through a Light Graph Convolution (LGConv) layer.
    The resulting node representations are aligned with an InfoNCE contrastive
    objective.

    After contrastive training, the module performs graph propagation on the
    original interaction graph to obtain the final structural embeddings:
    `Eu_struct` for users and `Ei_struct` for items.

    Expected input:
        A homogeneous PyG `Data` object containing:
        - `u_x`: user feature matrix of shape [num_users, dim_user]
        - `i_x`: item feature matrix of shape [num_items, dim_item]
        - `edge_index`: bipartite interaction edges in unified node indexing
        - `num_users`: number of user nodes

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - Eu_struct: Structural user embeddings of shape
              [num_users, dim_hidden]
            - Ei_struct: Structural item embeddings of shape
              [num_items, dim_hidden]
            - loss: Contrastive InfoNCE loss computed from the two augmented
              graph views
    """

    def __init__(self, cfg: GCLConfig):
        super().__init__()
        self.drop_edges_p = cfg.drop_edges_p

        self.u_proj = nn.Linear(cfg.dim_user, cfg.dim_hidden)
        self.i_proj = nn.Linear(cfg.dim_item, cfg.dim_hidden)

        self.gnn = LGConv()
        self.contrast_loss = InfoNCELoss(tau=cfg.tau)

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u_x, i_x, edge_index = data.u_x, data.i_x, data.edge_index

        assert edge_index is not None

        x = torch.cat([self.u_proj(u_x), self.i_proj(i_x)], dim=0)

        # View 1
        edge_index_v1, _ = dropout_edge(
            edge_index, p=self.drop_edges_p, force_undirected=True
        )

        # View 2
        edge_index_v2, _ = dropout_edge(
            edge_index, p=self.drop_edges_p, force_undirected=True
        )

        # Propagation
        z1 = self.gnn(x, edge_index_v1)
        z2 = self.gnn(x, edge_index_v2)

        loss = self.contrast_loss(z1, z2)

        z = self.gnn(x, edge_index)
        Eu_struct = z[: data.num_users]
        Ei_struct = z[data.num_users :]

        return Eu_struct, Ei_struct, loss


class InfoNCELoss(nn.Module):
    """
    Computes the InfoNCE loss function.

    This implementation is based on the implementation of the PyGCL library.

    Parameters:
        - tau (float): The temperature parameter for the softmax function.
    """

    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    def _similarity(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        sim = torch.exp(self._similarity(h1, h2) / self.tau)
        pos = sim.diag()
        neg = sim.sum(dim=1) - pos
        loss = -torch.log(pos / (pos + neg)).mean()
        return loss
