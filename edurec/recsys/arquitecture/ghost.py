from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.data import Data

from ... import config
from .gcl import GCL, GCLConfig, LossReduction
from .ranker import Ranker, RankerConfig


@dataclass
class GhostConfig:
    user_dim: int
    item_dim: int
    hidden_dim: int

    # GCL Defaults
    edge_dropout: float = config.DROP_EDGES_P
    temperature: float = config.TAU
    max_user_samples: int = config.MAX_SAMPLES_U
    max_item_samples: int = config.MAX_SAMPLES_I
    loss_reduction: str = config.LOSS_REDUCTION
    gnn_layers: int = config.GNN_LAYERS

    # Ranker Defaults
    n_heads: int = config.NUM_HEADS
    n_blocks: int = config.NUM_BLOCKS
    ff_dim: int = config.FF_DIM
    dropout: float = config.DROPOUT
    max_history_len: int = config.MAX_HISTORY_LEN

    @property
    def gcl(self) -> GCLConfig:
        return GCLConfig(
            dim_user=self.user_dim,
            dim_item=self.item_dim,
            dim_hidden=self.hidden_dim,
            drop_edges_p=self.edge_dropout,
            tau=self.temperature,
            max_samples_u=self.max_user_samples,
            max_samples_i=self.max_item_samples,
            loss_reduc=LossReduction(self.loss_reduction),
            num_layers=self.gnn_layers,
        )

    @property
    def ranker(self) -> RankerConfig:
        return RankerConfig(
            dim_model=self.hidden_dim,
            n_heads=self.n_heads,
            n_blocks=self.n_blocks,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            max_history_len=self.max_history_len,
        )


class Ghost(nn.Module):
    def __init__(self, cfg: GhostConfig):
        super().__init__()
        self.cfg = cfg

        self.proj_u_struct = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.proj_i_struct = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.proj_u_static = nn.Linear(cfg.user_dim, cfg.hidden_dim)
        self.proj_i_static = nn.Linear(cfg.item_dim, cfg.hidden_dim)
        self.proj_ctx = nn.LazyLinear(cfg.hidden_dim)

        self.gcl = GCL(cfg.gcl)
        self.ranker = Ranker(cfg.ranker)

    def forward(
        self,
        u_id: torch.Tensor,
        h_ids: torch.Tensor,
        h_ctx: torch.Tensor,
        c_ids: torch.Tensor,
        inter_graph: Data,
        u_static_global: torch.Tensor,  # [B, num_users_feats]
        i_static_global: torch.Tensor,  # [B, num_items_feats]
        hist_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        u_struct, i_struct = self.gcl(inter_graph)

        token_u = (
            self.proj_u_struct(u_struct[u_id])
            + self.proj_u_static(u_static_global[u_id])
        ).unsqueeze(1)  # [B, 1, D]

        tokens_i = (
            self.proj_i_struct(i_struct[h_ids])
            + self.proj_i_static(i_static_global[h_ids])
            + self.proj_ctx(h_ctx)
        )  # [B, L, D]

        tokens_c = self.proj_i_struct(i_struct[c_ids]) + self.proj_i_static(
            i_static_global[c_ids]
        )  # [B, K, D]

        scores = self.ranker(token_u, tokens_i, tokens_c, hist_mask=hist_mask)

        return scores
