from dataclasses import dataclass

import torch
from torch import nn

from .ranker import GhostRanker, RankerConfig
from .gcl import GhostGCL, GCLConfig, LossReduction
from edurec import config


@dataclass
class GhostConfig:
    user_dim: int
    item_dim: int
    hidden_dim: int

    # Static features
    u_static: torch.Tensor
    i_static: torch.Tensor

    # GCL
    edge_dropout: float = config.DROP_EDGES_P
    temperature: float = config.TAU
    max_user_samples: int = config.MAX_SAMPLES_U
    max_item_samples: int = config.MAX_SAMPLES_I
    loss_reduction: str = config.LOSS_REDUCTION

    # Ranker
    n_heads: int = 4
    n_blocks: int = 2
    ff_dim: int = 256
    dropout: float = 0.1
    max_history_len: int = 50


class Ghost(nn.Module):
    def __init__(self, cfg: GhostConfig):
        super().__init__()
        self.cfg = cfg

        self.register_buffer("u_static", cfg.u_static)
        self.register_buffer("i_static", cfg.i_static)

        gcl_cfg = GCLConfig(
            dim_user=cfg.user_dim,
            dim_item=cfg.item_dim,
            dim_hidden=cfg.hidden_dim,
            drop_edges_p=cfg.edge_dropout,
            tau=cfg.temperature,
            max_samples_u=cfg.max_user_samples,
            max_samples_i=cfg.max_item_samples,
            loss_reduc=LossReduction(cfg.loss_reduction),
        )

        ranker_cfg = RankerConfig(
            dim_model=cfg.hidden_dim,
            n_heads=cfg.n_heads,
            n_blocks=cfg.n_blocks,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
            max_history_len=cfg.max_history_len,
        )

        self.proj_u_struct = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.proj_i_struct = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.proj_u_static = nn.Linear(cfg.user_dim, cfg.hidden_dim)
        self.proj_i_static = nn.Linear(cfg.item_dim, cfg.hidden_dim)
        self.proj_ctx = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.gcl = GhostGCL(gcl_cfg)
        self.ranker = GhostRanker(ranker_cfg)

    def forward(self):
        pass
