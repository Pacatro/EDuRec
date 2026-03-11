from dataclasses import dataclass

import torch
from torch import nn

from ... import config
from .gcl import GCL, GCLConfig, LossReduction
from .ranker import Ranker, RankerConfig


@dataclass
class GhostConfig:
    user_dim: int
    item_dim: int
    hidden_dim: int
    u_static: torch.Tensor
    i_static: torch.Tensor

    # GCL Defaults
    edge_dropout: float = config.DROP_EDGES_P
    temperature: float = config.TAU
    max_user_samples: int = config.MAX_SAMPLES_U
    max_item_samples: int = config.MAX_SAMPLES_I
    loss_reduction: str = config.LOSS_REDUCTION

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

        self.register_buffer("u_static", cfg.u_static)
        self.register_buffer("i_static", cfg.i_static)

        self.proj_u_struct = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.proj_i_struct = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.proj_u_static = nn.Linear(cfg.user_dim, cfg.hidden_dim)
        self.proj_i_static = nn.Linear(cfg.item_dim, cfg.hidden_dim)
        self.proj_ctx = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.gcl = GCL(cfg.gcl)
        self.ranker = Ranker(cfg.ranker)

    def forward(self):
        pass
