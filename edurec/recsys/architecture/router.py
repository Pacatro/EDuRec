from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RouterConfig:
    ctx_dim: int
    hidden_dim: int = 32
    num_modules: int = 3
    dropout: float = 0.1


class Router(nn.Module):
    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg

        self.net = nn.Sequential(
            nn.LayerNorm(cfg.ctx_dim),
            nn.Linear(cfg.ctx_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_modules),
        )

        output_layer = self.net[-1]
        assert isinstance(output_layer, nn.Linear)

        # All modules have the same weight at the beginning
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        logits = self.net(ctx)
        weights = F.softmax(logits, dim=-1)
        return weights * self.cfg.num_modules
