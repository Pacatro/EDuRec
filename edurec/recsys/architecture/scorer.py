from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn


@dataclass
class ScorerConfig:
    emb_dim: int
    hidden_dims: list[int] = field(default_factory=list)
    dropout: float = 0.1
    scorer_type: Literal["mlp", "dot"] = "mlp"


class Scorer(nn.Module):
    def __init__(self, cfg: ScorerConfig):
        super().__init__()
        self.scorer_type = cfg.scorer_type

        if cfg.scorer_type == "dot":
            self.mlp = None
            return

        input_dim = cfg.emb_dim * 2

        layers = []
        prev_dim = input_dim

        for hidden_dim in cfg.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        if self.scorer_type == "dot":
            return user_emb @ item_emb.T

        if self.mlp is None:
            raise RuntimeError("MLP scorer is not initialized.")

        batch_size = user_emb.shape[0]
        num_items = item_emb.shape[0]

        u = user_emb.unsqueeze(1).expand(-1, num_items, -1)
        i = item_emb.unsqueeze(0).expand(batch_size, -1, -1)

        x = torch.cat([u, i], dim=-1)

        return self.mlp(x).squeeze(-1)
