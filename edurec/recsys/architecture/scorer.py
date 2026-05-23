from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class ScorerConfig:
    emb_dim: int
    hidden_dims: list[int] = field(default_factory=list)
    dropout: float = 0.1


class Scorer(nn.Module):
    def __init__(self, cfg: ScorerConfig):
        super().__init__()

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
        batch_size = user_emb.shape[0]
        num_items = item_emb.shape[0]

        u = user_emb.unsqueeze(1).expand(-1, num_items, -1)
        i = item_emb.unsqueeze(0).expand(batch_size, -1, -1)

        x = torch.cat([u, i], dim=-1)

        return self.mlp(x).squeeze(-1)
