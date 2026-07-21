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
    use_context: bool = False


class Scorer(nn.Module):
    def __init__(self, cfg: ScorerConfig):
        super().__init__()
        self.scorer_type = cfg.scorer_type
        self.use_context = cfg.use_context

        if cfg.scorer_type == "dot":
            self.mlp = None
            return

        input_dim = cfg.emb_dim * (3 if cfg.use_context else 2)

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

    def forward(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        context_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_context and context_emb is None:
            raise ValueError("context_emb is required by the context-aware scorer.")
        if not self.use_context and context_emb is not None:
            raise ValueError("context_emb was provided to a context-free scorer.")

        if self.scorer_type == "dot":
            scores = user_emb @ item_emb.T
            if context_emb is not None:
                scores = scores + context_emb @ item_emb.T
            return scores

        if self.mlp is None:
            raise RuntimeError("MLP scorer is not initialized.")

        batch_size = user_emb.shape[0]
        num_items = item_emb.shape[0]

        u = user_emb.unsqueeze(1).expand(-1, num_items, -1)
        i = item_emb.unsqueeze(0).expand(batch_size, -1, -1)

        parts = [u, i]
        if context_emb is not None:
            c = context_emb.unsqueeze(1).expand(-1, num_items, -1)
            parts.append(c)
        x = torch.cat(parts, dim=-1)

        return self.mlp(x).squeeze(-1)
