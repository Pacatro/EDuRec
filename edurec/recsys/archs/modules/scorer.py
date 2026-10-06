from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class ScorerConfig:
    emb_dim: int
    hidden_dims: list[int] = field(default_factory=list)
    dropout: float = 0.1


class Scorer(nn.Module):
    """Scores user-item pairs with an MLP over concatenated embeddings.

    Candidate scoring restricts the computation to ``[batch, num_candidates]``,
    which is much cheaper than the full ``[batch, num_items]`` pass used during
    evaluation. Full-catalog scoring is chunked over items to bound peak memory.
    """

    def __init__(self, cfg: ScorerConfig, chunk_size: int = 1024):
        super().__init__()
        self.chunk_size = chunk_size

        layers: list[nn.Module] = []
        prev_dim = cfg.emb_dim * 2

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
        item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return scores with shape ``[batch, num_candidates]``.

        Args:
            user_emb: User representation with shape ``[batch, emb_dim]``.
            item_emb: Full item representation table ``[num_items, emb_dim]``.
            item_ids: Optional ``[batch, num_candidates]`` item IDs to score.
                When provided, only those candidates are scored instead of the
                whole catalog.
        """
        if item_ids is not None:
            if item_ids.numel() == 0:
                return user_emb.new_empty((user_emb.size(0), 0))

            cand_emb = item_emb[item_ids]
            batch_size, num_candidates = item_ids.shape
            user = user_emb.unsqueeze(1).expand(batch_size, num_candidates, -1)
            return self.mlp(torch.cat([user, cand_emb], dim=-1)).squeeze(-1)

        batch_size = user_emb.shape[0]
        num_items = item_emb.shape[0]
        chunk_size = self.chunk_size if self.chunk_size > 0 else num_items

        scores = []
        for start in range(0, num_items, chunk_size):
            chunk_emb = item_emb[start : start + chunk_size]
            parts = [
                user_emb.unsqueeze(1).expand(batch_size, chunk_emb.size(0), -1),
                chunk_emb.unsqueeze(0).expand(batch_size, -1, -1),
            ]
            scores.append(self.mlp(torch.cat(parts, dim=-1)).squeeze(-1))
        return torch.cat(scores, dim=1)
