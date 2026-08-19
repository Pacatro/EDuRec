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
    """Scores user-item pairs against a candidate set or the full catalog.

    Candidate scoring restricts the computation to ``[batch, num_candidates]``,
    which is much cheaper than the full ``[batch, num_items]`` pass used during
    evaluation. Full-catalog scoring is chunked over items to bound peak memory.
    """

    def __init__(self, cfg: ScorerConfig, chunk_size: int = 1024):
        super().__init__()
        self.scorer_type = cfg.scorer_type
        self.use_context = cfg.use_context
        self.chunk_size = chunk_size

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
        item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return scores with shape ``[batch, num_candidates]``.

        Args:
            user_emb: User representations with shape ``[batch, emb_dim]``.
            item_emb: Full item representation table ``[num_items, emb_dim]``.
            context_emb: Optional context vectors ``[batch, emb_dim]``.
            item_ids: Optional ``[batch, num_candidates]`` item IDs to score.
                When provided, only those candidates are scored instead of the
                whole catalog.
        """
        if self.use_context and context_emb is None:
            raise ValueError("context_emb is required by the context-aware scorer.")
        if not self.use_context and context_emb is not None:
            raise ValueError("context_emb was provided to a context-free scorer.")

        if item_ids is not None:
            return self._score_candidates(user_emb, item_emb, context_emb, item_ids)
        return self._score_catalog(user_emb, item_emb, context_emb)

    def _score_candidates(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        context_emb: torch.Tensor | None,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score one candidate set per user: ``[batch, num_candidates]``."""
        if item_ids.ndim != 2 or item_ids.size(0) != user_emb.size(0):
            raise ValueError(
                "item_ids must have shape [batch, num_candidates] matching the "
                f"batch size, got {tuple(item_ids.shape)}."
            )
        if item_ids.numel() == 0:
            return user_emb.new_empty((user_emb.size(0), 0))

        cand_emb = item_emb[item_ids]

        if self.scorer_type == "dot":
            scores = torch.bmm(user_emb.unsqueeze(1), cand_emb.transpose(1, 2)).squeeze(
                1
            )
            if context_emb is not None:
                scores = scores + torch.bmm(
                    context_emb.unsqueeze(1), cand_emb.transpose(1, 2)
                ).squeeze(1)
            return scores

        if self.mlp is None:
            raise RuntimeError("MLP scorer is not initialized.")

        batch_size, num_candidates, emb_dim = cand_emb.shape
        parts = [
            user_emb.unsqueeze(1).expand(batch_size, num_candidates, emb_dim),
            cand_emb,
        ]
        if context_emb is not None:
            parts.append(
                context_emb.unsqueeze(1).expand(batch_size, num_candidates, emb_dim)
            )
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)

    def _score_catalog(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        context_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """Score the full catalog in chunks: ``[batch, num_items]``."""
        if self.scorer_type == "dot":
            scores = user_emb @ item_emb.T
            if context_emb is not None:
                scores = scores + context_emb @ item_emb.T
            return scores

        if self.mlp is None:
            raise RuntimeError("MLP scorer is not initialized.")

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
            if context_emb is not None:
                parts.append(
                    context_emb.unsqueeze(1).expand(batch_size, chunk_emb.size(0), -1)
                )
            scores.append(self.mlp(torch.cat(parts, dim=-1)).squeeze(-1))
        return torch.cat(scores, dim=1)
