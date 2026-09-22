from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn


@dataclass
class ScorerConfig:
    emb_dim: int
    num_user_sources: int = 1
    hidden_dims: list[int] = field(default_factory=list)
    dropout: float = 0.1
    scorer_type: Literal["mlp", "dot"] = "mlp"
    dot_temperature: float | None = None


class Scorer(nn.Module):
    """Scores concatenated user representations against item embeddings.

    Candidate scoring restricts the computation to ``[batch, num_candidates]``,
    which is much cheaper than the full ``[batch, num_items]`` pass used during
    evaluation. Full-catalog scoring is chunked over items to bound peak memory.
    """

    def __init__(self, cfg: ScorerConfig, chunk_size: int = 1024):
        super().__init__()
        self.scorer_type = cfg.scorer_type
        self.emb_dim = cfg.emb_dim
        self.num_user_sources = cfg.num_user_sources
        self.chunk_size = chunk_size

        if cfg.scorer_type == "dot":
            self.mlp = None
            # Embeddings are LayerNorm-ed, so a raw dot product produces logits
            # of order emb_dim and saturates the ranking cross-entropy. A
            # learnable scale (initialised to 1 / sqrt(emb_dim)) lets the model
            # calibrate the logit temperature.
            initial_scale = (
                1.0 / (cfg.emb_dim**0.5)
                if cfg.dot_temperature is None
                else 1.0 / cfg.dot_temperature
            )
            self.logit_scale = nn.Parameter(torch.tensor(initial_scale))
            return

        input_dim = cfg.emb_dim * (cfg.num_user_sources + 1)

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

    def _dot_scores(
        self, user_emb: torch.Tensor, item_emb: torch.Tensor
    ) -> torch.Tensor:
        """Sum the dot product of each user source with the item embeddings.

        ``user_emb`` concatenates ``num_user_sources`` representations of
        ``emb_dim`` each, so the score is their total similarity with the item.
        """
        sources = user_emb.split(self.emb_dim, dim=-1)
        scores = sum(source @ item_emb.T for source in sources)
        return scores * self.logit_scale.clamp(min=1e-3)

    def forward(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return scores with shape ``[batch, num_candidates]``.

        Args:
            user_emb: Concatenated user representations with shape
                ``[batch, num_user_sources * emb_dim]``.
            item_emb: Full item representation table ``[num_items, emb_dim]``.
            item_ids: Optional ``[batch, num_candidates]`` item IDs to score.
                When provided, only those candidates are scored instead of the
                whole catalog.
        """
        if item_ids is not None:
            return self._score_candidates(user_emb, item_emb, item_ids)
        return self._score_catalog(user_emb, item_emb)

    def _score_candidates(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
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
            sources = user_emb.split(self.emb_dim, dim=-1)
            scores = sum(
                torch.bmm(source.unsqueeze(1), cand_emb.transpose(1, 2)).squeeze(1)
                for source in sources
            )
            return scores * self.logit_scale.clamp(min=1e-3)

        if self.mlp is None:
            raise RuntimeError("MLP scorer is not initialized.")

        batch_size, num_candidates = item_ids.shape
        parts = [
            user_emb.unsqueeze(1).expand(batch_size, num_candidates, -1),
            cand_emb,
        ]
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)

    def _score_catalog(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Score the full catalog in chunks: ``[batch, num_items]``."""
        if self.scorer_type == "dot":
            return self._dot_scores(user_emb, item_emb)

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
            scores.append(self.mlp(torch.cat(parts, dim=-1)).squeeze(-1))
        return torch.cat(scores, dim=1)
