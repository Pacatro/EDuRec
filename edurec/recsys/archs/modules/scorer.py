from dataclasses import dataclass, field
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

from .... import settings


@dataclass
class ScorerConfig:
    emb_dim: int
    hidden_dims: list[int] = field(default_factory=list)
    dropout: float = 0.1
    scorer_type: Literal["dot", "mlp"] = "dot"
    dot_temperature: float | None = None


class Scorer(nn.Module):
    """Scores a user representation against item embeddings.

    The default scorer is a normalized dot product (cosine similarity) divided
    by a learnable temperature. Unit-normalising both sides forces a shared
    user-item space and lets the whole catalog be scored with one matrix
    multiplication. Candidate scoring restricts the computation to
    ``[batch, num_candidates]``; full-catalog scoring over the MLP fallback is
    chunked over items to bound peak memory.
    """

    def __init__(self, cfg: ScorerConfig, chunk_size: int = 1024):
        super().__init__()
        self.scorer_type = cfg.scorer_type
        self.chunk_size = chunk_size

        if cfg.scorer_type == "dot":
            self.mlp = None
            # Logits are bounded to [-1, 1] before the temperature division, so
            # the initial temperature controls how peaked the softmax starts.
            initial_temperature = (
                settings.DOT_TEMPERATURE
                if cfg.dot_temperature is None
                else cfg.dot_temperature
            )
            self.temperature = nn.Parameter(torch.tensor(initial_temperature))
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
            return self._score_candidates(user_emb, item_emb, item_ids)
        return self._score_catalog(user_emb, item_emb)

    def _score_candidates(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score one candidate set per user: ``[batch, num_candidates]``."""
        if item_ids.numel() == 0:
            return user_emb.new_empty((user_emb.size(0), 0))

        cand_emb = item_emb[item_ids]

        if self.scorer_type == "dot":
            user = F.normalize(user_emb, dim=-1)
            cand = F.normalize(cand_emb, dim=-1)
            scores = torch.bmm(user.unsqueeze(1), cand.transpose(1, 2)).squeeze(1)
            return scores / self.temperature.clamp(min=1e-3)

        mlp = cast(nn.Sequential, self.mlp)
        batch_size, num_candidates = item_ids.shape
        user_emb = user_emb.unsqueeze(1).expand(batch_size, num_candidates, -1)
        input = torch.cat([user_emb, cand_emb], dim=-1)

        return mlp(input).squeeze(-1)

    def _score_catalog(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Score the full catalog: ``[batch, num_items]``."""
        if self.scorer_type == "dot":
            user = F.normalize(user_emb, dim=-1)
            item = F.normalize(item_emb, dim=-1)
            return user @ item.T / self.temperature.clamp(min=1e-3)

        mlp = cast(nn.Sequential, self.mlp)
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
            scores.append(mlp(torch.cat(parts, dim=-1)).squeeze(-1))
        return torch.cat(scores, dim=1)
