from dataclasses import dataclass, field
from typing import Literal, cast

import torch
from torch import nn


@dataclass
class ScorerConfig:
    emb_dim: int
    hidden_dims: list[int] = field(default_factory=list)
    dropout: float = 0.1
    scorer_type: Literal["mlp", "dot"] = "mlp"
    dot_temperature: float | None = None
    interaction: Literal["concat", "product"] = "product"


class Scorer(nn.Module):
    """Scores a user representation against item embeddings.

    The MLP scorer concatenates the user representation, the item embedding and
    their element-wise product (``cfg.interaction == "product"``) so it can
    model multiplicative user-item interactions. Candidate scoring restricts the
    computation to ``[batch, num_candidates]``, which is much cheaper than the
    full ``[batch, num_items]`` pass used during evaluation. Full-catalog
    scoring is chunked over items to bound peak memory.
    """

    def __init__(self, cfg: ScorerConfig, chunk_size: int = 1024):
        super().__init__()
        self.scorer_type = cfg.scorer_type
        self.interaction = cfg.interaction
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

        input_dim = cfg.emb_dim * (3 if cfg.interaction == "product" else 2)

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

    def _pair_input(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.interaction == "product":
            return torch.cat([user_emb, item_emb, user_emb * item_emb], dim=-1)
        return torch.cat([user_emb, item_emb], dim=-1)

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
            scores = torch.bmm(user_emb.unsqueeze(1), cand_emb.transpose(1, 2)).squeeze(
                1
            )
            return scores * self.logit_scale.clamp(min=1e-3)

        mlp = cast(nn.Sequential, self.mlp)
        batch_size, num_candidates = item_ids.shape
        user = user_emb.unsqueeze(1).expand(batch_size, num_candidates, -1)
        input = self._pair_input(user, cand_emb)

        return mlp(input).squeeze(-1)

    def _score_catalog(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Score the full catalog in chunks: ``[batch, num_items]``."""
        if self.scorer_type == "dot":
            return user_emb @ item_emb.T * self.logit_scale.clamp(min=1e-3)

        mlp = cast(nn.Sequential, self.mlp)
        batch_size = user_emb.shape[0]
        num_items = item_emb.shape[0]
        chunk_size = self.chunk_size if self.chunk_size > 0 else num_items

        scores = []
        for start in range(0, num_items, chunk_size):
            chunk_emb = item_emb[start : start + chunk_size]
            user = user_emb.unsqueeze(1).expand(batch_size, chunk_emb.size(0), -1)
            item = chunk_emb.unsqueeze(0).expand(batch_size, -1, -1)
            scores.append(mlp(self._pair_input(user, item)).squeeze(-1))
        return torch.cat(scores, dim=1)
