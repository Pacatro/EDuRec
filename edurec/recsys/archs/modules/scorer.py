from dataclasses import dataclass, field
from typing import Literal, cast

import torch
from torch import nn


@dataclass
class ScorerConfig:
    emb_dim: int
    hidden_dims: list[int] = field(default_factory=list)
    dropout: float = 0.1
    scorer_type: Literal["mlp", "dot", "candidate_attention"] = "mlp"
    dot_temperature: float | None = None


class Scorer(nn.Module):
    """Scores user-item pairs with an MLP, dot product, or candidate attention.

    Candidate scoring restricts the computation to ``[batch, num_candidates]``,
    which is much cheaper than the full ``[batch, num_items]`` pass used during
    evaluation. Full-catalog scoring is chunked over items to bound peak memory.
    Candidate attention uses each candidate as a query over encoded history
    states before combining that context with the user and candidate vectors.
    """

    def __init__(self, cfg: ScorerConfig, chunk_size: int = 1024):
        super().__init__()
        self.scorer_type = cfg.scorer_type
        self.chunk_size = chunk_size

        self.query_proj: nn.Linear | None = None
        self.key_proj: nn.Linear | None = None
        self.value_proj: nn.Linear | None = None
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

        input_multiplier = 3 if cfg.scorer_type == "candidate_attention" else 2
        input_dim = cfg.emb_dim * input_multiplier
        if cfg.scorer_type == "candidate_attention":
            self.query_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=False)
            self.key_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=False)
            self.value_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=False)

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
        history_states: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return scores with shape ``[batch, num_candidates]``.

        Args:
            user_emb: User representation with shape ``[batch, emb_dim]``.
            item_emb: Full item representation table ``[num_items, emb_dim]``.
            item_ids: Optional ``[batch, num_candidates]`` item IDs to score.
                When provided, only those candidates are scored instead of the
                whole catalog.
        """
        if self.scorer_type == "candidate_attention":
            if history_states is None or history_mask is None:
                raise ValueError(
                    "candidate_attention scorer requires history_states and history_mask."
                )
            if item_ids is not None:
                return self._score_attention_candidates(
                    user_emb, item_emb, item_ids, history_states, history_mask
                )
            return self._score_attention_catalog(
                user_emb, item_emb, history_states, history_mask
            )
        if item_ids is not None:
            return self._score_candidates(user_emb, item_emb, item_ids)
        return self._score_catalog(user_emb, item_emb)

    def _candidate_context(
        self,
        candidate_emb: torch.Tensor,
        history_states: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attend from each candidate embedding to valid encoded history steps."""
        assert self.query_proj is not None
        assert self.key_proj is not None
        assert self.value_proj is not None
        query = self.query_proj(candidate_emb)
        keys = self.key_proj(history_states)
        values = self.value_proj(history_states)
        logits = torch.einsum("bcd,bld->bcl", query, keys) / (
            query.size(-1) ** 0.5
        )
        logits = logits.masked_fill(~history_mask.bool().unsqueeze(1), float("-inf"))
        # A zero attention row is used for users with no history.
        weights = torch.nan_to_num(logits.softmax(dim=-1))
        return torch.einsum("bcl,bld->bcd", weights, values)

    def _score_attention_pairs(
        self,
        user_emb: torch.Tensor,
        candidate_emb: torch.Tensor,
        history_states: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_candidates, _ = candidate_emb.shape
        context = self._candidate_context(candidate_emb, history_states, history_mask)
        user = user_emb.unsqueeze(1).expand(batch_size, num_candidates, -1)
        mlp = cast(nn.Sequential, self.mlp)
        return mlp(torch.cat([user, candidate_emb, context], dim=-1)).squeeze(-1)

    def _score_attention_candidates(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        item_ids: torch.Tensor,
        history_states: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        if item_ids.numel() == 0:
            return user_emb.new_empty((user_emb.size(0), 0))
        return self._score_attention_pairs(
            user_emb, item_emb[item_ids], history_states, history_mask
        )

    def _score_attention_catalog(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        history_states: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        chunk_size = (
            min(self.chunk_size, 256)
            if self.chunk_size > 0
            else item_emb.size(0)
        )
        scores = []
        for start in range(0, item_emb.size(0), chunk_size):
            chunk = item_emb[start : start + chunk_size]
            candidates = chunk.unsqueeze(0).expand(user_emb.size(0), -1, -1)
            scores.append(
                self._score_attention_pairs(
                    user_emb, candidates, history_states, history_mask
                )
            )
        return torch.cat(scores, dim=1)

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
        user_emb = user_emb.unsqueeze(1).expand(batch_size, num_candidates, -1)
        input = torch.cat([user_emb, cand_emb], dim=-1)

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
            parts = [
                user_emb.unsqueeze(1).expand(batch_size, chunk_emb.size(0), -1),
                chunk_emb.unsqueeze(0).expand(batch_size, -1, -1),
            ]
            scores.append(mlp(torch.cat(parts, dim=-1)).squeeze(-1))
        return torch.cat(scores, dim=1)
