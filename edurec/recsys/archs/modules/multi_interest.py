from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from .... import settings


@dataclass
class MultiInterestConfig:
    emb_dim: int
    num_interests: int = settings.NUM_INTERESTS
    dropout: float = settings.DROPOUT
    temperature: float | None = None
    chunk_size: int = 1024


class MultiInterestAttention(nn.Module):
    """Attend a sequence of hidden states into ``K`` interest vectors.

    ``K`` learned queries attend over the contextual history ``h1..hT`` so each
    query can specialise in a different latent topic. Padded positions are
    masked out and users without history receive zero interests.
    """

    def __init__(self, cfg: MultiInterestConfig):
        super().__init__()
        self.num_interests = cfg.num_interests
        self.queries = nn.Parameter(torch.randn(cfg.num_interests, cfg.emb_dim) * 0.02)
        self.query_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.key_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.scale = cfg.emb_dim**-0.5

    def forward(
        self,
        hidden: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[batch, num_interests, emb_dim]`` user interests."""
        mask = history_mask.bool()
        queries = self.query_proj(self.queries)
        keys = self.key_proj(hidden)

        scores = torch.einsum("kd,btd->bkt", queries, keys) * self.scale
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        # Users without history get an all ``-inf`` row; drop the resulting NaNs.
        attention = torch.nan_to_num(attention, nan=0.0)
        attention = self.attn_dropout(attention)

        return torch.einsum("bkt,btd->bkd", attention, hidden)


class InterestProfileGate(nn.Module):
    """Fuse a static user profile into each of the ``K`` interests.

    A shared gate decides, per embedding dimension, how much of the profile to
    blend into every interest, so cold-start users can fall back on it.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.gate = nn.Linear(emb_dim, emb_dim)

    def forward(
        self,
        interests: torch.Tensor,
        profile: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(profile)).unsqueeze(1)
        return gate * interests + (1.0 - gate) * profile.unsqueeze(1)


class MultiInterestScorer(nn.Module):
    """Multi-interest dot product over the item catalog.

    Each item is scored by its best-matching interest (``max`` over the ``K``
    dot products), which lets different interests retrieve different regions of
    the catalog in a single ranking.
    """

    def __init__(self, cfg: MultiInterestConfig):
        super().__init__()
        self.chunk_size = cfg.chunk_size
        initial_scale = (
            1.0 / (cfg.emb_dim**0.5)
            if cfg.temperature is None
            else 1.0 / cfg.temperature
        )
        self.logit_scale = nn.Parameter(torch.tensor(initial_scale))

    def forward(
        self,
        interests: torch.Tensor,
        item_emb: torch.Tensor,
        item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return scores ``[batch, num_candidates]``."""
        if item_ids is not None:
            return self._score_candidates(interests, item_emb, item_ids)
        return self._score_catalog(interests, item_emb)

    def _score_candidates(
        self,
        interests: torch.Tensor,
        item_emb: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        if item_ids.numel() == 0:
            return interests.new_empty((interests.size(0), 0))

        candidate_emb = item_emb[item_ids]
        similarity = torch.einsum("bkd,bcd->bkc", interests, candidate_emb)
        scores = similarity.max(dim=1).values
        return scores * self.logit_scale.clamp(min=1e-3)

    def _score_catalog(
        self,
        interests: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        num_items = item_emb.shape[0]
        chunk_size = self.chunk_size if self.chunk_size > 0 else num_items
        scale = cast(torch.Tensor, self.logit_scale).clamp(min=1e-3)

        scores = []
        for start in range(0, num_items, chunk_size):
            chunk = item_emb[start : start + chunk_size]
            similarity = torch.einsum("bkd,cd->bkc", interests, chunk)
            scores.append(similarity.max(dim=1).values * scale)
        return torch.cat(scores, dim=1)
