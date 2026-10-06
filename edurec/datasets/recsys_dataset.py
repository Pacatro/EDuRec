from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from edurec import settings


class RecSysQuery(NamedTuple):
    """One recommendation query with its strictly-past history.

    ``history_*`` tensors are aligned by history step: entry ``t`` of every
    history field describes the same past event. The target interaction never
    contributes to any history field.
    """

    query_id: torch.Tensor
    user_id: torch.Tensor
    history_items: torch.Tensor
    history_valid_mask: torch.Tensor
    history_dense_features: torch.Tensor
    history_cat_features: torch.Tensor
    history_timestamps: torch.Tensor
    history_delta_times: torch.Tensor
    target_item_id: torch.Tensor
    negative_item_ids: torch.Tensor
    sample_weight: torch.Tensor


class RecSysDataset(Dataset):
    """Dataset over interactions with precomputed, leakage-free histories.

    Interaction context and timestamps are not duplicated per history: each
    dataset stores only a ``history_context_index`` into the global interaction
    feature tables, and gathers the aligned features on the fly.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        history_items: torch.Tensor,
        history_valid_mask: torch.Tensor,
        history_context_index: torch.Tensor,
        interaction_dense: np.ndarray | torch.Tensor,
        interaction_cat: np.ndarray | torch.Tensor,
        interaction_timestamps: np.ndarray | torch.Tensor,
        negative_item_ids: np.ndarray | torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
    ):
        if len(history_items) != len(interactions):
            raise RuntimeError("Precomputed history must align with interactions.")
        if history_items.shape != history_context_index.shape:
            raise RuntimeError("History context index must align with history items.")

        interactions = interactions.reset_index(drop=True)
        self.user_ids = interactions[settings.USER_COL].to_numpy(copy=True)
        self.target_item_ids = interactions[settings.ITEM_COL].to_numpy(copy=True)
        self.n_interactions = len(interactions)
        self.negative_item_ids = (
            torch.empty((len(interactions), 0), dtype=torch.long)
            if negative_item_ids is None
            else torch.as_tensor(negative_item_ids, dtype=torch.long)
        )

        if self.negative_item_ids.ndim != 2 or self.negative_item_ids.size(0) != len(
            interactions
        ):
            raise RuntimeError(
                "Precomputed negatives must have shape [interactions, negatives]."
            )

        self.sample_weights = (
            torch.ones(len(interactions), dtype=torch.float32)
            if sample_weights is None
            else torch.as_tensor(sample_weights, dtype=torch.float32)
        )
        if self.sample_weights.shape != (len(interactions),):
            raise ValueError("Sample weights must have one value per interaction.")
        if (
            not torch.isfinite(self.sample_weights).all()
            or (self.sample_weights <= 0).any()
        ):
            raise ValueError("Sample weights must be finite and strictly positive.")

        self.history_items = history_items
        self.history_valid_mask = history_valid_mask
        self.history_context_index = history_context_index.long()
        self.interaction_dense = torch.as_tensor(
            interaction_dense, dtype=torch.float32
        )
        self.interaction_cat = torch.as_tensor(interaction_cat, dtype=torch.long)
        self.interaction_timestamps = torch.as_tensor(
            interaction_timestamps, dtype=torch.float64
        )

    def __len__(self) -> int:
        return self.n_interactions

    def _gather_context(
        self, context_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather aligned ``(dense, cat, timestamps, delta)`` for one history."""
        valid = context_index >= 0
        valid_f = valid.unsqueeze(-1).to(torch.float32)
        safe = context_index.clamp(min=0)

        dense = self.interaction_dense[safe] * valid_f
        cat = self.interaction_cat[safe] * valid.unsqueeze(-1)

        # Compute gaps in float64 (epoch seconds lose seconds at float32) and
        # only cast the small delta back to float32.
        timestamps64 = self.interaction_timestamps[safe] * valid.to(torch.float64)
        delta64 = torch.zeros_like(timestamps64)
        if timestamps64.numel() > 1:
            delta64[1:] = (timestamps64[1:] - timestamps64[:-1]).clamp(min=0.0)
        timestamps = (timestamps64 * valid.to(torch.float64)).to(torch.float32)
        delta = (delta64 * valid.to(torch.float64)).to(torch.float32)
        return dense, cat, timestamps, delta

    def __getitem__(self, idx: int) -> RecSysQuery:
        context_index = self.history_context_index[idx]
        dense, cat, timestamps, delta = self._gather_context(context_index)

        return RecSysQuery(
            query_id=torch.tensor(idx, dtype=torch.long),
            user_id=torch.tensor(int(self.user_ids[idx]), dtype=torch.long),
            history_items=self.history_items[idx],
            history_valid_mask=self.history_valid_mask[idx],
            history_dense_features=dense,
            history_cat_features=cat,
            history_timestamps=timestamps,
            history_delta_times=delta,
            target_item_id=torch.tensor(
                int(self.target_item_ids[idx]), dtype=torch.long
            ),
            negative_item_ids=self.negative_item_ids[idx],
            sample_weight=self.sample_weights[idx],
        )


def rating_sample_weights(
    splits: dict[str, pd.DataFrame],
) -> dict[str, torch.Tensor]:
    """Weight explicit events using a rating scale fitted on training only.

    Map training min/max ratings to [0.1, 1], then normalize by the training
    mean weight. Low ratings retain a small contribution; equal ratings and
    implicit events receive unit weights. Held-out ratings are clipped to the
    training range. Weights are runtime tensors, never dataset columns.
    """
    train = splits["train"]
    if settings.RATING_COL not in train:
        return {name: torch.ones(len(df)) for name, df in splits.items()}

    ratings: dict[str, torch.Tensor] = {}
    for name, df in splits.items():
        values = pd.to_numeric(df[settings.RATING_COL], errors="coerce").to_numpy(
            dtype=np.float32
        )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} ratings must be finite numeric values.")
        ratings[name] = torch.from_numpy(values)
    if ratings["train"].numel() == 0:
        raise ValueError("Rating weights require a nonempty training split.")
    low, high = ratings["train"].amin(), ratings["train"].amax()
    if high == low:
        return {name: torch.ones(len(df)) for name, df in splits.items()}
    weights = {
        name: 0.1 + 0.9 * ((values - low) / (high - low)).clamp(0, 1)
        for name, values in ratings.items()
    }
    mean = weights["train"].mean()
    return {name: values / mean for name, values in weights.items()}
