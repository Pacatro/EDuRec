from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import settings


class RecSysQuery(NamedTuple):
    query_id: torch.Tensor
    user_id: torch.Tensor
    history_items: torch.Tensor
    history_valid_mask: torch.Tensor
    history_times: torch.Tensor
    target_item_id: torch.Tensor
    negative_item_ids: torch.Tensor


class RecSysDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        history_items: torch.Tensor,
        history_valid_mask: torch.Tensor,
        history_times: torch.Tensor | None = None,
        negative_item_ids: np.ndarray | torch.Tensor | None = None,
    ):
        if len(history_items) != len(interactions):
            raise RuntimeError("Precomputed history must align with interactions.")

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

        self.history_items = history_items
        self.history_valid_mask = history_valid_mask
        self.history_times = (
            torch.zeros(history_items.shape, dtype=torch.float32)
            if history_times is None
            else torch.as_tensor(history_times, dtype=torch.float32)
        )

        if self.history_times.shape != self.history_items.shape:
            raise RuntimeError(
                "Precomputed history times must align with history item IDs."
            )

    def __len__(self) -> int:
        return self.n_interactions

    def __getitem__(self, idx: int) -> RecSysQuery:
        return RecSysQuery(
            query_id=torch.tensor(idx, dtype=torch.long),
            user_id=torch.tensor(int(self.user_ids[idx]), dtype=torch.long),
            history_items=self.history_items[idx],
            history_valid_mask=self.history_valid_mask[idx],
            history_times=self.history_times[idx],
            target_item_id=torch.tensor(
                int(self.target_item_ids[idx]), dtype=torch.long
            ),
            negative_item_ids=self.negative_item_ids[idx],
        )
