from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import settings
from .user_history import UserHistory


class RecSysQuery(NamedTuple):
    query_id: torch.Tensor
    user_id: torch.Tensor
    history_items: torch.Tensor
    history_valid_mask: torch.Tensor
    context: torch.Tensor
    target_item_id: torch.Tensor
    negative_item_ids: torch.Tensor


class RecSysDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        history: UserHistory,
        num_ctx_feats: int,
        context_cols: list[str] | None = None,
        negative_item_ids: np.ndarray | torch.Tensor | None = None,
    ):
        if len(history.items) != len(interactions):
            raise RuntimeError("Precomputed history must align with interactions.")

        interactions = interactions.reset_index(drop=True)
        self.num_ctx_feats = num_ctx_feats
        context_cols = context_cols or []

        if len(context_cols) != num_ctx_feats:
            raise RuntimeError(
                "context_cols must contain exactly num_ctx_feats columns."
            )

        self.context = torch.as_tensor(
            interactions[context_cols].to_numpy(dtype=np.float32, copy=True),
            dtype=torch.float32,
        )

        self.user_ids = interactions[settings.USER_COL].to_numpy(copy=True)
        self.target_item_ids = interactions[settings.ITEM_COL].to_numpy(copy=True)
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

        self.history_items = history.items
        self.history_valid_mask = history.valid_mask

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> RecSysQuery:
        return RecSysQuery(
            query_id=torch.tensor(idx, dtype=torch.long),
            user_id=torch.tensor(int(self.user_ids[idx]), dtype=torch.long),
            history_items=self.history_items[idx],
            history_valid_mask=self.history_valid_mask[idx],
            context=self.context[idx],
            target_item_id=torch.tensor(
                int(self.target_item_ids[idx]), dtype=torch.long
            ),
            negative_item_ids=self.negative_item_ids[idx],
        )
