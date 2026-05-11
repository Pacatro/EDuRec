from typing import NamedTuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import settings
from .user_history import UserHistory


class RankingQuery(NamedTuple):
    query_id: torch.Tensor
    user_id: torch.Tensor
    history_items: torch.Tensor
    history_ctx: torch.Tensor
    history_valid_mask: torch.Tensor
    target_item_id: torch.Tensor


class RankerDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        precomputed_history: UserHistory,
        num_ctx_feats: int,
    ):
        if len(precomputed_history.items) != len(interactions):
            raise RuntimeError("Precomputed history must align with interactions.")

        interactions = interactions.reset_index(drop=True)
        self.num_ctx_feats = num_ctx_feats

        self.user_ids = interactions[settings.USER_COL].to_numpy(copy=True)
        self.target_item_ids = interactions[settings.ITEM_COL].to_numpy(copy=True)
        self.history_items = precomputed_history.items
        self.history_ctx = precomputed_history.ctx
        self.history_valid_mask = precomputed_history.valid_mask

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> RankingQuery:
        return RankingQuery(
            query_id=torch.tensor(idx, dtype=torch.long),
            user_id=torch.tensor(int(self.user_ids[idx]), dtype=torch.long),
            history_items=self.history_items[idx],
            history_ctx=self.history_ctx[idx],
            history_valid_mask=self.history_valid_mask[idx],
            target_item_id=torch.tensor(
                int(self.target_item_ids[idx]), dtype=torch.long
            ),
        )
