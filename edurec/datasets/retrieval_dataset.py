from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import settings
from .user_history import UserHistory


@dataclass
class RetrievalQuery:
    query_id: torch.Tensor
    user_id: torch.Tensor
    history_items: torch.Tensor
    history_ctx: torch.Tensor
    history_valid_mask: torch.Tensor
    positive_item_id: torch.Tensor


class RetrievalDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        precomputed_history: UserHistory,
        num_ctx_feats: int,
        positives_only: bool = True,
    ):
        if len(precomputed_history.items) != len(interactions):
            raise RuntimeError(
                "Precomputed history must align with interactions for RetrievalDataset."
            )

        self.num_ctx_feats = int(num_ctx_feats)

        history_items = precomputed_history.items
        history_ctx = precomputed_history.ctx
        history_valid_mask = precomputed_history.valid_mask

        df = interactions.reset_index(drop=True)

        if positives_only:
            positive_mask = (df[settings.RELEVANT_COL] > 0).to_numpy(
                dtype=bool, copy=False
            )
            df = df[positive_mask].reset_index(drop=True)
            history_items = history_items[positive_mask]
            history_ctx = history_ctx[positive_mask]
            history_valid_mask = history_valid_mask[positive_mask]

        self.user_ids = df[settings.USER_COL].to_numpy(copy=True)
        self.positive_item_ids = df[settings.ITEM_COL].to_numpy(copy=True)
        self.targets = df[settings.RELEVANT_COL].astype("float32").to_numpy(copy=True)

        self.history_items = history_items
        self.history_ctx = history_ctx
        self.history_valid_mask = history_valid_mask

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> RetrievalQuery:
        user_id = int(self.user_ids[idx])
        positive_item_id = int(self.positive_item_ids[idx])

        return RetrievalQuery(
            query_id=torch.tensor(idx, dtype=torch.long),
            user_id=torch.tensor(user_id, dtype=torch.long),
            history_items=self.history_items[idx],
            history_ctx=self.history_ctx[idx],
            history_valid_mask=self.history_valid_mask[idx],
            positive_item_id=torch.tensor(positive_item_id, dtype=torch.long),
        )
