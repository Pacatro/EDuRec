import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import config


class ElearningDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        n_negatives: int = 0,
        min_rating: float | None = None,
    ) -> None:
        self.n_negatives = n_negatives
        self.min_rating = min_rating

        self.user_ids = interactions[config.USER_COL].values
        self.item_ids = interactions[config.ITEM_COL].values
        self.targets = interactions[config.RELEVANT_COL].values

        self.all_item_ids = np.unique(np.array(self.item_ids))
        self.user_history = (
            interactions[interactions[config.RELEVANT_COL] > 0]
            .groupby(config.USER_COL)[config.ITEM_COL]
            .apply(set)
            .to_dict()
        )

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_idx = self.user_ids[idx]
        item_idx = self.item_ids[idx]
        target = self.targets[idx]

        if self.n_negatives > 0:
            neg_itms = self._sample_negatives(user_idx)

            items = np.concatenate([[item_idx], neg_itms])
            users = np.full(len(items), user_idx)
            targets = np.concatenate(
                [[target], np.zeros(self.n_negatives, dtype=np.float32)]
            )

            return (
                torch.tensor(users, dtype=torch.long),
                torch.tensor(items, dtype=torch.long),
                torch.tensor(targets, dtype=torch.float32),
            )

        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(item_idx, dtype=torch.long),
            torch.tensor(target, dtype=torch.float32),
        )

    def _sample_negatives(self, user_idx: int) -> np.ndarray:
        seen = self.user_history.get(user_idx, set())
        negatives = []

        while len(negatives) < self.n_negatives:
            samples = np.random.choice(self.all_item_ids, size=self.n_negatives * 2)
            for s in samples:
                if s not in seen and s not in negatives:
                    negatives.append(s)
                if len(negatives) == self.n_negatives:
                    break

        return np.array(negatives)
