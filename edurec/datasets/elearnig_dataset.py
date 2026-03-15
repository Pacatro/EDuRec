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
            .apply(list)
            .to_dict()
        )

        self.user_history_set = {
            u: set(items) for u, items in self.user_history.items()
        }

    def __len__(self) -> int:
        return len(self.user_ids)

    def _get_history_and_mask(self, user_idx: int, current_item: int):
        full_hist = self.user_history.get(user_idx, [])

        # Take all items before the current one
        try:
            pos_idx = full_hist.index(current_item)
            history = full_hist[:pos_idx]
        except ValueError:
            history = full_hist.copy()

        history = history[-config.MAX_HISTORY_LEN :]
        hist_len = len(history)

        hist_tensor = torch.zeros(config.MAX_HISTORY_LEN, dtype=torch.long)
        mask = torch.zeros(config.MAX_HISTORY_LEN, dtype=torch.bool)

        if hist_len > 0:
            hist_tensor[:hist_len] = torch.tensor(history, dtype=torch.long)
            mask[:hist_len] = True  # True = Real data, False = Padding

        return hist_tensor, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user_idx = self.user_ids[idx]
        item_idx = self.item_ids[idx]
        target = float(self.targets[idx])

        hist_tensor, mask = self._get_history_and_mask(user_idx, item_idx)

        if self.n_negatives > 0:
            print("GENERATING SEQUENCES")
            neg_itms = self._sample_negatives(user_idx)

            candidates = torch.tensor(
                np.concatenate([[item_idx], neg_itms]), dtype=torch.long
            )

            targets = torch.zeros(len(candidates), dtype=torch.float32)
            targets[0] = target

            return {
                "user_id": torch.tensor(user_idx, dtype=torch.long),
                "history": hist_tensor,
                "candidates": candidates,
                "mask": mask,
                "target": targets,
            }

        return {
            "user_id": torch.tensor(user_idx, dtype=torch.long),
            "history": hist_tensor,
            "candidate": torch.tensor(item_idx, dtype=torch.long),
            "mask": mask,
            "target": torch.tensor(target, dtype=torch.float32),
        }

    def _sample_negatives(self, user_idx: int) -> np.ndarray:
        seen = self.user_history_set.get(user_idx, set())
        negatives = []

        while len(negatives) < self.n_negatives:
            samples = np.random.choice(self.all_item_ids, size=self.n_negatives * 2)
            for s in samples:
                if s not in seen and s not in negatives:
                    negatives.append(s)
                if len(negatives) == self.n_negatives:
                    break
        return np.array(negatives)
