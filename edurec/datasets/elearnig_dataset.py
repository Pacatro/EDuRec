import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import config


class ElearningDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        global_history: dict[int, list],
        num_ctx_feats: int,
        n_negatives: int = 0,
        min_rating: float | None = None,
    ) -> None:
        self.n_negatives = n_negatives
        self.min_rating = min_rating
        self.global_history = global_history
        self.num_ctx_feats = num_ctx_feats

        self.user_ids = interactions[config.USER_COL].values
        self.item_ids = interactions[config.ITEM_COL].values
        self.targets = interactions[config.RELEVANT_COL].values
        self.all_item_ids = np.unique(np.array(self.item_ids))

        self.user_history = {
            u_id: set(item[0] for item in history)
            for u_id, history in self.global_history.items()
        }

    def __len__(self) -> int:
        return len(self.user_ids)

    def _get_history_and_mask(self, user_idx: int, current_item: int):
        full_hist = self.global_history.get(user_idx, [])

        # Take all items before the current one
        try:
            pos_idx = next(i for i, x in enumerate(full_hist) if x[0] == current_item)
            history_data = full_hist[:pos_idx]
        except StopIteration:
            history_data = full_hist.copy()

        if len(history_data) > config.MAX_HISTORY_LEN:
            history_data = history_data[-config.MAX_HISTORY_LEN :]

        hist_len = len(history_data)
        hist_items = torch.zeros(config.MAX_HISTORY_LEN, dtype=torch.long)
        mask = torch.zeros(config.MAX_HISTORY_LEN, dtype=torch.bool)

        hist_ctx = torch.zeros(
            config.MAX_HISTORY_LEN, self.num_ctx_feats, dtype=torch.float32
        )

        if hist_len == 0:
            return hist_items, hist_ctx, mask

        hist_items[:hist_len] = torch.tensor(
            [x[0] for x in history_data], dtype=torch.long
        )
        hist_ctx[:hist_len] = torch.tensor(
            [x[1] for x in history_data], dtype=torch.float32
        )
        mask[:hist_len] = True

        return hist_items, hist_ctx, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user_idx = self.user_ids[idx]
        item_idx = self.item_ids[idx]
        target = float(self.targets[idx])
        query_id = torch.tensor(idx, dtype=torch.long)

        hist_items, hist_ctx, mask = self._get_history_and_mask(user_idx, item_idx)

        if self.n_negatives <= 0:
            return {
                "query_id": query_id,
                "user_id": torch.tensor(user_idx, dtype=torch.long),
                "history_items": hist_items,
                "history_ctx": hist_ctx,
                "candidates": torch.tensor(item_idx, dtype=torch.long).unsqueeze(0),
                "mask": mask,
                "target": torch.tensor(target, dtype=torch.float32),
            }

        neg_itms = self._sample_negatives(user_idx)

        candidates = torch.tensor(
            np.concatenate([[item_idx], neg_itms]), dtype=torch.long
        )  # [1 + n_neg]

        targets = torch.zeros(len(candidates), dtype=torch.float32)  # [1 + n_neg]
        targets[0] = target

        perm = torch.randperm(len(candidates))
        shuffled_candidates = candidates[perm]
        shuffled_targets = targets[perm]

        positive_position = (perm == 0).nonzero(as_tuple=True)[0].item()

        return {
            "query_id": query_id,
            "user_id": torch.tensor(user_idx, dtype=torch.long),
            "history_items": hist_items,
            "history_ctx": hist_ctx,
            "candidates": shuffled_candidates,
            "mask": mask,
            "target": shuffled_targets,
            "positive_position": torch.tensor(positive_position, dtype=torch.long),
        }

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
