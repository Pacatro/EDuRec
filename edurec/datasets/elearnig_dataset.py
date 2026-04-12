from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import config


@dataclass
class History:
    items: torch.Tensor
    ctx: torch.Tensor
    valid_mask: torch.Tensor


class ElearningDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        precomputed_history: History,
        seen_items_by_user: dict[int, set[int]],
        num_ctx_feats: int,
        all_item_ids: np.ndarray,
        n_negatives: int = 0,
        min_rating: float | None = None,
    ) -> None:
        self.n_negatives = n_negatives
        self.min_rating = min_rating
        self.seen_items_by_user = seen_items_by_user
        self.num_ctx_feats = num_ctx_feats

        if len(precomputed_history.items) != len(interactions):
            raise RuntimeError("Precomputed history must align with interactions.")

        history_items = precomputed_history.items
        history_ctx = precomputed_history.ctx
        history_valid_mask = precomputed_history.valid_mask

        if self.n_negatives > 0:
            positive_mask = (interactions[config.RELEVANT_COL] > 0).to_numpy(
                dtype=bool, copy=False
            )
            interactions = interactions[positive_mask].reset_index(drop=True)
            history_items = history_items[positive_mask]
            history_ctx = history_ctx[positive_mask]
            history_valid_mask = history_valid_mask[positive_mask]

        self.user_ids = interactions[config.USER_COL].values
        self.item_ids = interactions[config.ITEM_COL].values
        self.targets = interactions[config.RELEVANT_COL].astype(np.float32).values
        self.all_item_ids = np.asarray(all_item_ids, dtype=np.int64)
        self.history_items = history_items
        self.history_ctx = history_ctx
        self.history_valid_mask = history_valid_mask

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user_idx = self.user_ids[idx]
        item_idx = self.item_ids[idx]
        target = float(self.targets[idx])
        query_id = torch.tensor(idx, dtype=torch.long)
        hist_items = self.history_items[idx]
        hist_ctx = self.history_ctx[idx]
        history_valid_mask = self.history_valid_mask[idx]

        if self.n_negatives <= 0:
            return {
                "query_id": query_id,
                "user_id": torch.tensor(user_idx, dtype=torch.long),
                "history_items": hist_items,
                "history_ctx": hist_ctx,
                "history_valid_mask": history_valid_mask,
                "candidate_ids": torch.tensor(
                    item_idx + 1,
                    dtype=torch.long,
                ).unsqueeze(0),
                "candidate_labels": torch.tensor(target, dtype=torch.float32),
                "positive_position": torch.tensor(0, dtype=torch.long),
            }

        neg_items = self._sample_negatives(user_idx, item_idx)

        candidates = torch.tensor(
            np.concatenate([[item_idx], neg_items]), dtype=torch.long
        )  # [1 + n_neg]

        targets = torch.zeros(len(candidates), dtype=torch.float32)  # [1 + n_neg]
        targets[0] = 1.0

        perm = torch.randperm(len(candidates))
        shuffled_candidates = candidates[perm] + 1
        shuffled_targets = targets[perm]

        positive_position = (perm == 0).nonzero(as_tuple=True)[0].item()

        return {
            "query_id": query_id,
            "user_id": torch.tensor(user_idx, dtype=torch.long),
            "history_items": hist_items,
            "history_ctx": hist_ctx,
            "history_valid_mask": history_valid_mask,
            "candidate_ids": shuffled_candidates,
            "candidate_labels": shuffled_targets,
            "positive_position": torch.tensor(positive_position, dtype=torch.long),
        }

    def _sample_negatives(self, user_idx: int, current_item: int) -> np.ndarray:
        seen = set(self.seen_items_by_user.get(int(user_idx), set()))
        seen.add(int(current_item))

        available_items = np.array(
            [item_id for item_id in self.all_item_ids if int(item_id) not in seen],
            dtype=np.int64,
        )

        if available_items.size < self.n_negatives:
            raise RuntimeError(
                "Not enough negative items available for the requested query."
            )

        return np.random.choice(
            available_items,
            size=self.n_negatives,
            replace=False,
        )
