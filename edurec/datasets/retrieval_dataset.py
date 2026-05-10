from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info

from .. import settings
from .user_history import UserHistory


class RetrievalQuery(NamedTuple):
    query_id: torch.Tensor
    user_id: torch.Tensor
    history_items: torch.Tensor
    history_ctx: torch.Tensor
    history_valid_mask: torch.Tensor
    positive_item_id: torch.Tensor
    negative_item_ids: torch.Tensor


class RetrievalDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        precomputed_history: UserHistory,
        num_ctx_feats: int,
        num_items: int,
        num_negatives: int,
        sample_negatives: bool = True,
        positives_only: bool = True,
    ):
        if len(precomputed_history.items) != len(interactions):
            raise RuntimeError(
                "Precomputed history must align with interactions for RetrievalDataset."
            )
        if num_items <= 1:
            raise RuntimeError("RetrievalDataset requires at least two items.")
        if num_negatives <= 0:
            raise RuntimeError("RetrievalDataset requires a positive num_negatives.")

        self.num_ctx_feats = int(num_ctx_feats)
        self.num_items = int(num_items)
        self.num_negatives = int(num_negatives)
        self.sample_negatives = sample_negatives
        self.all_item_ids = np.arange(self.num_items, dtype=np.int64)
        self._rng_by_worker: dict[int, np.random.Generator] = {}

        history_items = precomputed_history.items
        history_ctx = precomputed_history.ctx
        history_valid_mask = precomputed_history.valid_mask

        full_df = interactions.reset_index(drop=True)
        self.user_seen_items = self._build_user_seen_items(full_df)

        df = full_df

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
            negative_item_ids=(
                self._sample_negative_item_ids(
                    user_id=user_id,
                    positive_item_id=positive_item_id,
                )
                if self.sample_negatives
                else torch.empty(0, dtype=torch.long)
            ),
        )

    def _build_user_seen_items(
        self,
        interactions: pd.DataFrame,
    ) -> dict[int, set[int]]:
        user_seen_items: dict[int, set[int]] = {}

        for user_id, item_id in interactions[
            [settings.USER_COL, settings.ITEM_COL]
        ].itertuples(index=False, name=None):
            user_seen_items.setdefault(int(user_id), set()).add(int(item_id))

        return user_seen_items

    def _sample_negative_item_ids(
        self,
        user_id: int,
        positive_item_id: int,
    ) -> torch.Tensor:
        blocked_items = set(self.user_seen_items.get(user_id, set()))
        blocked_items.add(positive_item_id)

        num_available = self.num_items - len(blocked_items)
        if num_available <= 0:
            raise RuntimeError(
                "No candidate negatives are available for retrieval sampling."
            )

        rng = self._get_rng()
        if num_available < self.num_negatives:
            available_item_ids = self._available_item_ids(blocked_items)
            sampled = rng.choice(
                available_item_ids,
                size=self.num_negatives,
                replace=True,
            )
            return torch.as_tensor(sampled, dtype=torch.long)

        negatives: list[int] = []
        chosen_items: set[int] = set()

        while len(negatives) < self.num_negatives:
            num_remaining = self.num_negatives - len(negatives)
            candidates = rng.integers(
                low=0,
                high=self.num_items,
                size=max(num_remaining * 2, 8),
                endpoint=False,
            )

            for candidate in candidates.tolist():
                if candidate in blocked_items or candidate in chosen_items:
                    continue

                negatives.append(candidate)
                chosen_items.add(candidate)

                if len(negatives) == self.num_negatives:
                    break

        return torch.tensor(negatives, dtype=torch.long)

    def _available_item_ids(self, blocked_items: set[int]) -> np.ndarray:
        blocked_item_ids = np.fromiter(blocked_items, dtype=np.int64)
        keep_mask = ~np.isin(self.all_item_ids, blocked_item_ids)
        return self.all_item_ids[keep_mask]

    def _get_rng(self) -> np.random.Generator:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1

        rng = self._rng_by_worker.get(worker_id)
        if rng is None:
            seed = (
                worker_info.seed if worker_info is not None else torch.initial_seed()
            ) % (2**32)
            rng = np.random.default_rng(seed)
            self._rng_by_worker[worker_id] = rng

        return rng
