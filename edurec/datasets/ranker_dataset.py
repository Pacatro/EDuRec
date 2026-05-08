from typing import NamedTuple

import numpy as np
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
    candidate_ids: torch.Tensor
    candidate_labels: torch.Tensor
    positive_position: torch.Tensor


class RankerDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        precomputed_history: UserHistory,
        num_ctx_feats: int,
        split: str,
    ):
        if len(precomputed_history.items) != len(interactions):
            raise RuntimeError("Precomputed history must align with interactions.")

        interactions = interactions.reset_index(drop=True)

        self.split = split
        self.num_ctx_feats = num_ctx_feats
        self.user_ids = interactions[settings.USER_COL].to_numpy(copy=True)
        self.item_ids = interactions[settings.ITEM_COL].to_numpy(copy=True)
        self.targets = (
            interactions[settings.RELEVANT_COL].astype(np.float32).to_numpy(copy=True)
        )

        self.history_items = precomputed_history.items
        self.history_ctx = precomputed_history.ctx
        self.history_valid_mask = precomputed_history.valid_mask

        self.candidate_ids = self._get_optional_column(
            interactions, settings.CANDIDATE_IDS_COL
        )
        self.candidate_labels = self._get_optional_column(
            interactions, settings.CANDIDATE_LABELS_COL
        )
        self.positive_positions = self._get_optional_column(
            interactions, settings.POSITIVE_POSITION_COL
        )

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> RankingQuery:
        user_id = int(self.user_ids[idx])
        next_item_id = int(self.item_ids[idx])
        target = float(self.targets[idx])

        candidate_ids = self._get_candidate_ids(idx, next_item_id)
        candidate_labels = self._get_candidate_labels(
            idx, candidate_ids, next_item_id, target
        )
        positive_position = self._get_positive_position(candidate_labels)

        return RankingQuery(
            query_id=torch.tensor(idx, dtype=torch.long),
            user_id=torch.tensor(user_id, dtype=torch.long),
            history_items=self.history_items[idx],
            history_ctx=self.history_ctx[idx],
            history_valid_mask=self.history_valid_mask[idx],
            candidate_ids=candidate_ids,
            candidate_labels=candidate_labels,
            positive_position=positive_position,
        )

    def _get_optional_column(
        self,
        interactions: pd.DataFrame,
        column: str,
    ) -> list | None:
        return interactions[column].tolist() if column in interactions.columns else None

    def _get_candidate_ids(self, idx: int, next_item_id: int) -> torch.Tensor:
        if self.candidate_ids is None:
            return torch.tensor([next_item_id + 1], dtype=torch.long)

        candidate_ids = torch.as_tensor(self.candidate_ids[idx], dtype=torch.long)
        if candidate_ids.ndim != 1:
            raise RuntimeError("candidate_ids must contain 1D candidate lists.")

        return candidate_ids + 1

    def _get_candidate_labels(
        self,
        idx: int,
        candidate_ids: torch.Tensor,
        next_item_id: int,
        target: float,
    ) -> torch.Tensor:
        if self.candidate_labels is not None:
            candidate_labels = torch.as_tensor(
                self.candidate_labels[idx], dtype=torch.float32
            )
            if candidate_labels.shape != candidate_ids.shape:
                raise RuntimeError(
                    "candidate_labels must have the same shape as candidate_ids."
                )
            return candidate_labels

        candidate_labels = torch.zeros_like(candidate_ids, dtype=torch.float32)
        positive_mask = candidate_ids == (next_item_id + 1)

        if positive_mask.any():
            candidate_labels[positive_mask] = target
            return candidate_labels

        raise RuntimeError(
            "Precomputed candidate_ids must include the target item or provide "
            "candidate_labels explicitly."
        )

    def _get_positive_position(self, candidate_labels: torch.Tensor) -> torch.Tensor:
        positive_positions = (candidate_labels > 0).nonzero(as_tuple=True)[0]

        if positive_positions.numel() == 1:
            return positive_positions[0]

        if positive_positions.numel() == 0:
            if self.split == "train":
                raise RuntimeError(
                    "Training ranking queries require exactly one positive candidate."
                )
            return torch.tensor(-1, dtype=torch.long)

        raise RuntimeError(
            "Next-item ranking requires at most one positive candidate per query."
        )
