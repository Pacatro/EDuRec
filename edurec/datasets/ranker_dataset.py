from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import settings


@dataclass
class History:
    items: torch.Tensor
    ctx: torch.Tensor
    valid_mask: torch.Tensor


class RankerBatch(NamedTuple):
    query_id: torch.Tensor
    user_id: torch.Tensor
    history_items: torch.Tensor
    history_ctx: torch.Tensor
    history_valid_mask: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_labels: torch.Tensor
    positive_position: torch.Tensor | None


class RankerDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        precomputed_history: History,
        num_ctx_feats: int,
    ):
        self.num_ctx_feats = num_ctx_feats

        if len(precomputed_history.items) != len(interactions):
            raise RuntimeError("Precomputed history must align with interactions.")

        interactions = interactions.reset_index(drop=True)
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

    def __getitem__(self, idx: int) -> RankerBatch:
        user_idx = self.user_ids[idx]
        item_idx = self.item_ids[idx]
        target = float(self.targets[idx])
        query_id = torch.tensor(idx, dtype=torch.long)
        hist_items = self.history_items[idx]
        hist_ctx = self.history_ctx[idx]
        history_valid_mask = self.history_valid_mask[idx]

        candidates = self._get_candidate_ids(idx, item_idx)
        candidate_labels = self._get_candidate_labels(idx, candidates, item_idx, target)
        positive_position = self._get_positive_position(idx, candidate_labels)

        return RankerBatch(
            query_id=query_id,
            user_id=torch.tensor(user_idx, dtype=torch.long),
            history_items=hist_items,
            history_ctx=hist_ctx,
            history_valid_mask=history_valid_mask,
            candidate_ids=candidates,
            candidate_labels=candidate_labels,
            positive_position=positive_position,
        )

    def _get_optional_column(
        self, interactions: pd.DataFrame, column: str
    ) -> list[object] | None:
        if column not in interactions.columns:
            return None

        return interactions[column].tolist()

    def _get_candidate_ids(self, idx: int, item_idx: int) -> torch.Tensor:
        if self.candidate_ids is None:
            return torch.tensor([item_idx + 1], dtype=torch.long)

        candidates = torch.as_tensor(self.candidate_ids[idx], dtype=torch.long)
        if candidates.ndim != 1:
            raise RuntimeError("candidate_ids must contain 1D candidate lists.")
        return candidates + 1

    def _get_candidate_labels(
        self,
        idx: int,
        candidates: torch.Tensor,
        item_idx: int,
        target: float,
    ) -> torch.Tensor:
        if self.candidate_labels is not None:
            labels = torch.as_tensor(self.candidate_labels[idx], dtype=torch.float32)
            if labels.shape != candidates.shape:
                raise RuntimeError(
                    "candidate_labels must have the same shape as candidate_ids."
                )
            return labels

        labels = torch.zeros_like(candidates, dtype=torch.float32)
        matches = candidates == item_idx + 1
        if matches.any():
            labels[matches] = float(target)
            return labels

        raise RuntimeError(
            "Precomputed candidate_ids must include the target item or provide "
            "candidate_labels explicitly."
        )

    def _get_positive_position(
        self, idx: int, candidate_labels: torch.Tensor
    ) -> torch.Tensor:
        if self.positive_positions is not None:
            return torch.tensor(self.positive_positions[idx], dtype=torch.long)

        positive_idx = (candidate_labels > 0).nonzero(as_tuple=True)[0]

        if len(positive_idx) == 1:
            return positive_idx.to(dtype=torch.long)[0]

        if candidate_labels.numel() == 1:
            return torch.tensor(0, dtype=torch.long)

        raise RuntimeError(
            "Each precomputed candidate list must contain exactly one positive "
            "candidate or provide positive_position explicitly."
        )
