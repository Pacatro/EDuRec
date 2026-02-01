from typing import Any, Hashable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import config


class ElearningDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        n_negatives: int = 0,
        min_rating: float = 0.0,
        item_catalog: pd.DataFrame | None = None,
        user_history: dict[Any, Any] | None = None,
        all_item_ids: np.ndarray | None = None,
        id_cols: list[str] | None = None,
        numeric_cols: list[str] | None = None,
        sampling_weights: np.ndarray | None = None,
    ) -> None:
        self.df = df.copy()
        self.n_negatives = n_negatives
        self.item_catalog = item_catalog
        self.user_history = user_history
        self.all_item_ids = all_item_ids
        self.columns = df.columns.tolist()
        self.id_cols = id_cols or []
        self.numeric_cols = numeric_cols or []
        self.min_rating = min_rating
        self.sampling_weights = sampling_weights

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> list[dict[str, torch.Tensor]]:
        pos_row = self.df.iloc[idx].to_dict()
        results = [self._to_tensor_dict(pos_row)]

        if self.n_negatives > 0:
            self._generate_neg_samples(pos_row, results)

        return results

    def _to_tensor_dict(self, row: dict) -> dict[str, torch.Tensor]:
        result = {}
        for k, v in row.items():
            v = self._ensure_scalar(v)
            if k in self.id_cols:
                result[k] = torch.tensor(v, dtype=torch.long)
            elif k in (config.RELEVANT_COL,):
                result[k] = torch.tensor(v, dtype=torch.bool)
            elif k in self.numeric_cols or k in (config.RATING_COL,):
                result[k] = torch.tensor(v, dtype=torch.float32)
            else:
                result[k] = torch.tensor(v)
        return result

    def _ensure_scalar(self, v: torch.Tensor) -> Any:
        if hasattr(v, "item"):
            return v.item()
        elif hasattr(v, "tolist"):
            return v.tolist()
        return v

    def _generate_neg_samples(
        self, row: dict[Hashable, Any], results: list[dict[str, torch.Tensor]]
    ) -> None:
        assert self.user_history is not None, (
            "There must be a user history in order to generate negatives samples."
        )

        user_id = row[config.USER_COL]

        seen = self.user_history.get(user_id, set())

        neg_candidates = []
        while len(neg_candidates) < self.n_negatives:
            assert self.all_item_ids is not None
            ids = np.random.choice(
                self.all_item_ids,
                size=self.n_negatives * 2,
                p=self.sampling_weights,
            )
            for nid in ids:
                if nid not in seen and nid not in neg_candidates:
                    neg_candidates.append(nid)
                if len(neg_candidates) >= self.n_negatives:
                    break

        for neg_id in neg_candidates:
            neg_row = row.copy()
            neg_row[config.ITEM_COL] = neg_id
            neg_row[config.RELEVANT_COL] = False
            neg_row[config.RATING_COL] = self.min_rating

            if self.item_catalog is not None:
                neg_row.update(self.item_catalog.loc[neg_id].to_dict())

            results.append(self._to_tensor_dict(neg_row))
