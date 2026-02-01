from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .. import config
from .data_processor import DataProcessor
from .loaders import DatasetName, load_data
from .utils import get_column_types, global_preprocessing


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

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> list[dict[str, torch.Tensor]]:
        pos_row = self.df.iloc[idx].to_dict()
        user_id = pos_row[config.USER_COL]
        results = [self._to_tensor_dict(pos_row)]

        if self.n_negatives > 0 and self.user_history is not None:
            seen = self.user_history.get(user_id, set())

            for _ in range(self.n_negatives):
                while True:
                    assert self.all_item_ids is not None
                    neg_id = np.random.choice(self.all_item_ids)
                    if neg_id not in seen:
                        break

                neg_row = pos_row.copy()
                neg_row[config.ITEM_COL] = neg_id
                neg_row[config.RELEVANT_COL] = False
                neg_row[config.RATING_COL] = self.min_rating

                if self.item_catalog is not None:
                    neg_row.update(self.item_catalog.loc[neg_id].to_dict())

                results.append(self._to_tensor_dict(neg_row))

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


def collate_fn(batch: list[list[dict]]) -> dict[str, torch.Tensor]:
    flattened_batch = [item for sublist in batch for item in sublist]
    result = {}
    for key in flattened_batch[0].keys():
        tensors = [d[key] for d in flattened_batch]
        if tensors[0].dtype == torch.float32:
            result[key] = torch.stack(tensors).float()
        else:
            result[key] = torch.stack(tensors).long()
    return result


class ElearningDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: DatasetName,
        batch_size: int,
        test_size: float,
        val_size: float,
        n_neg_val: int = config.N_NEG_VAL,
        n_neg_test: int = config.N_NEG_TEST,
        save_data: bool = False,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.save_data = save_data
        self.n_neg_val = n_neg_val
        self.n_neg_test = n_neg_test

        self.id_cols = [config.USER_COL, config.ITEM_COL]
        self.numeric_cols: list[str] = []
        self.categorical_lengths: dict[str, int] = {}

        self.processed_path = (
            Path(config.DATA_FOLDER) / "preprocessed" / f"{self.dataset_name.value}.csv"
        )

        self.df = load_data(dataset)
        self._process_data()

        self.user_history = (
            self.df.groupby(config.USER_COL)[config.ITEM_COL].apply(set).to_dict()
        )
        self.all_item_ids = self.df[config.ITEM_COL].unique()
        label_cols = [config.RATING_COL, config.RELEVANT_COL, config.USER_COL]
        self.item_catalog = (
            self.df.drop_duplicates(config.ITEM_COL)
            .set_index(config.ITEM_COL)
            .drop(columns=label_cols, errors="ignore")
        )

    def _split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if config.TIME_COL in self.df.columns:
            self.df = self.df.sort_values(by=config.TIME_COL)
            train_val_df, test_df = train_test_split(
                self.df,
                test_size=self.test_size,
                shuffle=False,
                random_state=self.random_state,
            )
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.val_size,
                shuffle=False,
                random_state=self.random_state,
            )
        else:
            train_val_df, test_df = train_test_split(
                self.df,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.val_size / (1 - self.test_size),
                random_state=self.random_state,
            )

        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

        return train_df, val_df, test_df

    def _process_data(self) -> None:
        global_preprocessing(self.df)

        train_df, val_df, test_df = self._split()

        self.numeric_cols, self.categorical_lengths = get_column_types(
            train_df, self.id_cols
        )

        self.preprocessor = DataProcessor(
            self.numeric_cols,
            list(self.categorical_lengths.keys()),
            self.id_cols,
        )

        self.train_df, self.val_df, self.test_df = self.preprocessor.fit_transform(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        if self.test_df is None:
            return

        # TODO: Find a way to not concatenate the data
        self.df = pd.concat(
            [self.train_df, self.val_df, self.test_df], ignore_index=True
        )

        if self.save_data:
            # Save the preprocessed data
            self.processed_path.parent.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(self.processed_path, index=False)

    def setup(self, stage: str | None = None) -> None:
        match stage:
            case "fit":
                self.train_ds = ElearningDataset(self.train_df)
                self.val_ds = ElearningDataset(
                    self.val_df,
                    n_negatives=self.n_neg_val,
                    min_rating=self.min_rating,
                    item_catalog=self.item_catalog,
                    user_history=self.user_history,
                    all_item_ids=self.all_item_ids,
                    id_cols=self.id_cols,
                    numeric_cols=self.numeric_cols,
                )
            case "test":
                if self.test_df is not None:
                    self.test_ds = ElearningDataset(
                        self.test_df,
                        n_negatives=self.n_neg_test,
                        min_rating=self.min_rating,
                        item_catalog=self.item_catalog,
                        user_history=self.user_history,
                        all_item_ids=self.all_item_ids,
                        id_cols=self.id_cols,
                        numeric_cols=self.numeric_cols,
                    )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader | None:
        if self.test_ds is None:
            return None

        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn,
        )

    @property
    def num_users(self) -> int:
        return int(self.df[config.USER_COL].nunique())

    @property
    def num_items(self) -> int:
        return int(self.df[config.ITEM_COL].nunique())

    @property
    def numeric_features(self) -> list[str]:
        return self.numeric_cols

    @property
    def cat_cardinalities(self) -> dict[str, int]:
        return {k: v + 2 for k, v in self.categorical_lengths.items()}

    @property
    def sparsity(self) -> float:
        return 1 - len(self.df) / (self.num_users * self.num_items)

    @property
    def min_rating(self) -> float:
        return float(self.df[config.RATING_COL].min())

    @property
    def max_rating(self) -> float:
        return float(self.df[config.RATING_COL].max())
