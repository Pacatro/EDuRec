from pathlib import Path
from typing import cast

import lightning as L
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .. import config
from .loaders import DatasetName, load_data
from .preprocessor import Preprocessor
from .utils import get_column_types, global_preprocessing


class ElearningDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy().map(lambda x: torch.tensor(x))
        self.columns = df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return cast(dict[str, torch.Tensor], self.df.iloc[idx].to_dict())


class ElearningDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: DatasetName,
        batch_size: int,
        test_size: float,
        val_size: float,
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

        self.id_cols = [config.USER_COL, config.ITEM_COL]
        self.numeric_cols: list[str] = []
        self.categorical_lengths: dict[str, int] = {}

        self.processed_path = (
            Path(config.DATA_FOLDER) / "preprocessed" / f"{self.dataset_name.value}.csv"
        )

        self.df = load_data(dataset)
        self._process_data()

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

        self.preprocessor = Preprocessor(
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
                self.val_ds = ElearningDataset(self.val_df)
            case "test":
                self.test_ds = (
                    ElearningDataset(self.test_df) if self.test_df is not None else None
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=config.NUM_WORKERS
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=config.NUM_WORKERS
        )

    def test_dataloader(self) -> DataLoader | None:
        if self.test_ds is None:
            return None

        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=config.NUM_WORKERS
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
    def threshold(self) -> float:
        return float(self.df[config.TARGET_COL].mean())

    @property
    def min_rating(self) -> float:
        return float(self.df[config.TARGET_COL].min())

    @property
    def max_rating(self) -> float:
        return float(self.df[config.TARGET_COL].max())
