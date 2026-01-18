import lightning as L
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset

from . import config
from .datasets import DatasetName, load_data


class ElearningDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.df.iloc[idx].map(lambda x: torch.tensor(x)).to_dict()


class ElearningDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: DatasetName,
        batch_size: int,
        test_size: float,
        val_size: float,
        threshold: float,
        random_state: int,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.threshold = threshold
        self.random_state = random_state

        self.df = load_data(dataset)

    def _split(self) -> None:
        if config.TIME_COL in self.df.columns:
            self.df = self.df.sort_values(by=config.TIME_COL)
            train_val_df, self.test_df = train_test_split(
                self.df,
                test_size=self.test_size,
                shuffle=False,
                random_state=self.random_state,
            )
            self.train_df, self.val_df = train_test_split(
                train_val_df,
                test_size=self.val_size / (1 - self.test_size),
                shuffle=False,
                random_state=self.random_state,
            )
        else:
            train_val_df, self.test_df = train_test_split(
                self.df,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            self.train_df, self.val_df = train_test_split(
                train_val_df,
                test_size=self.val_size / (1 - self.test_size),
                random_state=self.random_state,
            )

    def setup(self, stage: str | None = None) -> None:
        self._split()

        assert isinstance(self.train_df, pd.DataFrame)
        assert isinstance(self.val_df, pd.DataFrame)
        assert isinstance(self.test_df, pd.DataFrame)

        match stage:
            case "fit":
                self.train_ds = ElearningDataset(self.train_df)
                self.val_ds = ElearningDataset(self.val_df)
            case "test":
                self.test_ds = ElearningDataset(self.test_df)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=config.NUM_WORKERS
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=config.NUM_WORKERS
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=config.NUM_WORKERS
        )
