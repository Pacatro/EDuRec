import lightning as L
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from . import config
from .datasets import DatasetName, load_data
from .preprocess import Preprocessor


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
        save_data: bool = True,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.threshold = threshold
        self.random_state = random_state
        self.save_data = save_data

    def _is_data_on_disk(self) -> bool:
        p = Path(config.DATA_FOLDER) / self.dataset_name.value
        return p.exists() and p.is_dir() and any(p.iterdir())

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
                test_size=self.val_size / (1 - self.test_size),
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

    def setup(self, stage: str | None = None) -> None:
        data_path = Path(config.DATA_FOLDER) / self.dataset_name.value

        if self._is_data_on_disk():
            self.train_df = pd.read_csv(data_path / "train.csv")
            self.val_df = pd.read_csv(data_path / "val.csv")
            self.test_df = pd.read_csv(data_path / "test.csv")
        else:
            self.df = load_data(self.dataset_name)
            self.df[config.USER_COL] = LabelEncoder().fit_transform(
                self.df[config.USER_COL]
            )
            self.df[config.ITEM_COL] = LabelEncoder().fit_transform(
                self.df[config.ITEM_COL]
            )

            train_df, val_df, test_df = self._split()

            preprocessor = Preprocessor(
                train_df=train_df, val_df=val_df, test_df=test_df
            )

            self.train_df, self.val_df, self.test_df = preprocessor.fit_transform()

            if self.save_data:
                data_path.mkdir(parents=True, exist_ok=True)
                self.train_df.to_csv(data_path / "train.csv", index=False)
                self.val_df.to_csv(data_path / "val.csv", index=False)
                self.test_df.to_csv(data_path / "test.csv", index=False)

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
