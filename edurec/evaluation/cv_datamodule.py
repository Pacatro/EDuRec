import lightning as L
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from .. import config
from ..data.datamodule import ElearningDataset
from ..data.datasets import DatasetName, load_raw_data
from ..data.preprocessor import Preprocessor


class CvElearningDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: DatasetName,
        batch_size: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset
        self.batch_size = batch_size
        self.random_state = random_state
        self.train_idx = train_idx
        self.val_idx = val_idx

        self.preprocessor = Preprocessor()
        self.df = load_raw_data(self.dataset_name)
        self._process_data()

    def _process_data(self) -> None:
        print("Preprocessing data")
        self.df[config.USER_COL] = LabelEncoder().fit_transform(
            self.df[config.USER_COL]
        )
        self.df[config.ITEM_COL] = LabelEncoder().fit_transform(
            self.df[config.ITEM_COL]
        )

        train_df = self.df.iloc[self.train_idx].reset_index(drop=True)
        val_df = self.df.iloc[self.val_idx].reset_index(drop=True)

        self.train_df, self.val_df, _ = self.preprocessor.fit_transform(
            train_df=train_df, val_df=val_df, test_df=None
        )

    def setup(self, stage: str | None = None) -> None:
        if stage == "test":
            raise ValueError("Test data not available for this datamodule.")

        self.train_ds = ElearningDataset(self.train_df)
        self.val_ds = ElearningDataset(self.val_df)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=config.NUM_WORKERS
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=config.NUM_WORKERS
        )

    @property
    def num_users(self) -> int:
        return int(self.df[config.USER_COL].nunique())

    @property
    def num_items(self) -> int:
        return int(self.df[config.ITEM_COL].nunique())

    @property
    def numeric_features(self) -> list[str]:
        return self.preprocessor.numeric_cols

    @property
    def cat_cardinalities(self) -> dict[str, int]:
        return self.preprocessor.categorical_lengths

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
