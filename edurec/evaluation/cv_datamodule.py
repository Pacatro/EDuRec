import lightning as L
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .. import config
from ..datasets import DataProcessor, ElearningDataset
from ..datasets.data_processor import get_column_types
from ..datasets.loaders import RawDataset
from ..datasets.utils import collate_fn


class CvElearningDataModule(L.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame | RawDataset,
        batch_size: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        n_neg: int = config.N_NEG_TEST,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(df, RawDataset):
            self.df = DataProcessor.merge_raw_features(
                interactions_df=df.interactions,
                users_df=df.u_feats,
                items_df=df.i_feats,
            )
        else:
            self.df = df.copy()
        self.batch_size = batch_size
        self.random_state = random_state
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.n_neg = n_neg

        self.id_cols = [config.USER_COL, config.ITEM_COL]
        self.id_lengths: dict[str, int] = {}

        self._process_data()

        self.user_history = (
            self.df.groupby(config.USER_COL)[config.ITEM_COL].apply(set).to_dict()
        )
        self.all_item_ids = self.df[config.ITEM_COL].unique()
        self.item_catalog = self.item_features_df.set_index(config.ITEM_COL)

    def _process_data(self) -> None:
        self.has_time = config.TIME_COL in self.df.columns

        train_df = self.df.iloc[self.train_idx].reset_index(drop=True)
        val_df = self.df.iloc[self.val_idx].reset_index(drop=True)

        (
            self.numeric_cols,
            self.categorical_lengths,
            self.list_cols,
            self.text_cols,
        ) = get_column_types(train_df)

        preprocessor = DataProcessor(
            self.numeric_cols,
            self.cat_cols,
            self.text_cols,
            self.list_cols,
            self.id_cols,
            self.has_time,
        )

        self.train_df, self.val_df, _ = preprocessor.fit_transform(
            train_df=train_df, val_df=val_df, test_df=None
        )

        self.df = pd.concat([self.train_df, self.val_df], ignore_index=True)
        self.user_features_df, self.item_features_df = (
            preprocessor.split_entity_feature_frames(self.df)
        )
        self.user_feature_tensors, self.item_feature_tensors = (
            preprocessor.build_entity_tensors(self.df)
        )

    def setup(self, stage: str | None = None) -> None:
        if stage == "test":
            raise ValueError("Test data not available for this datamodule.")

        self.train_ds = ElearningDataset(
            self.train_df,
            id_cols=self.id_cols,
            numeric_cols=self.numeric_cols,
        )
        self.val_ds = ElearningDataset(
            self.val_df,
            n_negatives=self.n_neg,
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
    def cat_cols(self) -> list[str]:
        return list(self.categorical_lengths.keys())

    @property
    def cat_cardinalities(self) -> dict[str, int]:
        return {k: v + 2 for k, v in self.categorical_lengths.items()}

    @property
    def min_rating(self) -> float:
        return float(self.df[config.RATING_COL].min())

    @property
    def max_rating(self) -> float:
        return float(self.df[config.RATING_COL].max())
