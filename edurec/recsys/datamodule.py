from pathlib import Path

import lightning as L
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .. import config
from ..datasets import (
    DataProcessor,
    DatasetName,
    ElearningDataset,
    load_data,
)
from ..datasets.utils import collate_fn


class ElearningDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: DatasetName,
        batch_size: int,
        test_ratio: float,
        val_ratio: float,
        min_interactions: int = config.MIN_INTERACTIONS,
        n_neg_train: int = config.N_NEG_TRAIN,
        n_neg_val: int = config.N_NEG_VAL,
        n_neg_test: int = config.N_NEG_TEST,
        save_data: bool = False,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.min_interactions = min_interactions
        self.random_state = random_state
        self.save_data = save_data
        self.n_neg_train = n_neg_train
        self.n_neg_val = n_neg_val
        self.n_neg_test = n_neg_test

        self.has_time = False

        self.processed_path = (
            Path(config.DATA_FOLDER) / "preprocessed" / f"{self.dataset_name.value}.csv"
        )

        raw_dataset = load_data(dataset)
        self.interactions = raw_dataset.interactions
        self.users_feats = raw_dataset.u_feats
        self.items_feats = raw_dataset.i_feats
        self.schema = raw_dataset.schema

        self._process_data()

        # Dataset stats
        self.num_users = len(self.users_feats)
        self.num_items = len(self.items_feats)
        self.sparsity = 1 - len(self.interactions) / (self.num_users * self.num_items)
        self.min_rating = self.interactions[config.RATING_COL].min()

        # self.user_history = (
        #     self.df.groupby(config.USER_COL)[config.ITEM_COL].apply(set).to_dict()
        # )
        # self.all_item_ids = self.df[config.ITEM_COL].unique()
        # self.item_catalog = self.item_features_df.set_index(config.ITEM_COL)

    def _split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = self.interactions
        rng = np.random.default_rng(self.random_state)

        if config.TIME_COL in df.columns:
            df = df.sort_values(by=config.TIME_COL, kind="mergesort")

        train_parts, val_parts, test_parts = [], [], []

        for _, g in df.groupby(config.USER_COL, sort=False):
            n = len(g)

            if n < self.min_interactions:
                train_parts.append(g)
                continue

            n_test = np.maximum(1, int(np.floor(n * self.test_ratio)))
            n_val = np.maximum(1, int(np.floor(n * self.val_ratio)))

            if n_test + n_val >= n:
                n_test, n_val = 1, 1

            if config.TIME_COL in g.columns:
                test_g = g.iloc[-n_test:]
                val_g = g.iloc[-(n_test + n_val) : -n_test]
                train_g = g.iloc[: -(n_test + n_val)]
            else:
                idx = g.index.to_numpy()
                rng.shuffle(idx)

                test_idx = idx[:n_test]
                val_idx = idx[n_test : n_test + n_val]
                train_idx = idx[n_test + n_val :]

                test_g = g.loc[test_idx]
                val_g = g.loc[val_idx]
                train_g = g.loc[train_idx]

            train_parts.append(train_g)
            test_parts.append(test_g)
            val_parts.append(val_g)

        train_df = (
            pd.concat(train_parts, axis=0)
            .sample(frac=1, random_state=self.random_state)
            .reset_index(drop=True)
        )

        val_df = pd.concat(val_parts, axis=0).reset_index(drop=True)
        test_df = pd.concat(test_parts, axis=0).reset_index(drop=True)

        return train_df, val_df, test_df

    def _process_data(self) -> None:
        self.train_inter, self.val_inter, self.test_inter = self._split_data()
        self.has_time = config.TIME_COL in self.interactions.columns

        self.data_processor = DataProcessor(schema=self.schema)

        train_user_ids = self.train_inter[config.USER_COL].unique()
        train_item_ids = self.train_inter[config.ITEM_COL].unique()
        val_user_ids = self.val_inter[config.USER_COL].unique()
        val_item_ids = self.val_inter[config.ITEM_COL].unique()
        test_user_ids = self.test_inter[config.USER_COL].unique()
        test_item_ids = self.test_inter[config.ITEM_COL].unique()

        def _subset_or_full(
            df: pd.DataFrame, id_col: str, ids: np.ndarray
        ) -> pd.DataFrame:
            subset = df[df[id_col].isin(ids)].reset_index(drop=True)
            return subset if not subset.empty else df.copy()

        users_train = _subset_or_full(self.users_feats, config.USER_COL, train_user_ids)
        items_train = _subset_or_full(self.items_feats, config.ITEM_COL, train_item_ids)
        users_val = _subset_or_full(self.users_feats, config.USER_COL, val_user_ids)
        items_val = _subset_or_full(self.items_feats, config.ITEM_COL, val_item_ids)
        users_test = _subset_or_full(self.users_feats, config.USER_COL, test_user_ids)
        items_test = _subset_or_full(self.items_feats, config.ITEM_COL, test_item_ids)

        self.data_processor.fit(
            users_train=users_train,
            items_train=items_train,
            interactions_train=self.train_inter,
        )

        self.train_processed = self.data_processor.transform(
            users=users_train,
            items=items_train,
            interactions=self.train_inter,
        )
        self.val_processed = self.data_processor.transform(
            users=users_val,
            items=items_val,
            interactions=self.val_inter,
        )
        self.test_processed = self.data_processor.transform(
            users=users_test,
            items=items_test,
            interactions=self.test_inter,
        )

        if self.save_data:
            self.processed_path.parent.mkdir(parents=True, exist_ok=True)
            self.train_inter.to_csv(
                self.processed_path.with_name(
                    f"{self.dataset_name.value}_train_inter.csv"
                ),
                index=False,
            )
            self.val_inter.to_csv(
                self.processed_path.with_name(
                    f"{self.dataset_name.value}_val_inter.csv"
                ),
                index=False,
            )
            self.test_inter.to_csv(
                self.processed_path.with_name(
                    f"{self.dataset_name.value}_test_inter.csv"
                ),
                index=False,
            )

    # def setup(self, stage: str | None = None) -> None:
    #     match stage:
    #         case "fit":
    #             self.train_ds = ElearningDataset(
    #                 self.train_df,
    #                 n_negatives=self.n_neg_train,
    #                 min_rating=self.min_rating,
    #                 # item_catalog=self.item_catalog,
    #                 # user_history=self.user_history,
    #                 # all_item_ids=self.all_item_ids,
    #                 # numeric_cols=self.numeric_cols,
    #             )
    #             self.val_ds = ElearningDataset(
    #                 self.val_df,
    #                 n_negatives=self.n_neg_val,
    #                 min_rating=self.min_rating,
    #                 # item_catalog=self.item_catalog,
    #                 # user_history=self.user_history,
    #                 # all_item_ids=self.all_item_ids,
    #                 # numeric_cols=self.numeric_cols,
    #             )
    #         case "test":
    #             if self.test_df is not None:
    #                 self.test_ds = ElearningDataset(
    #                     self.test_df,
    #                     n_negatives=self.n_neg_test,
    #                     min_rating=self.min_rating,
    #                     # item_catalog=self.item_catalog,
    #                     # user_history=self.user_history,
    #                     # all_item_ids=self.all_item_ids,
    #                     # numeric_cols=self.numeric_cols,
    #                 )

    # def train_dataloader(self) -> DataLoader:
    #     return DataLoader(
    #         self.train_ds,
    #         batch_size=self.batch_size,
    #         num_workers=config.NUM_WORKERS,
    #         collate_fn=collate_fn,
    #     )
    #
    # def val_dataloader(self) -> DataLoader:
    #     return DataLoader(
    #         self.val_ds,
    #         batch_size=self.batch_size,
    #         num_workers=config.NUM_WORKERS,
    #         collate_fn=collate_fn,
    #     )
    #
    # def test_dataloader(self) -> DataLoader | None:
    #     if self.test_ds is None:
    #         return None
    #
    #     return DataLoader(
    #         self.test_ds,
    #         batch_size=self.batch_size,
    #         num_workers=config.NUM_WORKERS,
    #         collate_fn=collate_fn,
    #     )

    # @property
    # def cat_cardinalities(self) -> dict[str, int]:
    #     return {k: v + 2 for k, v in self.categorical_lengths.items()}
