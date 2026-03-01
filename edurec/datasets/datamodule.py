from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .. import config
from .data_processor import DataProcessor
from .elearnig_dataset import ElearningDataset
from .loaders import DatasetName, load_data


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
        self.save_hyperparameters()
        self.dataset_name = dataset
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.min_interactions = min_interactions
        self.n_neg_train = n_neg_train
        self.n_neg_val = n_neg_val
        self.n_neg_test = n_neg_test
        self.save_data = save_data
        self.random_state = random_state

        self.processed_path = (
            Path(config.DATA_FOLDER) / "preprocessed" / f"{self.dataset_name.value}.csv"
        )

        raw_dataset = load_data(dataset)
        self.interactions = raw_dataset.interactions
        self.users_feats = raw_dataset.u_feats
        self.items_feats = raw_dataset.i_feats
        self.schema = raw_dataset.schema

        # Dataset stats
        self.num_users = len(self.users_feats)
        self.num_items = len(self.items_feats)
        self.sparsity = 1 - len(self.interactions) / (self.num_users * self.num_items)
        self.min_rating = self.interactions[config.RATING_COL].min()

        self.data_processor = DataProcessor(schema=self.schema)
        self.is_processed = False

    def setup(self, stage: str | None = None) -> None:
        if not self.is_processed:
            self._train_inter, self._val_inter, self._test_inter = self._split_data()

            # Siempre fitteamos con el catálogo completo para evitar IDs -1
            self.data_processor.fit(
                users_train=self.users_feats,
                items_train=self.items_feats,
                interactions_train=self._train_inter,
            )

            processed_all = self.data_processor.transform(
                users=self.users_feats, items=self.items_feats
            )

            assert processed_all.users is not None and processed_all.items is not None

            self.user_feats_tensor = self._generate_tensor(
                processed_all.users, config.USER_COL
            )
            self.item_feats_tensor = self._generate_tensor(
                processed_all.items, config.ITEM_COL
            )

            self.is_processed = True

        match stage:
            case "fit" | None:
                p_train = self.data_processor.transform(interactions=self._train_inter)
                assert p_train.interactions is not None
                self.train_ds = ElearningDataset(
                    p_train.interactions, n_negatives=self.n_neg_train
                )

                p_val = self.data_processor.transform(interactions=self._val_inter)
                assert p_val.interactions is not None
                self.val_ds = ElearningDataset(
                    p_val.interactions, n_negatives=self.n_neg_val
                )
            case "test":
                p_test = self.data_processor.transform(interactions=self._test_inter)
                assert p_test.interactions is not None
                self.test_ds = ElearningDataset(
                    p_test.interactions, n_negatives=self.n_neg_test
                )

    def _generate_tensor(self, df: pd.DataFrame, id_col: str) -> torch.Tensor:
        df_sorted = df.sort_values(id_col)
        print(df_sorted)
        feat_cols = [c for c in df_sorted.columns if c != id_col]
        return torch.tensor(df_sorted[feat_cols].values, dtype=torch.float32)

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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader | None:
        if self.test_ds is None:
            return None

        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
        )
