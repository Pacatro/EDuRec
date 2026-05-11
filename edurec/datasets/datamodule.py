from dataclasses import dataclass
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from .. import settings
from .dataprocessor import DataProcessor
from .loaders import DatasetName, RawDataset, Schema, load_raw_data
from .ranker_dataset import RankerDataset
from .user_history import UserHistory


@dataclass
class ProcessedArtifacts:
    train: pd.DataFrame | None = None
    val: pd.DataFrame | None = None
    test: pd.DataFrame | None = None
    u_static_feats: torch.Tensor | None = None
    i_static_feats: torch.Tensor | None = None
    data_processor: DataProcessor | None = None

    @property
    def is_ready(self) -> bool:
        return all(
            value is not None
            for value in (
                self.train,
                self.val,
                self.test,
                self.u_static_feats,
                self.i_static_feats,
                self.data_processor,
            )
        )


class ElearningDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: DatasetName,
        batch_size: int,
        test_ratio: float,
        val_ratio: float,
        min_interactions: int = settings.MIN_INTERACTIONS,
        remove_sparse: bool = settings.REMOVE_SPARSE,
        use_processed_data: bool = False,
        random_state: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.min_interactions = min_interactions
        self.remove_sparse = remove_sparse
        self.use_processed_data = use_processed_data
        self.random_state = random_state

        self.processed_folder = Path(settings.PROCESSED_FOLDER) / dataset.value
        self.raw_dataset: RawDataset | None = None
        self.artifacts = ProcessedArtifacts()
        self.next_item_hist_by_split: dict[str, UserHistory] = {}

        self.excluded_cols = {
            settings.USER_COL,
            settings.ITEM_COL,
            settings.RELEVANT_COL,
            settings.RATING_COL,
            settings.TIME_COL,
        }

        cache_files = (
            "train.feather",
            "val.feather",
            "test.feather",
            "static_feats.safetensors",
            "processor.joblib",
        )

        if self.use_processed_data and all(
            (self.processed_folder / name).exists() for name in cache_files
        ):
            tensors = load_file(self.processed_folder / "static_feats.safetensors")
            self.artifacts = ProcessedArtifacts(
                train=pd.read_feather(self.processed_folder / "train.feather"),
                val=pd.read_feather(self.processed_folder / "val.feather"),
                test=pd.read_feather(self.processed_folder / "test.feather"),
                u_static_feats=tensors["u_static_feats"],
                i_static_feats=tensors["i_static_feats"],
                data_processor=DataProcessor.load(
                    self.processed_folder / "processor.joblib"
                ),
            )
        else:
            raw = load_raw_data(dataset)
            self.raw_dataset = RawDataset(
                interactions=self._clean_cols(raw.interactions),
                u_feats=self._clean_cols(raw.u_feats),
                i_feats=self._clean_cols(raw.i_feats),
                schema=raw.schema,
            )
            self.artifacts.data_processor = DataProcessor(schema=raw.schema)

    @property
    def is_processed(self) -> bool:
        return self.artifacts.is_ready

    @property
    def schema(self) -> Schema:
        if self.raw_dataset is not None:
            return self.raw_dataset.schema
        if self.artifacts.data_processor is not None:
            return self.artifacts.data_processor.schema
        raise RuntimeError("Schema is not available.")

    @property
    def data_processor(self) -> DataProcessor:
        if self.artifacts.data_processor is None:
            raise RuntimeError("Data processor is not available.")
        return self.artifacts.data_processor

    @property
    def u_static_feats(self) -> torch.Tensor | None:
        return self.artifacts.u_static_feats

    @property
    def i_static_feats(self) -> torch.Tensor | None:
        return self.artifacts.i_static_feats

    @property
    def num_users(self) -> int:
        if self.u_static_feats is not None:
            return self.u_static_feats.shape[0]
        return 0 if self.raw_dataset is None else len(self.raw_dataset.u_feats)

    @property
    def num_items(self) -> int:
        if self.i_static_feats is not None:
            return self.i_static_feats.shape[0]
        return 0 if self.raw_dataset is None else len(self.raw_dataset.i_feats)

    @property
    def num_raw_users(self) -> int:
        return 0 if self.raw_dataset is None else len(self.raw_dataset.u_feats)

    @property
    def num_raw_items(self) -> int:
        return 0 if self.raw_dataset is None else len(self.raw_dataset.i_feats)

    @property
    def num_interactions(self) -> int:
        if self.is_processed:
            return sum(
                len(df)
                for df in (
                    self.artifacts.train,
                    self.artifacts.val,
                    self.artifacts.test,
                )
                if df is not None
            )
        return 0 if self.raw_dataset is None else len(self.raw_dataset.interactions)

    @property
    def num_ctx_feats(self) -> int:
        if self.artifacts.train is not None:
            return len(
                [c for c in self.artifacts.train.columns if c not in self.excluded_cols]
            )
        if self.raw_dataset is not None:
            return len(
                [
                    c
                    for c in self.raw_dataset.interactions.columns
                    if c not in self.excluded_cols
                ]
            )
        return 0

    @property
    def sparsity(self) -> float:
        return (
            0.0
            if self.num_users == 0 or self.num_items == 0
            else 1 - self.num_interactions / (self.num_users * self.num_items)
        )

    @property
    def num_user_feats(self) -> int:
        metadata = self.data_processor.feature_metadata.get("users")
        if metadata is not None:
            return (
                len(metadata.dense_cols)
                + len(metadata.text_embedding_cols)
                + len(metadata.categorical_cols)
            )
        if self.raw_dataset is not None:
            return len(
                [
                    col
                    for col in self.raw_dataset.u_feats.columns
                    if col != settings.USER_COL
                ]
            )
        return 0

    @property
    def num_item_feats(self) -> int:
        metadata = self.data_processor.feature_metadata.get("items")
        if metadata is not None:
            return (
                len(metadata.dense_cols)
                + len(metadata.text_embedding_cols)
                + len(metadata.categorical_cols)
            )
        if self.raw_dataset is not None:
            return len(
                [
                    col
                    for col in self.raw_dataset.i_feats.columns
                    if col != settings.ITEM_COL
                ]
            )
        return 0

    @property
    def num_user_dense_feats(self) -> int:
        metadata = self.data_processor.feature_metadata.get("users")
        return (
            0
            if metadata is None
            else len(metadata.dense_cols) + len(metadata.text_embedding_cols)
        )

    @property
    def num_item_dense_feats(self) -> int:
        metadata = self.data_processor.feature_metadata.get("items")
        return (
            0
            if metadata is None
            else len(metadata.dense_cols) + len(metadata.text_embedding_cols)
        )

    @property
    def user_cat_cardinalities(self) -> list[int]:
        metadata = self.data_processor.feature_metadata.get("users")
        if metadata is None:
            return []
        return [
            metadata.categorical_cardinalities[col] for col in metadata.categorical_cols
        ]

    @property
    def item_cat_cardinalities(self) -> list[int]:
        metadata = self.data_processor.feature_metadata.get("items")
        if metadata is None:
            return []
        return [
            metadata.categorical_cardinalities[col] for col in metadata.categorical_cols
        ]

    def setup(self, stage: str | None = None):
        if not self.is_processed:
            if self.raw_dataset is None:
                raise RuntimeError("Raw dataset is not available.")

            train, val, test = self._split_data(self.raw_dataset.interactions)
            users = self.raw_dataset.u_feats
            items = self.raw_dataset.i_feats

            if self.remove_sparse:
                users, items, train, val, test = self._filter_sparse(
                    users, items, train, val, test
                )

            train = self._add_relevance(train)
            val = self._add_relevance(val)
            test = self._add_relevance(test)
            self.artifacts = self._preprocess(users, items, train, val, test)
            self._save_processed_data()

        self._build_histories()

        if stage in ("fit", None):
            self.train_ds = self._make_dataset("train")
            self.val_ds = self._make_dataset("val")
        elif stage == "test":
            self.test_ds = self._make_dataset("test")

    def _clean_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = (
            df.columns.str.lower()
            .str.strip()
            .str.replace(" ", "_")
            .str.replace(r"[^\w]", "", regex=True)
        )
        return df

    def _split_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(self.random_state)
        has_time = settings.TIME_COL in df.columns
        parts = {"train": [], "val": [], "test": []}

        for _, user_df in df.groupby(settings.USER_COL, sort=False):
            if has_time:
                user_df = user_df.sort_values(settings.TIME_COL, kind="mergesort")

            n = len(user_df)
            if n < self.min_interactions:
                parts["train"].append(user_df)
                continue

            n_test = max(1, int(np.floor(n * self.test_ratio)))
            n_val = max(1, int(np.floor(n * self.val_ratio)))
            if n_test + n_val >= n:
                n_test = n_val = 1

            if has_time:
                parts["train"].append(user_df.iloc[: -(n_test + n_val)])
                parts["val"].append(user_df.iloc[-(n_test + n_val) : -n_test])
                parts["test"].append(user_df.iloc[-n_test:])
            else:
                order = rng.permutation(n)
                parts["test"].append(user_df.iloc[order[:n_test]])
                parts["val"].append(user_df.iloc[order[n_test : n_test + n_val]])
                parts["train"].append(user_df.iloc[order[n_test + n_val :]])

        train_split = pd.concat(parts["train"], axis=0).reset_index(drop=True)
        val_split = pd.concat(parts["val"], axis=0).reset_index(drop=True)
        test_split = pd.concat(parts["test"], axis=0).reset_index(drop=True)

        return train_split, val_split, test_split

    def _filter_sparse(
        self,
        users: pd.DataFrame,
        items: pd.DataFrame,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        while True:
            n_prev = len(train)
            valid_users = train[settings.USER_COL].value_counts(sort=False)
            valid_items = train[settings.ITEM_COL].value_counts(sort=False)
            valid_users = valid_users[valid_users >= self.min_interactions].index
            valid_items = valid_items[valid_items >= self.min_interactions].index
            mask = train[settings.USER_COL].isin(valid_users) & train[
                settings.ITEM_COL
            ].isin(valid_items)
            train = train.loc[mask].reset_index(drop=True)
            if len(train) == n_prev:
                break

        user_mask = users[settings.USER_COL].isin(valid_users)
        item_mask = items[settings.ITEM_COL].isin(valid_items)
        val_mask = val[settings.USER_COL].isin(valid_users) & val[
            settings.ITEM_COL
        ].isin(valid_items)
        test_mask = test[settings.USER_COL].isin(valid_users) & test[
            settings.ITEM_COL
        ].isin(valid_items)

        return (
            users.loc[user_mask].reset_index(drop=True),
            items.loc[item_mask].reset_index(drop=True),
            train,
            val.loc[val_mask].reset_index(drop=True),
            test.loc[test_mask].reset_index(drop=True),
        )

    def _add_relevance(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if settings.RELEVANT_COL in df.columns:
            return df.reset_index(drop=True)

        if settings.RATING_COL not in df.columns:
            df[settings.RELEVANT_COL] = 1
            return df.reset_index(drop=True)

        user_ratings = df.groupby(settings.USER_COL)[settings.RATING_COL]
        threshold = np.where(
            user_ratings.transform("count") < self.min_interactions,
            df[settings.RATING_COL].mean(),
            user_ratings.transform("mean"),
        )
        df[settings.RELEVANT_COL] = df[settings.RATING_COL] >= threshold
        return df.reset_index(drop=True)

    def _preprocess(
        self,
        users: pd.DataFrame,
        items: pd.DataFrame,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> ProcessedArtifacts:
        processor = self.data_processor
        processor.fit(users_train=users, items_train=items, interactions_train=train)

        entities = processor.transform(users=users, items=items)
        splits = {
            "train": processor.transform(interactions=train),
            "val": processor.transform(interactions=val),
            "test": processor.transform(interactions=test),
        }

        if entities.users is None or entities.items is None:
            raise RuntimeError("User/item features were not processed.")

        users_df = entities.users
        items_df = entities.items
        if entities.text_embeddings["users"] is not None:
            users_df = pd.concat([users_df, entities.text_embeddings["users"]], axis=1)
        if entities.text_embeddings["items"] is not None:
            items_df = pd.concat([items_df, entities.text_embeddings["items"]], axis=1)

        split_dfs: dict[str, pd.DataFrame] = {}
        for name, processed in splits.items():
            if processed.interactions is None:
                raise RuntimeError(f"{name} interactions were not processed.")
            split_dfs[name] = processed.interactions
            if processed.text_embeddings["inter"] is not None:
                split_dfs[name] = pd.concat(
                    [split_dfs[name], processed.text_embeddings["inter"]],
                    axis=1,
                )
            split_dfs[name] = split_dfs[name].reset_index(drop=True)

        static_feats = {}
        for name, df, prefix, id_col in (
            ("users", users_df, "users", settings.USER_COL),
            ("items", items_df, "items", settings.ITEM_COL),
        ):
            metadata = processor.feature_metadata[prefix]
            cols = (
                metadata.dense_cols
                + metadata.text_embedding_cols
                + metadata.categorical_cols
            )
            static_feats[name] = torch.as_tensor(
                df.sort_values(id_col)[cols].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )

        return ProcessedArtifacts(
            train=split_dfs["train"],
            val=split_dfs["val"],
            test=split_dfs["test"],
            u_static_feats=static_feats["users"],
            i_static_feats=static_feats["items"],
            data_processor=processor,
        )

    def _build_histories(self) -> None:
        self.next_item_hist_by_split = {}
        user_state: dict[int, list[tuple[int, list[float]]]] = {}

        for split in ("train", "val", "test"):
            df = getattr(self.artifacts, split)
            if df is None:
                raise RuntimeError(f"Processed split {split} is not available.")

            ctx_cols = [col for col in df.columns if col not in self.excluded_cols]
            history = UserHistory(
                items=torch.zeros(
                    (len(df), settings.MAX_HISTORY_LEN), dtype=torch.long
                ),
                ctx=torch.zeros(
                    (len(df), settings.MAX_HISTORY_LEN, len(ctx_cols)),
                    dtype=torch.float32,
                ),
                valid_mask=torch.zeros(
                    (len(df), settings.MAX_HISTORY_LEN), dtype=torch.bool
                ),
            )
            users = df[settings.USER_COL].to_numpy(dtype=np.int64)
            items = df[settings.ITEM_COL].to_numpy(dtype=np.int64)
            ctx = (
                df[ctx_cols].to_numpy(dtype=np.float32)
                if ctx_cols
                else np.empty((len(df), 0), dtype=np.float32)
            )

            for row_idx, user_id in enumerate(users):
                past = user_state.get(int(user_id), [])[-settings.MAX_HISTORY_LEN :]
                if past:
                    history.items[row_idx, : len(past)] = torch.tensor(
                        [item_id + 1 for item_id, _ in past],
                        dtype=torch.long,
                    )
                    history.valid_mask[row_idx, : len(past)] = True
                    if ctx_cols:
                        history.ctx[row_idx, : len(past)] = torch.tensor(
                            [values for _, values in past],
                            dtype=torch.float32,
                        )

                user_state.setdefault(int(user_id), []).append(
                    (int(items[row_idx]), ctx[row_idx].tolist())
                )

            self.next_item_hist_by_split[split] = history

    def _save_processed_data(self) -> None:
        self.processed_folder.mkdir(parents=True, exist_ok=True)

        for split in ("train", "val", "test"):
            df = getattr(self.artifacts, split)
            if df is not None:
                df.to_feather(self.processed_folder / f"{split}.feather")

        if self.u_static_feats is None or self.i_static_feats is None:
            raise RuntimeError("Static features are not available.")

        save_file(
            {
                "u_static_feats": self.u_static_feats.contiguous(),
                "i_static_feats": self.i_static_feats.contiguous(),
            },
            self.processed_folder / "static_feats.safetensors",
        )
        self.data_processor.save(self.processed_folder / "processor.joblib")

    def _make_dataset(self, split: str) -> RankerDataset:
        df = getattr(self.artifacts, split)
        if df is None:
            raise RuntimeError(f"Processed split {split} is not available.")

        positive_mask = (df[settings.RELEVANT_COL] > 0).to_numpy(copy=True)
        history_mask = torch.as_tensor(positive_mask, dtype=torch.bool)
        history = self.next_item_hist_by_split[split]

        return RankerDataset(
            interactions=df.loc[positive_mask].reset_index(drop=True),
            precomputed_history=UserHistory(
                items=history.items[history_mask],
                ctx=history.ctx[history_mask],
                valid_mask=history.valid_mask[history_mask],
            ),
            num_ctx_feats=self.num_ctx_feats,
        )

    def build_inter_graph(self, split: str) -> Data:
        interactions = getattr(self.artifacts, split)

        user_idx = torch.as_tensor(
            interactions[settings.USER_COL].to_numpy(copy=True),
            dtype=torch.long,
        )
        item_idx = (
            torch.as_tensor(
                interactions[settings.ITEM_COL].to_numpy(copy=True),
                dtype=torch.long,
            )
            + self.num_users
        )

        edge_index = torch.cat(
            [
                torch.stack([user_idx, item_idx], dim=0),
                torch.stack([item_idx, user_idx], dim=0),
            ],
            dim=1,
        ).contiguous()

        graph = Data(edge_index=edge_index, num_nodes=self.num_users + self.num_items)
        graph.num_users = self.num_users
        graph.num_items = self.num_items
        graph.node_type = torch.cat(
            [
                torch.zeros(self.num_users, dtype=torch.long),
                torch.ones(self.num_items, dtype=torch.long),
            ]
        )
        return graph

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=settings.NUM_WORKERS,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=settings.NUM_WORKERS,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=settings.NUM_WORKERS,
            shuffle=False,
        )
