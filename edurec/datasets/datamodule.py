from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from .. import settings
from .atomic_files import save_atomic_files
from .cache import ProcessedData, processed_cache_exists
from .dataprocessor import DataProcessor
from .loaders import DatasetName, RawData, Schema, load_raw_data
from .preprocessing import (
    add_relevance,
    clean_cols,
    filter_sparse,
    get_relevance_threshold,
    preprocess,
    split_data,
)
from .recsys_dataset import RecSysDataset
from .user_history import UserHistory, build_histories


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
        save_atomic_files: bool = False,
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
        self.save_atomic_files = save_atomic_files
        self.random_state = random_state

        self.processed_folder = Path(settings.PROCESSED_FOLDER) / dataset.value
        self.atomic_folder = Path(settings.ATOMICFILES_FOLDER) / dataset.value
        self.raw_dataset: RawData | None = None
        self.artifacts = ProcessedData()

        self.excluded_cols = {
            settings.USER_COL,
            settings.ITEM_COL,
            settings.RELEVANT_COL,
            settings.RATING_COL,
            settings.TIME_COL,
        }

        if self.use_processed_data and processed_cache_exists(self.processed_folder):
            self.artifacts = ProcessedData.load(self.processed_folder)
        else:
            raw = load_raw_data(dataset)
            self.raw_dataset = RawData(
                interactions=clean_cols(raw.interactions),
                user_features=clean_cols(raw.user_features),
                item_features=clean_cols(raw.item_features),
                schema=raw.schema,
            )
            self.artifacts.data_processor = DataProcessor(schema=raw.schema)

    def setup(self, stage: str | None = None):
        if not self.is_processed:
            if self.raw_dataset is None:
                raise RuntimeError("Raw dataset is not available.")

            users = self.raw_dataset.user_features
            items = self.raw_dataset.item_features
            interactions = self.raw_dataset.interactions

            if self.remove_sparse:
                users, items, interactions = filter_sparse(
                    users,
                    items,
                    interactions,
                    min_interactions=self.min_interactions,
                )

            train, val, test = split_data(
                interactions,
                test_ratio=self.test_ratio,
                val_ratio=self.val_ratio,
                min_interactions=self.min_interactions,
                random_state=self.random_state,
            )

            thresholds = get_relevance_threshold(train)
            train = add_relevance(train, thresholds)
            val = add_relevance(val, thresholds)
            test = add_relevance(test, thresholds)

            self.artifacts = preprocess(
                processor=self.data_processor,
                users=users,
                items=items,
                train=train,
                val=val,
                test=test,
            )
            self.artifacts.save(self.processed_folder)

        if self.save_atomic_files:
            self.atomic_files = save_atomic_files(
                self.artifacts,
                dataset_name=self.dataset_name.value,
                output_dir=self.atomic_folder,
            )

        # Build the sequential histories using only relevant (positive) interactions
        relevant_splits = {}
        for split in ("train", "val", "test"):
            df = getattr(self.artifacts, split)
            if df is not None:
                relevant_splits[split] = df[df[settings.RELEVANT_COL] > 0].reset_index(
                    drop=True
                )
            else:
                relevant_splits[split] = None

        histories = build_histories(
            relevant_splits,
            excluded_cols=self.excluded_cols,
        )

        if stage in ("fit", None):
            self.train_ds = self._make_dataset("train", histories)
            self.val_ds = self._make_dataset("val", histories)
        elif stage == "test":
            self.test_ds = self._make_dataset("test", histories)

    def _make_dataset(
        self,
        split: str,
        histories: dict[str, UserHistory],
    ) -> RecSysDataset:
        df = getattr(self.artifacts, split)
        if df is None:
            raise RuntimeError(f"Processed split {split} is not available.")

        # Filter dataset to positive interactions
        positive_mask = (df[settings.RELEVANT_COL] > 0).to_numpy(copy=True)
        interactions = df.loc[positive_mask].reset_index(drop=True)

        # histories[split] already aligns perfectly row-by-row with interactions
        history = histories[split]

        return RecSysDataset(
            interactions=interactions,
            history=history,
            num_ctx_feats=self.num_ctx_feats,
        )

    def build_inter_graph(self) -> Data:
        # We only build the graph based on the training interactions.
        interactions = self.artifacts.train

        assert interactions is not None, (
            "Data must be processed before creating the graph"
        )
        interactions = interactions[interactions[settings.RELEVANT_COL] > 0]

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
    def u_static_feats(self) -> torch.Tensor:
        if self.artifacts.u_static_feats is None:
            raise RuntimeError("Static features are not available.")

        return self.artifacts.u_static_feats

    @property
    def i_static_feats(self) -> torch.Tensor:
        if self.artifacts.i_static_feats is None:
            raise RuntimeError("Static features are not available.")

        return self.artifacts.i_static_feats

    @property
    def num_users(self) -> int:
        if self.artifacts.u_static_feats is not None:
            return self.artifacts.u_static_feats.shape[0]
        return 0 if self.raw_dataset is None else len(self.raw_dataset.user_features)

    @property
    def num_items(self) -> int:
        if self.artifacts.i_static_feats is not None:
            return self.artifacts.i_static_feats.shape[0]
        return 0 if self.raw_dataset is None else len(self.raw_dataset.item_features)

    @property
    def num_raw_users(self) -> int:
        return 0 if self.raw_dataset is None else len(self.raw_dataset.user_features)

    @property
    def num_raw_items(self) -> int:
        return 0 if self.raw_dataset is None else len(self.raw_dataset.item_features)

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
                    for col in self.raw_dataset.user_features.columns
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
                    for col in self.raw_dataset.item_features.columns
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
    def num_user_text_feats(self) -> int:
        metadata = self.data_processor.feature_metadata.get("users")
        return 0 if metadata is None else len(metadata.text_embedding_cols)

    @property
    def num_item_text_feats(self) -> int:
        metadata = self.data_processor.feature_metadata.get("items")
        return 0 if metadata is None else len(metadata.text_embedding_cols)

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
