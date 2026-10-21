from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData

from edurec import settings
from edurec.datasets.cache import (
    CACHE_VERSION,
    ProcessedData,
    processed_cache_exists,
    read_manifest,
    source_fingerprint,
)
from edurec.datasets.features.graph import build_knowledge_graph
from edurec.datasets.features.interactions import (
    InteractionFeatureTables,
    build_interaction_features,
)
from edurec.datasets.features.tabular import DataProcessor
from edurec.datasets.pipeline.build import BuildConfig, build_processed
from edurec.datasets.pipeline.prepare import selection_metadata
from edurec.datasets.pipeline.splits import build_histories, generate_negative_samples
from edurec.datasets.recbole_export import save_atomic_files
from edurec.datasets.registry import (
    DatasetName,
    RawData,
    download_raw_data,
    get_spec,
    load_raw_data,
)
from edurec.datasets.torch_data import RecSysDataset


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
        max_history: int = settings.MAX_HISTORY_LEN,
        deduplicate_interactions: bool = settings.DEDUPLICATE_INTERACTIONS,
        persistent_workers: bool = True,
        max_users: int | None = settings.SELECTION_MAX_USERS,
        selection_seed: int = settings.SELECTION_SEED,
        max_interactions: int | None = settings.SELECTION_MAX_INTERACTIONS,
        max_items: int | None = None,
        community_fields: list[str] | None = None,
        min_history_retention: float = 0.7,
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
        self.max_history = max_history
        self.deduplicate_interactions = deduplicate_interactions
        self.persistent_workers = persistent_workers
        self.data_variant = dataset.value
        self.max_interactions = max_interactions
        self.max_users = max_users
        self.selection_seed = selection_seed
        self.max_items = max_items
        self.community_fields = community_fields
        self.min_history_retention = min_history_retention

        self.cache_params = {
            "version": CACHE_VERSION,
            "dataset": dataset.value,
            "min_interactions": min_interactions,
            "test_ratio": test_ratio,
            "val_ratio": val_ratio,
            "random_state": random_state,
            "remove_sparse": remove_sparse,
            "deduplicate_interactions": deduplicate_interactions,
            "feature_types": list(settings.PREPROCESS_FEATURE_TYPES),
            "text_embedding_model": settings.TEXT_EMBEDDING_MODEL,
            "text_embedding_dim": settings.TEXT_EMBEDDING_DIM,
            "text_max_tokens": settings.TEXT_MAX_TOKENS,
            "max_history_len": max_history,
        }

        self.cache_params["selection"] = selection_metadata(
            self.max_users,
            self.selection_seed,
            self.max_interactions,
            max_items=max_items,
            community_fields=community_fields,
            min_interactions=min_interactions,
            min_history_retention=min_history_retention,
        )
        if dataset == DatasetName.MOOCCUBEX:
            self.cache_params["encode_concept_text"] = (
                settings.MOOCCUBEX_ENCODE_CONCEPT_TEXT
            )

        self.processed_folder = Path(settings.PROCESSED_FOLDER) / self.data_variant
        self.atomic_folder = Path(settings.ATOMICFILES_FOLDER) / self.data_variant

        self.artifacts = ProcessedData()
        self._knowledge_graph: HeteroData | None = None
        self._interaction_tables: InteractionFeatureTables | None = None

    @property
    def raw_dataset(self) -> RawData | None:
        """Selected source tables from the last build; None when loaded from cache."""
        return self.artifacts.selected_raw

    def _source_metadata(self) -> dict[str, dict[str, int]]:
        return source_fingerprint(
            settings.RAW_DATA_FOLDER / get_spec(self.dataset_name).folder
        )

    def _compatible_cache(self) -> bool:
        return processed_cache_exists(
            self.processed_folder,
            expected_selection=self.cache_params["selection"],
            expected_params={
                **self.cache_params,
                "raw_sources": self._source_metadata(),
            },
        )

    def prepare_data(self) -> None:
        if self.use_processed_data and self._compatible_cache():
            return

        download_raw_data(self.dataset_name)

    def setup(self, stage: str | None = None):
        if not self.is_processed:
            if self.use_processed_data and self._compatible_cache():
                self.artifacts = ProcessedData.load(self.processed_folder)
                manifest = read_manifest(self.processed_folder)
                for key in ("selection_report", "split_report", "raw_sources"):
                    if key in manifest:
                        self.cache_params[key] = manifest[key]
            else:
                self._process_raw_data()

        splits = self.artifacts.splits()
        relevant_splits = {
            split: df.loc[df[settings.RELEVANT_COL] > 0].reset_index(drop=True)
            for split, df in splits.items()
        }
        for split, frame in relevant_splits.items():
            if frame.empty:
                raise ValueError(
                    f"{self.data_variant}: {split} split has no positive interactions. "
                    "Check filtering, relevance thresholds and split ratios."
                )
            for column, size in (
                (settings.USER_COL, self.num_users),
                (settings.ITEM_COL, self.num_items),
            ):
                if not frame[column].between(0, size - 1).all():
                    raise ValueError(
                        f"{self.data_variant}: {split} contains invalid {column} values."
                    )

        if self.save_atomic_files:
            self.atomic_files = save_atomic_files(
                self.artifacts,
                dataset_name=self.data_variant,
                output_dir=self.atomic_folder,
            )

        self._interaction_tables = build_interaction_features(
            relevant_splits, self.data_processor
        )

        histories = build_histories(
            relevant_splits,
            max_history=self.max_history,
            enabled=self.has_temporal_order,
        )

        if stage in ("fit", None):
            train_negatives = None

            if not self.is_explicit:
                train_interactions = relevant_splits["train"]
                all_observed = pd.concat(
                    relevant_splits.values(),
                    ignore_index=True,
                )

                train_negatives = generate_negative_samples(
                    interactions=train_interactions,
                    item_ids=np.arange(self.num_items),
                    num_negatives=settings.TRAIN_NEGATIVES_PER_POSITIVE,
                    random_state=self.random_state,
                    observed_interactions=all_observed,
                )

            self.train_ds = self._make_dataset(
                relevant_splits["train"],
                histories["train"],
                negative_item_ids=train_negatives,
            )

            self.val_ds = self._make_dataset(relevant_splits["val"], histories["val"])

        elif stage == "test":
            self.test_ds = self._make_dataset(
                relevant_splits["test"],
                histories["test"],
            )

    def validate_topks(self, topks: list[int]) -> None:
        """Reject impossible ranking cutoffs before any model is trained."""
        if not topks or any(k < 1 or k > self.num_items for k in topks):
            raise ValueError(
                f"{self.data_variant}: top-k values must be in [1, {self.num_items}], "
                f"got {topks}."
            )

    def _make_dataset(
        self,
        interactions: pd.DataFrame,
        history: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        negative_item_ids: np.ndarray | None = None,
    ) -> RecSysDataset:
        history_items, history_valid_mask, history_context_index = history
        tables = self.interaction_tables

        return RecSysDataset(
            interactions=interactions,
            history_items=history_items,
            history_valid_mask=history_valid_mask,
            history_context_index=history_context_index,
            interaction_dense=tables.dense,
            interaction_cat=tables.cat,
            interaction_timestamps=tables.timestamps,
            negative_item_ids=negative_item_ids,
        )

    def _build_config(self) -> BuildConfig:
        return BuildConfig(
            test_ratio=self.test_ratio,
            val_ratio=self.val_ratio,
            min_interactions=self.min_interactions,
            remove_sparse=self.remove_sparse,
            deduplicate=self.deduplicate_interactions,
            random_state=self.random_state,
            max_users=self.max_users,
            max_interactions=self.max_interactions,
            selection_seed=self.selection_seed,
            max_items=self.max_items,
            community_fields=self.community_fields,
            min_history_retention=self.min_history_retention,
        )

    def _process_raw_data(self) -> None:
        self.cache_params["raw_sources"] = self._source_metadata()
        self.artifacts = build_processed(
            load_raw_data(self.dataset_name), self._build_config()
        )
        self.cache_params.update(self.artifacts.reports)
        self.artifacts.save(self.processed_folder, manifest=self.cache_params)

    @property
    def knowledge_graph(self) -> HeteroData:
        if self._knowledge_graph is None:
            artifacts = self.artifacts

            if artifacts.i_static_feats is None or artifacts.item_features is None:
                raise RuntimeError("Knowledge graph requires item data.")

            self._knowledge_graph = build_knowledge_graph(
                artifacts.item_features,
                artifacts.i_static_feats,
                self.data_processor,
            )

        return self._knowledge_graph

    @property
    def kg_node_counts(self) -> dict[str, int]:
        return {
            node: self.knowledge_graph[node].num_nodes
            for node in self.knowledge_graph.node_types
            if node not in {"user", "item"}
        }

    @property
    def kg_edge_types(self) -> list[tuple[str, str, str]]:
        return list(self.knowledge_graph.edge_types)

    def _data_generator(self) -> torch.Generator | None:
        if self.random_state is None:
            return None

        generator = torch.Generator()
        generator.manual_seed(int(self.random_state))

        return generator

    def _dataloader(
        self,
        dataset: RecSysDataset,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=settings.NUM_WORKERS,
            shuffle=shuffle,
            persistent_workers=self.persistent_workers and settings.NUM_WORKERS > 0,
            generator=self._data_generator() if shuffle else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_ds, shuffle=False)

    @property
    def is_processed(self) -> bool:
        return self.artifacts.is_ready

    @property
    def feedback_type(self) -> str:
        return "explicit" if self.is_explicit else "implicit"

    @property
    def is_explicit(self) -> bool:
        interactions = (
            self.raw_dataset.interactions
            if self.raw_dataset is not None
            else self.artifacts.train
        )

        if interactions is None:
            raise RuntimeError("Interactions are not available.")

        return settings.RATING_COL in interactions.columns

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
    def u_cat_feats(self) -> torch.Tensor:
        if self.artifacts.u_cat_feats is None:
            raise RuntimeError("User categorical features are not available.")

        return self.artifacts.u_cat_feats

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
    def has_history(self) -> bool:
        train_ds = getattr(self, "train_ds", None)

        return bool(
            self.has_temporal_order
            and train_ds is not None
            and train_ds.history_valid_mask.numel() > 0
            and train_ds.history_valid_mask.any().item()
        )

    @property
    def has_temporal_order(self) -> bool:
        interactions = (
            self.raw_dataset.interactions
            if self.raw_dataset is not None
            else self.artifacts.train
        )

        return interactions is not None and settings.TIME_COL in interactions.columns

    @property
    def split_strategy(self) -> str:
        return "temporal" if self.has_temporal_order else "random"

    @property
    def sparsity(self) -> float:
        if self.num_users == 0 or self.num_items == 0:
            return 0.0

        return 1 - (self.num_interactions / (self.num_users * self.num_items))

    @property
    def num_user_feats(self) -> int:
        metadata = self.data_processor.feature_metadata["users"]

        if metadata is not None:
            return (
                len(metadata.numeric_cols)
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
        metadata = self.data_processor.feature_metadata["items"]

        if metadata is not None:
            return (
                len(metadata.numeric_cols)
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
        metadata = self.data_processor.feature_metadata["users"]

        return (
            0
            if metadata is None
            else (len(metadata.numeric_cols) + len(metadata.text_embedding_cols))
        )

    @property
    def num_item_dense_feats(self) -> int:
        metadata = self.data_processor.feature_metadata["items"]

        return (
            0
            if metadata is None
            else (len(metadata.numeric_cols) + len(metadata.text_embedding_cols))
        )

    @property
    def interaction_tables(self) -> InteractionFeatureTables:
        if self._interaction_tables is None:
            self._interaction_tables = build_interaction_features(
                self.artifacts.splits(), self.data_processor
            )

        return self._interaction_tables

    @property
    def num_interaction_dense_feats(self) -> int:
        return len(self.interaction_tables.dense_cols)

    @property
    def interaction_cat_cardinalities(self) -> list[int]:
        return list(self.interaction_tables.cat_cardinalities)

    @property
    def num_user_text_feats(self) -> int:
        metadata = self.data_processor.feature_metadata["users"]

        return 0 if metadata is None else len(metadata.text_embedding_cols)

    @property
    def num_item_text_feats(self) -> int:
        metadata = self.data_processor.feature_metadata["items"]

        return 0 if metadata is None else len(metadata.text_embedding_cols)

    @property
    def user_cat_cardinalities(self) -> list[int]:
        metadata = self.data_processor.feature_metadata["users"]

        if metadata is None:
            return []

        return [
            metadata.categorical_cardinalities[col] for col in metadata.categorical_cols
        ]

    @property
    def item_cat_cardinalities(self) -> list[int]:
        metadata = self.data_processor.feature_metadata["items"]

        if metadata is None:
            return []

        return [
            metadata.categorical_cardinalities[col] for col in metadata.categorical_cols
        ]
