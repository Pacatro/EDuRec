"""Lightning data module: assemble queries and data loaders.

All data views (interaction context, histories, negatives, knowledge graph and
RecBole exports) come from :class:`~edurec.datasets.adapters.DataAdapter`. This
module only builds/loads the canonical ``ProcessedData``, validates the cache
fingerprint and serves ``DataLoader`` instances.
"""

import warnings
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from edurec import settings
from edurec.datasets.adapters import DataAdapter
from edurec.datasets.adapters.queries import QueryConfig
from edurec.datasets.cache import ProcessedData, replace_folder
from edurec.datasets.features.tabular import DataProcessor
from edurec.datasets.pipeline.build import (
    BuildConfig,
    build_processed,
    expected_manifest,
    split_strategy_for,
)
from edurec.datasets.registry import (
    DatasetName,
    RawData,
    download_raw_data,
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

        self.processed_folder = Path(settings.PROCESSED_FOLDER) / self.data_variant
        self.atomic_folder = Path(settings.ATOMICFILES_FOLDER) / self.data_variant

        self.artifacts = ProcessedData()
        self._adapter: DataAdapter | None = None

    @property
    def raw_dataset(self) -> RawData | None:
        """Selected source tables from the last build; None when loaded from cache."""
        return self.artifacts.selected_raw

    @property
    def adapter(self) -> DataAdapter:
        """Lazy facade over the processed artifacts."""
        if self._adapter is None:
            if not self.is_processed:
                raise RuntimeError(
                    f"{self.data_variant}: data is not prepared; call setup() first."
                )
            self._adapter = DataAdapter(self.artifacts)

        return self._adapter

    @property
    def query_config(self) -> QueryConfig:
        """History and negative policy used by every query view of this run."""
        return QueryConfig(
            max_history=self.max_history,
            negatives_per_positive=settings.TRAIN_NEGATIVES_PER_POSITIVE,
            random_state=self.random_state,
            use_histories=self.has_temporal_order,
        )

    def prepare_data(self) -> None:
        if self.use_processed_data and self.processed_folder.exists():
            return

        download_raw_data(self.dataset_name)

    def setup(self, stage: str | None = None):
        self._ensure_artifacts()

        config = self.query_config
        if stage in ("fit", None):
            self.train_ds = self.adapter.get_queries("train", config=config)
            self.val_ds = self.adapter.get_queries("val", config=config)
        if stage in ("test", None):
            self.test_ds = self.adapter.get_queries("test", config=config)

    def export_recbole(self, *, mode: str = "general") -> dict[str, Path]:
        """Write RecBole atomic files for the current processed data.

        ``mode="sequential"`` also writes the prebuilt ``*_seq`` benchmark
        splits. The folder is replaced atomically, so a failure leaves the
        previous export untouched.
        """
        return replace_folder(
            self.atomic_folder,
            lambda staging: self.adapter.export_recbole(
                staging,
                mode=mode,
                dataset_name=self.data_variant,
                config=self.query_config,
            ),
        )

    def validate_topks(self, topks: list[int]) -> None:
        """Reject impossible ranking cutoffs before any model is trained."""
        if not topks or any(k < 1 or k > self.num_items for k in topks):
            raise ValueError(
                f"{self.data_variant}: top-k values must be in [1, {self.num_items}], "
                f"got {topks}."
            )

    def _ensure_artifacts(self) -> None:
        if self.is_processed:
            return

        if self.use_processed_data and self.processed_folder.exists():
            self._load_processed_data()
        else:
            self._process_raw_data()

    def _load_processed_data(self) -> None:
        try:
            artifacts = ProcessedData.load(self.processed_folder)
        except Exception as exc:
            raise RuntimeError(
                f"{self.data_variant}: cannot load processed data from "
                f"{self.processed_folder}. It was written by an incompatible "
                "version; delete the folder and rebuild without "
                "--use-processed."
            ) from exc

        if not artifacts.is_ready:
            raise RuntimeError(
                f"{self.data_variant}: processed data in {self.processed_folder} "
                "is incomplete; delete the folder and rebuild without "
                "--use-processed."
            )

        self._validate_manifest(artifacts)
        self._set_artifacts(artifacts)

    def _validate_manifest(self, artifacts: ProcessedData) -> None:
        """Reject a cache built with other data options, loudly but actionably."""
        if artifacts.manifest is None:
            warnings.warn(
                f"{self.data_variant}: processed cache has no manifest; "
                "compatibility with the current data options cannot be verified.",
                stacklevel=2,
            )
            return

        if artifacts.train is None:
            raise RuntimeError("Processed training interactions are not available.")
        expected = expected_manifest(
            self.data_variant,
            self._build_config(),
            strategy=split_strategy_for(artifacts.train),
        )
        issue = artifacts.manifest.incompatibility(expected)
        if issue is not None:
            raise RuntimeError(
                f"{self.data_variant}: cannot reuse processed data in "
                f"{self.processed_folder}: {issue}. Delete the folder and rebuild "
                "without --use-processed, or pass the original data options."
            )

    def _set_artifacts(self, artifacts: ProcessedData) -> None:
        self.artifacts = artifacts
        self._adapter = None

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
        artifacts = build_processed(
            load_raw_data(self.dataset_name),
            self._build_config(),
            dataset_name=self.data_variant,
        )
        self._set_artifacts(artifacts)
        artifacts.save(self.processed_folder)

    @property
    def kg_node_counts(self) -> dict[str, int]:
        graph = self.adapter.get_knowledge_graph()

        return {
            node: graph[node].num_nodes
            for node in graph.node_types
            if node not in {"user", "item"}
        }

    @property
    def kg_edge_types(self) -> list[tuple[str, str, str]]:
        return list(self.adapter.get_knowledge_graph().edge_types)

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
    def num_interaction_dense_feats(self) -> int:
        return len(self.adapter.get_interaction_features().dense_cols)

    @property
    def interaction_cat_cardinalities(self) -> list[int]:
        return list(self.adapter.get_interaction_features().cat_cardinalities)

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
