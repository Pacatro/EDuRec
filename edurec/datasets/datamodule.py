import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from .. import settings
from .data_processor import DataProcessor
from .user_history import UserHistory
from .loaders import DatasetName, RawDataset, Schema, load_raw_data
from .ranker_dataset import RankerDataset
from .retrieval_dataset import RetrievalDataset


class Phase(StrEnum):
    RETRIEVAL = "retrieval"
    RANKING = "ranking"


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
        return (
            self.train is not None
            and self.val is not None
            and self.test is not None
            and self.u_static_feats is not None
            and self.i_static_feats is not None
            and self.data_processor is not None
        )


class ElearningDataModule(L.LightningDataModule):
    """
    Implements the end-to-end data pipeline for the recommendation system.

    This module handles raw data ingestion, per-user temporal or random splitting,
    feature preprocessing via a specialized `DataProcessor`, and persistence of
    processed artifacts to disk to skip redundant computations in future runs.

    Parameters:
        dataset (DatasetName): The name of the dataset to use.
        batch_size (int): The batch size for training and validation.
        test_ratio (float): The ratio of test interactions to the total dataset.
        val_ratio (float): The ratio of validation interactions to the total dataset.
        min_interactions (int): The minimum number of interactions per user.
        use_processed_data (bool): Whether to use pre-processed data from disk.
        random_state (int | None): The random seed to use for data splitting.
    """

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

        self.processed_folder = (
            Path(settings.PROCESSED_FOLDER) / self.dataset_name.value
        )

        self.raw_dataset: RawDataset | None = None
        self.artifacts = ProcessedArtifacts()
        self.next_item_hist_by_split: dict[str, UserHistory] = {}
        self.phase: Phase | None = None
        self._loaded_processed_cache = False

        self.excluded_cols = [
            settings.USER_COL,
            settings.ITEM_COL,
            settings.RELEVANT_COL,
            settings.RATING_COL,
            settings.TIME_COL,
            settings.INTERACTION_ORDER_COL,
            settings.CANDIDATE_IDS_COL,
            settings.CANDIDATE_LABELS_COL,
            settings.POSITIVE_POSITION_COL,
            settings.TARGET_IN_CANDIDATES_COL,
            settings.TARGET_FORCED_COL,
        ]

        self._load_data()

    @property
    def interactions(self) -> pd.DataFrame:
        if self.raw_dataset is None:
            raise RuntimeError("Raw interactions are not available.")
        return self.raw_dataset.interactions

    @property
    def users_feats(self) -> pd.DataFrame:
        if self.raw_dataset is None:
            raise RuntimeError("Raw user features are not available.")
        return self.raw_dataset.u_feats

    @property
    def items_feats(self) -> pd.DataFrame:
        if self.raw_dataset is None:
            raise RuntimeError("Raw item features are not available.")
        return self.raw_dataset.i_feats

    @property
    def schema(self) -> Schema:
        if self.raw_dataset is not None:
            return self.raw_dataset.schema
        if self.data_processor is not None:
            return self.data_processor.schema
        raise RuntimeError("Schema is not available.")

    @property
    def u_static_feats(self) -> torch.Tensor | None:
        return self.artifacts.u_static_feats

    @property
    def i_static_feats(self) -> torch.Tensor | None:
        return self.artifacts.i_static_feats

    @property
    def data_processor(self) -> DataProcessor | None:
        return self.artifacts.data_processor

    @property
    def _processed_data(self) -> dict[str, pd.DataFrame | None]:
        return {
            "train": self.artifacts.train,
            "val": self.artifacts.val,
            "test": self.artifacts.test,
        }

    @property
    def is_processed(self) -> bool:
        """Whether processed splits and static features are available."""
        return self.artifacts.is_ready

    @property
    def num_users(self) -> int:
        """Return total number of users from processed or raw features."""
        if self.u_static_feats is not None:
            return self.u_static_feats.shape[0]

        return len(self.raw_dataset.u_feats) if self.raw_dataset is not None else 0

    @property
    def num_items(self) -> int:
        """Return total number of items from processed or raw features."""
        if self.i_static_feats is not None:
            return self.i_static_feats.shape[0]

        return len(self.raw_dataset.i_feats) if self.raw_dataset is not None else 0

    @property
    def num_interactions(self) -> int:
        """Return interaction count across all splits or raw interactions."""
        if self.is_processed:
            return sum(
                len(df) for df in self._processed_data.values() if df is not None
            )

        return len(self.raw_dataset.interactions) if self.raw_dataset is not None else 0

    @property
    def num_ctx_feats(self) -> int:
        """Return number of context features."""
        if self.is_processed:
            df = self._processed_data["train"]
            if df is not None:
                return len([c for c in df.columns if c not in self.excluded_cols])

        return (
            len([c for c in self.raw_dataset.interactions.columns])
            if self.raw_dataset is not None
            else 0
        )

    @property
    def sparsity(self) -> float:
        """Compute dataset sparsity as 1 - interactions/(users*items)."""
        n_inter = self.num_interactions
        n_users = self.num_users
        n_items = self.num_items
        if n_users == 0 or n_items == 0:
            return 0.0
        return 1 - (n_inter / (n_users * n_items))

    @property
    def num_user_feats(self) -> int:
        if not self.is_processed and self.raw_dataset is not None:
            return len(self.raw_dataset.u_feats.columns)
        if self.data_processor is not None:
            metadata = self.data_processor.feature_metadata["users"]
            return len(metadata.dense_cols) + len(metadata.categorical_cols)
        if self.u_static_feats is not None:
            return self.u_static_feats.shape[1]
        return 0

    @property
    def num_item_feats(self) -> int:
        if not self.is_processed and self.raw_dataset is not None:
            return len(self.raw_dataset.i_feats.columns)
        if self.data_processor is not None:
            metadata = self.data_processor.feature_metadata["items"]
            return len(metadata.dense_cols) + len(metadata.categorical_cols)
        if self.i_static_feats is not None:
            return self.i_static_feats.shape[1]
        return 0

    @property
    def num_user_dense_feats(self) -> int:
        if self.data_processor is None:
            return 0
        return len(self.data_processor.feature_metadata["users"].dense_cols)

    @property
    def num_item_dense_feats(self) -> int:
        if self.data_processor is None:
            return 0
        return len(self.data_processor.feature_metadata["items"].dense_cols)

    @property
    def user_cat_cardinalities(self) -> list[int]:
        if self.data_processor is None:
            return []
        metadata = self.data_processor.feature_metadata["users"]
        return [
            metadata.categorical_cardinalities[col] for col in metadata.categorical_cols
        ]

    @property
    def item_cat_cardinalities(self) -> list[int]:
        if self.data_processor is None:
            return []
        metadata = self.data_processor.feature_metadata["items"]
        return [
            metadata.categorical_cardinalities[col] for col in metadata.categorical_cols
        ]

    def _load_data(self):
        """Load raw inputs or processed cache depending on configuration."""
        required_files = [
            self._processed_split_filename("train"),
            self._processed_split_filename("val"),
            self._processed_split_filename("test"),
            "static_feats.safetensors",
            "processor.joblib",
            "preprocess_metadata.json",
        ]

        cache_exists = self.processed_folder.exists() and all(
            (self.processed_folder / f).exists() for f in required_files
        )

        if self.use_processed_data and cache_exists and self._has_compatible_cache():
            self._load_processed_data()
            return

        self._load_raw_dataset()

    def _load_raw_dataset(self):
        raw_dataset = load_raw_data(self.dataset_name)
        self.raw_dataset = self._prepare_raw_dataset(raw_dataset)
        self.artifacts = ProcessedArtifacts(
            data_processor=DataProcessor(schema=self.raw_dataset.schema)
        )
        self._loaded_processed_cache = False

    def _has_compatible_cache(self) -> bool:
        metadata_path = self.processed_folder / "preprocess_metadata.json"
        if not metadata_path.exists():
            return False

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata == self._build_cache_metadata()

    def _build_cache_metadata(self) -> dict[str, int | str | list[str]]:
        return {
            "preprocess_cache_version": settings.PREPROCESS_CACHE_VERSION,
            "feature_types": list(settings.PREPROCESS_FEATURE_TYPES),
            "text_preprocess_strategy": settings.TEXT_PREPROCESS_STRATEGY,
            "text_embedding_model": settings.TEXT_EMBEDDING_MODEL,
            "text_embedding_dim": settings.TEXT_EMBEDDING_DIM,
        }

    def _prepare_raw_dataset(self, raw_dataset: RawDataset) -> RawDataset:
        interactions = self._clean_cols_names(raw_dataset.interactions)
        items = self._clean_cols_names(raw_dataset.i_feats)
        users = self._clean_cols_names(raw_dataset.u_feats)

        if self.remove_sparse:
            interactions, users, items = self._filter_sparse_iterative(
                interactions, users, items
            )

        interactions = self._add_relevant_col(interactions)
        interactions = self._add_interaction_order(interactions)

        return RawDataset(
            interactions=interactions,
            i_feats=items,
            u_feats=users,
            schema=raw_dataset.schema,
        )

    def _clean_cols_names(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        cleaned.columns = (
            cleaned.columns.str.lower()
            .str.strip()
            .str.replace(" ", "_")
            .str.replace(r"[^\w]", "", regex=True)
        )
        return cleaned

    def _filter_sparse_iterative(
        self,
        interactions: pd.DataFrame,
        users: pd.DataFrame,
        items: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        while True:
            prev_len = len(interactions)

            interactions, users = self._filter_sparse(
                interactions=interactions,
                features=users,
                col=settings.USER_COL,
                min_interactions=self.min_interactions,
            )

            interactions, items = self._filter_sparse(
                interactions=interactions,
                features=items,
                col=settings.ITEM_COL,
                min_interactions=self.min_interactions,
            )

            if len(interactions) == prev_len:
                break

        return interactions, users, items

    def _filter_sparse(
        self,
        interactions: pd.DataFrame,
        features: pd.DataFrame,
        col: str,
        min_interactions: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        counts = interactions[col].value_counts(sort=False)
        valid_ids = counts[counts >= min_interactions].index

        filtered_interactions = interactions[
            interactions[col].isin(valid_ids)
        ].reset_index(drop=True)
        filtered_features = features[features[col].isin(valid_ids)].reset_index(
            drop=True
        )

        return filtered_interactions, filtered_features

    def _add_relevant_col(self, interactions: pd.DataFrame) -> pd.DataFrame:
        if (
            settings.RELEVANT_COL in interactions.columns
            or settings.RATING_COL not in interactions.columns
        ):
            return interactions

        interactions = interactions.copy()
        global_threshold = interactions[settings.RATING_COL].mean()
        user_stats = interactions.groupby(settings.USER_COL)[settings.RATING_COL]
        mean_user_ratings = user_stats.transform("mean")
        count_user_ratings = user_stats.transform("count")

        thresholds = np.where(
            count_user_ratings < self.min_interactions,
            global_threshold,
            mean_user_ratings,
        )
        interactions[settings.RELEVANT_COL] = (
            interactions[settings.RATING_COL] >= thresholds
        )

        return interactions

    def _add_interaction_order(self, interactions: pd.DataFrame) -> pd.DataFrame:
        interactions = interactions.copy()
        interactions[settings.INTERACTION_ORDER_COL] = np.arange(
            len(interactions), dtype=np.int64
        )
        return interactions

    def _load_processed_data(self):
        """Load cached splits, static features, and fitted preprocessor."""
        assert self.processed_folder is not None and self.processed_folder.exists()

        processed_splits = {
            split: pd.read_feather(
                self.processed_folder / self._processed_split_filename(split)
            )
            for split in ["train", "val", "test"]
        }

        static_feats = load_file(self.processed_folder / "static_feats.safetensors")
        self.artifacts = ProcessedArtifacts(
            train=processed_splits["train"],
            val=processed_splits["val"],
            test=processed_splits["test"],
            u_static_feats=static_feats["u_static_feats"],
            i_static_feats=static_feats["i_static_feats"],
            data_processor=DataProcessor.load(
                self.processed_folder / "processor.joblib"
            ),
        )
        self._loaded_processed_cache = True

    def setup(self, stage: str | None = None, phase: Phase | None = None):
        """Prepare processed datasets for training/validation/testing stages."""
        if phase is not None:
            self.phase = phase
        elif self.phase is None:
            raise RuntimeError("setup() requires a phase before building datasets.")

        if not self.is_processed:
            train_raw, val_raw, test_raw = self._split_data()
            self.artifacts = self._prepare_processed_artifacts(
                train_raw=train_raw,
                val_raw=val_raw,
                test_raw=test_raw,
            )
            if not self._loaded_processed_cache:
                self._save_processed_data()

        self._build_runtime_state()

        match stage:
            case "fit" | None:
                self.train_ds = self._make_dataset("train")
                self.val_ds = self._make_dataset("val")
            case "test":
                self.test_ds = self._make_dataset("test")

    def _split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits global interactions into Train/Val/Test sets.
        Applies temporal splitting (last-n) if timestamps are available,
        otherwise performs a per-user random shuffle.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train/Val/Test splits.
        """
        assert not self.is_processed
        df = self.interactions
        rng = np.random.default_rng(self.random_state)

        if settings.TIME_COL in df.columns:
            df = df.sort_values(
                by=[settings.TIME_COL, settings.INTERACTION_ORDER_COL],
                kind="mergesort",
            )

        train_parts, val_parts, test_parts = [], [], []

        for _, g in df.groupby(settings.USER_COL, sort=False):
            n = len(g)

            if n < self.min_interactions:
                train_parts.append(g)
                continue

            n_test = np.maximum(1, int(np.floor(n * self.test_ratio)))
            n_val = np.maximum(1, int(np.floor(n * self.val_ratio)))

            if n_test + n_val >= n:
                n_test, n_val = 1, 1

            if settings.TIME_COL in g.columns:
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

    def _prepare_processed_artifacts(
        self,
        train_raw: pd.DataFrame,
        val_raw: pd.DataFrame,
        test_raw: pd.DataFrame,
    ) -> ProcessedArtifacts:
        """
        Fits the `DataProcessor` on training data and transforms all splits.
        Also generates static feature matrices and persists results to the cache.
        """
        data_processor = self._require_data_processor()
        data_processor.fit(
            users_train=self.users_feats,
            items_train=self.items_feats,
            interactions_train=train_raw,
        )

        processed_all = data_processor.transform(
            users=self.users_feats, items=self.items_feats
        )
        assert processed_all.users is not None and processed_all.items is not None

        u_static_feats = self._generate_static_feats(
            processed_all.users, settings.USER_COL
        )
        i_static_feats = self._generate_static_feats(
            processed_all.items, settings.ITEM_COL
        )

        p_train = data_processor.transform(interactions=train_raw)
        p_val = data_processor.transform(interactions=val_raw)
        p_test = data_processor.transform(interactions=test_raw)

        return ProcessedArtifacts(
            train=p_train.interactions,
            val=p_val.interactions,
            test=p_test.interactions,
            u_static_feats=u_static_feats,
            i_static_feats=i_static_feats,
            data_processor=data_processor,
        )

    def _build_runtime_state(self):
        self.next_item_hist_by_split = {}
        user_history_state = {}

        for split in ("train", "val", "test"):
            split_df = getattr(self.artifacts, split)
            self.next_item_hist_by_split[split], user_history_state = (
                self._precompute_history_for_split(split_df, user_history_state)
            )

    def _precompute_history_for_split(
        self,
        df: pd.DataFrame | None,
        initial_state: dict[int, list[tuple[int, list[float]]]],
    ) -> tuple[UserHistory, dict[int, list[tuple[int, list[float]]]]]:
        if df is None:
            num_rows = 0
            history_shape = (num_rows, settings.MAX_HISTORY_LEN)
            empty_history = UserHistory(
                items=torch.zeros(history_shape, dtype=torch.long),
                ctx=torch.zeros(
                    (num_rows, settings.MAX_HISTORY_LEN, self.num_ctx_feats),
                    dtype=torch.float32,
                ),
                valid_mask=torch.zeros(history_shape, dtype=torch.bool),
            )

            return empty_history, self._clone_user_history_state(initial_state)

        ctx_cols = [col for col in df.columns if col not in self.excluded_cols]
        num_rows = len(df)

        history_shape = (num_rows, settings.MAX_HISTORY_LEN)
        history = UserHistory(
            items=torch.zeros(history_shape, dtype=torch.long),
            ctx=torch.zeros(
                (num_rows, settings.MAX_HISTORY_LEN, self.num_ctx_feats),
                dtype=torch.float32,
            ),
            valid_mask=torch.zeros(history_shape, dtype=torch.bool),
        )

        user_history_state = self._clone_user_history_state(initial_state)

        if num_rows == 0:
            return history, user_history_state

        working_df = df.reset_index(drop=True).copy()
        working_df["row_pos"] = np.arange(num_rows, dtype=np.int64)
        ordered_df = working_df.sort_values(
            by=self._get_history_sort_columns(working_df),
            kind="mergesort",
        )

        for row in ordered_df.itertuples(index=False):
            row_pos = int(row.row_pos)  # type: ignore
            user_id = int(getattr(row, settings.USER_COL))
            past_interactions = user_history_state.get(user_id, [])
            self._write_history_row(
                history=history,
                row_pos=row_pos,
                past_interactions=past_interactions,
            )

            item_id = int(getattr(row, settings.ITEM_COL))
            ctx_values = [float(getattr(row, col)) for col in ctx_cols]
            user_history_state.setdefault(user_id, []).append((item_id, ctx_values))

        return history, user_history_state

    def _clone_user_history_state(
        self,
        history_state: dict[int, list[tuple[int, list[float]]]],
    ) -> dict[int, list[tuple[int, list[float]]]]:
        return {
            int(user_id): [(item_id, list(ctx_vals)) for item_id, ctx_vals in entries]
            for user_id, entries in history_state.items()
        }

    def _get_history_sort_columns(self, df: pd.DataFrame) -> list[str]:
        sort_cols = [settings.USER_COL]
        if settings.TIME_COL in df.columns:
            sort_cols.append(settings.TIME_COL)
        sort_cols.append(settings.INTERACTION_ORDER_COL)
        return sort_cols

    def _write_history_row(
        self,
        history: UserHistory,
        row_pos: int,
        past_interactions: list[tuple[int, list[float]]],
    ):
        truncated_history = past_interactions[-settings.MAX_HISTORY_LEN :]
        hist_len = len(truncated_history)

        if hist_len == 0:
            return

        history.items[row_pos, :hist_len] = torch.tensor(
            [item_id + 1 for item_id, _ in truncated_history],
            dtype=torch.long,
        )
        history.valid_mask[row_pos, :hist_len] = True

        if self.num_ctx_feats > 0:
            history.ctx[row_pos, :hist_len] = torch.tensor(
                [ctx_vals for _, ctx_vals in truncated_history],
                dtype=torch.float32,
            )

    def _generate_static_feats(self, df: pd.DataFrame, id_col: str) -> torch.Tensor:
        """
        Convert sorted entity features into a 2D tensor matrix with
        shape (N, F), where N is the number of entities and F is the number
        of features.
        """
        df_sorted = df.sort_values(id_col)
        data_processor = self._require_data_processor()
        prefix = "users" if id_col == settings.USER_COL else "items"
        metadata = data_processor.feature_metadata[prefix]
        feat_cols = metadata.dense_cols + metadata.categorical_cols
        return torch.tensor(df_sorted[feat_cols].values, dtype=torch.float32)

    def _save_processed_data(self):
        """Persist processed splits, static tensors, and preprocessing artifacts."""
        self.processed_folder.mkdir(parents=True, exist_ok=True)

        for split, df in self._processed_data.items():
            if df is None:
                continue
            df.to_feather(self.processed_folder / self._processed_split_filename(split))

        assert self.u_static_feats is not None and self.i_static_feats is not None

        save_file(
            {
                "u_static_feats": self.u_static_feats.contiguous(),
                "i_static_feats": self.i_static_feats.contiguous(),
            },
            self.processed_folder / "static_feats.safetensors",
        )

        self._require_data_processor().save(self.processed_folder / "processor.joblib")
        (self.processed_folder / "preprocess_metadata.json").write_text(
            json.dumps(self._build_cache_metadata(), indent=2),
            encoding="utf-8",
        )

    def _processed_split_filename(self, split: str) -> str:
        return f"{split}.feather"

    def _make_dataset(self, split: str) -> RankerDataset | RetrievalDataset:
        """Create a processed dataset for the selected training phase."""
        df = getattr(self.artifacts, split)

        if df is None:
            raise RuntimeError(
                f"Data must be processed before creating the dataset for {split}"
            )

        match self.phase:
            case Phase.RETRIEVAL:
                return RetrievalDataset(
                    interactions=df,
                    precomputed_history=self.next_item_hist_by_split[split],
                    num_ctx_feats=self.num_ctx_feats,
                )
            case Phase.RANKING:
                ranking_df, ranking_history = self._build_ranking_split(df, split)
                return RankerDataset(
                    interactions=ranking_df,
                    precomputed_history=ranking_history,
                    num_ctx_feats=self.num_ctx_feats,
                    split=split,
                )
            case _:
                raise RuntimeError(f"Unsupported phase: {self.phase!r}")

    def _build_ranking_split(
        self, df: pd.DataFrame, split: str
    ) -> tuple[pd.DataFrame, UserHistory]:
        positive_mask = (df[settings.RELEVANT_COL] > 0).to_numpy(copy=True)
        ranking_df = df.loc[positive_mask].reset_index(drop=True)

        split_history = self.next_item_hist_by_split[split]
        history_index = torch.as_tensor(positive_mask, dtype=torch.bool)
        ranking_history = UserHistory(
            items=split_history.items[history_index],
            ctx=split_history.ctx[history_index],
            valid_mask=split_history.valid_mask[history_index],
        )

        return ranking_df, ranking_history

    def _require_data_processor(self) -> DataProcessor:
        if self.data_processor is None:
            raise RuntimeError("Data processor is not available.")
        return self.data_processor

    def create_inter_graph(self) -> Data:
        """
        Constructs a homogeneous bipartite graph from processed user-item training interactions.

        The graph uses a unified node index space [0, ..., num_users + num_items - 1],
        where item indices are offset by the total number of users. This structure
        is optimized for homogeneous GNNs and Graph Contrastive Learning (GCL)
        augmentations like Edge Dropout.

        Normaly, the correct representation of this graph is using HeteroData,
        but because of we are going to use GCL and all the librarys and methods
        for this use homogeneous graphs, we are going to use Data instead.

        Topology:
            - Nodes: Unified set representing both users and items.
            - Edges: Undirected (bidirectional) interactions stored in a single
              COO tensor to facilitate symmetric message passing.

        Edge Index Structure [2, 2 * N]:
            Row 0: [u_1, ..., u_n, (i_1 + offset), ..., (i_n + offset)]
            Row 1: [(i_1 + offset), ..., (i_n + offset), u_1, ..., u_n]
            Mapping: u_k <-> (i_k + offset).

        Returns:
            Data: A PyG Data object containing the unified edge_index,
                  and raw features (u_x, i_x) for later projection.

        Raises:
            RuntimeError: If called before data/features are processed and cached.
        """
        df_train = self.artifacts.train

        if df_train is None or self.u_static_feats is None:
            raise RuntimeError("Data must be processed before creating the graph")

        pos_train = df_train[df_train[settings.RELEVANT_COL] > 0]

        u_idx = torch.tensor(pos_train[settings.USER_COL].values, dtype=torch.long)
        i_idx = (
            torch.tensor(pos_train[settings.ITEM_COL].values, dtype=torch.long)
            + self.num_users
        )

        edge_index_u2i = torch.stack([u_idx, i_idx], dim=0)
        edge_index_i2u = torch.stack([i_idx, u_idx], dim=0)

        full_edge_index = torch.cat(
            [edge_index_u2i, edge_index_i2u], dim=1
        ).contiguous()

        num_nodes = self.num_users + self.num_items
        data = Data(edge_index=full_edge_index, num_nodes=num_nodes)

        data.num_users = self.num_users
        data.num_items = self.num_items
        data.node_type = torch.cat(
            [
                torch.zeros(self.num_users, dtype=torch.long),
                torch.ones(self.num_items, dtype=torch.long),
            ]
        )

        return data

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
