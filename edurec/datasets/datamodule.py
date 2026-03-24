from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from .. import config
from .data_processor import DataProcessor
from .elearnig_dataset import ElearningDataset
from .loaders import DatasetName, load_raw_data


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
        n_neg_train (int): The number of negative interactions to sample for training.
        n_neg_val (int): The number of negative interactions to sample for validation.
        n_neg_test (int): The number of negative interactions to sample for testing.
        use_processed_data (bool): Whether to use pre-processed data from disk.
        random_state (int | None): The random seed to use for data splitting.
    """

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
        use_processed_data: bool = False,
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
        self.use_processed_data = use_processed_data
        self.random_state = random_state

        self.processed_folder = Path(config.PROCESSED_FOLDER) / self.dataset_name.value

        self.u_static: torch.Tensor | None = None
        self.i_static: torch.Tensor | None = None

        self.data_processor: DataProcessor | None = None
        self._processed_data: dict[str, pd.DataFrame | None] = {
            "train": None,
            "val": None,
            "test": None,
        }

        self.global_history = {}

        self._load_data()

    @property
    def is_processed(self) -> bool:
        """Whether processed splits and static features are available."""
        return self.u_static is not None and self._processed_data["train"] is not None

    @property
    def num_users(self) -> int:
        """Return total number of users from processed or raw features."""
        if self.u_static is not None:
            return self.u_static.shape[0]
        return len(self.users_feats) if hasattr(self, "users_feats") else 0

    @property
    def num_items(self) -> int:
        """Return total number of items from processed or raw features."""
        if self.i_static is not None:
            return self.i_static.shape[0]
        return len(self.items_feats) if hasattr(self, "items_feats") else 0

    @property
    def num_interactions(self) -> int:
        """Return interaction count across all splits or raw interactions."""
        if self.is_processed:
            return sum(
                len(df) for df in self._processed_data.values() if df is not None
            )
        return len(self.interactions) if hasattr(self, "interactions") else 0

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
        if self.u_static is not None:
            return self.u_static.shape[1]
        return len(self.users_feats.columns) - 1 if hasattr(self, "users_feats") else 0

    @property
    def num_item_feats(self) -> int:
        if self.i_static is not None:
            return self.i_static.shape[1]
        return len(self.items_feats.columns) - 1 if hasattr(self, "items_feats") else 0

    def _load_data(self):
        """Load raw inputs or processed cache depending on configuration."""
        required_files = [
            "train.csv",
            "val.csv",
            "test.csv",
            "static_feats.safetensors",
            "processor.joblib",
        ]

        cache_exists = self.processed_folder.exists() and all(
            (self.processed_folder / f).exists() for f in required_files
        )

        if self.use_processed_data and cache_exists:
            self._load_processed_data()
            return

        print("[DATA] Loading raw data")
        raw_dataset = load_raw_data(self.dataset_name)
        self.interactions = raw_dataset.interactions
        self.users_feats = raw_dataset.u_feats
        self.items_feats = raw_dataset.i_feats
        self.schema = raw_dataset.schema
        self.data_processor = DataProcessor(schema=self.schema)

    def _load_processed_data(self):
        """Load cached splits, static features, and fitted preprocessor."""
        assert self.processed_folder is not None and self.processed_folder.exists()

        print(
            f"[CACHE] Loading processed data and preprocessor from {self.processed_folder}"
        )

        for split in ["train", "val", "test"]:
            self._processed_data[split] = pd.read_csv(
                self.processed_folder / f"{split}.csv"
            )

        static_feats = load_file(self.processed_folder / "static_feats.safetensors")

        self.u_static = static_feats["u_static"]
        self.i_static = static_feats["i_static"]

        self.data_processor = DataProcessor.load(
            self.processed_folder / "processor.joblib"
        )

    def setup(self, stage: str | None = None) -> None:
        """Prepare processed datasets for training/validation/testing stages."""
        if not self.is_processed:
            train_raw, val_raw, test_raw = self._split_data()
            self._preprocess(train_raw, val_raw, test_raw)

        self.global_history = self._generate_global_history()

        match stage:
            case "fit" | None:
                self.train_ds = self._make_dataset("train", self.n_neg_train)
                self.val_ds = self._make_dataset("val", self.n_neg_val)
            case "test":
                self.test_ds = self._make_dataset("test", self.n_neg_test)

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

    def _preprocess(
        self,
        train_raw: pd.DataFrame,
        val_raw: pd.DataFrame,
        test_raw: pd.DataFrame,
    ):
        """
        Fits the `DataProcessor` on training data and transforms all splits.
        Also generates static feature matrices and persists results to the cache.
        """
        assert self.data_processor is not None
        self.data_processor.fit(
            users_train=self.users_feats,
            items_train=self.items_feats,
            interactions_train=train_raw,
        )

        processed_all = self.data_processor.transform(
            users=self.users_feats, items=self.items_feats
        )
        assert processed_all.users is not None and processed_all.items is not None

        self.u_static = self._generate_static_feats(
            processed_all.users, config.USER_COL
        )
        self.i_static = self._generate_static_feats(
            processed_all.items, config.ITEM_COL
        )

        p_train = self.data_processor.transform(interactions=train_raw)
        p_val = self.data_processor.transform(interactions=val_raw)
        p_test = self.data_processor.transform(interactions=test_raw)

        self._processed_data["train"] = p_train.interactions
        self._processed_data["val"] = p_val.interactions
        self._processed_data["test"] = p_test.interactions

        if not self.use_processed_data:
            self._save_processed_data()

    def _generate_global_history(self) -> dict[int, list]:
        train_p = self._processed_data["train"]

        assert train_p is not None

        excluded_cols = [
            config.USER_COL,
            config.ITEM_COL,
            config.RELEVANT_COL,
            config.RATING_COL,
        ]

        if config.TIME_COL in train_p.columns:
            excluded_cols.append(config.TIME_COL)

        ctx_cols = [c for c in train_p.columns if c not in excluded_cols]

        if config.TIME_COL in train_p.columns:
            train_p = train_p.sort_values(config.TIME_COL)

        pos_inter = train_p[train_p[config.RELEVANT_COL] > 0]

        global_history = {}
        for u_id, group in pos_inter.groupby(config.USER_COL):
            items = group[config.ITEM_COL].tolist()
            ctx_vals = group[ctx_cols].values.tolist()

            global_history[u_id] = list(zip(items, ctx_vals))

        return global_history

    def _generate_static_feats(self, df: pd.DataFrame, id_col: str) -> torch.Tensor:
        """
        Convert sorted entity features into a 2D tensor matrix with
        shape (N, F), where N is the number of entities and F is the number
        of features.
        """
        df_sorted = df.sort_values(id_col)
        feat_cols = [c for c in df_sorted.columns if c != id_col]
        return torch.tensor(df_sorted[feat_cols].values, dtype=torch.float32)

    def _save_processed_data(self):
        """Persist processed splits, static tensors, and preprocessing artifacts."""
        print("[CACHE] Saving processed data")

        self.processed_folder.mkdir(parents=True, exist_ok=True)

        for split, df in self._processed_data.items():
            if df is None:
                continue
            df.to_csv(self.processed_folder / f"{split}.csv", index=False)

        assert self.u_static is not None and self.i_static is not None

        save_file(
            {
                "u_static": self.u_static.contiguous(),
                "i_static": self.i_static.contiguous(),
            },
            self.processed_folder / "static_feats.safetensors",
        )

        assert self.data_processor is not None
        self.data_processor.save(self.processed_folder / "processor.joblib")

    def _make_dataset(self, split: str, n_negatives: int) -> ElearningDataset:
        """Create an `ElearningDataset` for a processed split."""
        df = self._processed_data.get(split)

        if df is None:
            raise RuntimeError(
                f"Data must be processed before creating the dataset for {split}"
            )

        return ElearningDataset(
            interactions=df,
            global_history=self.global_history,
            num_ctx_feats=self._num_inter_feats(df),
            n_negatives=n_negatives,
        )

    def _num_inter_feats(self, df: pd.DataFrame) -> int:
        excluded_cols = [
            config.USER_COL,
            config.ITEM_COL,
            config.RELEVANT_COL,
            config.RATING_COL,
            config.TIME_COL,
        ]

        return len([c for c in df.columns if c not in excluded_cols])

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
        df_train = self._processed_data["train"]

        if df_train is None or self.u_static is None:
            raise RuntimeError("Data must be processed before creating the graph")

        pos_train = df_train[df_train[config.RELEVANT_COL] > 0]

        u_idx = torch.tensor(pos_train[config.USER_COL].values, dtype=torch.long)
        i_idx = (
            torch.tensor(pos_train[config.ITEM_COL].values, dtype=torch.long)
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
            num_workers=config.NUM_WORKERS,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
        )
