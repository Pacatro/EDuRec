"""The single facade every consumer uses to read processed data.

:class:`DataAdapter` exposes lazy, cached views over one canonical
:class:`~edurec.datasets.cache.ProcessedData`. It transforms existing data; it
never changes interactions, labels or splits, and every collaborative view is
built from the TRAIN split only.

Memoised views are shared read-only objects: do not mutate returned frames,
tensors or graphs in place. Views are keyed by their content (view name, split
and the :class:`~edurec.datasets.adapters.queries.QueryConfig` fields), so a
different history length or negative policy never reuses another view.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from edurec import settings
from edurec.datasets.adapters import graphs, queries, recbole
from edurec.datasets.adapters.queries import InteractionFeatureTables, QueryConfig
from edurec.datasets.cache import ProcessedData
from edurec.datasets.features.tabular import DataProcessor
from edurec.datasets.torch_data import RecSysDataset

SPLIT_NAMES = ("train", "val", "test")


class DataAdapter:
    """Read-only views over ``data``.

    Construction is cheap: graphs, histories and atomic files are built on
    first request and then memoised. ``data`` must already be prepared; use
    :class:`~edurec.datasets.ElearningDataModule` to build or load it.
    """

    def __init__(self, data: ProcessedData):
        if not data.is_ready:
            raise ValueError("ProcessedData is not ready; build or load it first.")

        self.data = data
        self._cache: dict[tuple[Any, ...], Any] = {}
        self._relevant_splits: dict[str, pd.DataFrame] | None = None

    # ------------------------------------------------------------------
    # Derived metadata
    # ------------------------------------------------------------------
    @property
    def dataset_name(self) -> str:
        """Name recorded in the cache manifest, or an empty string if absent."""
        manifest = self.data.manifest

        return "" if manifest is None else manifest.dataset

    @property
    def num_users(self) -> int:
        if self.data.u_static_feats is None:
            raise RuntimeError("User features are not available.")

        return self.data.u_static_feats.shape[0]

    @property
    def num_items(self) -> int:
        if self.data.i_static_feats is None:
            raise RuntimeError("Item features are not available.")

        return self.data.i_static_feats.shape[0]

    @property
    def is_explicit(self) -> bool:
        if self.data.train is None:
            raise RuntimeError("Processed training interactions are not available.")

        return settings.RATING_COL in self.data.train.columns

    @property
    def has_temporal_order(self) -> bool:
        if self.data.train is None:
            raise RuntimeError("Processed training interactions are not available.")

        return settings.TIME_COL in self.data.train.columns

    # ------------------------------------------------------------------
    # Interaction views
    # ------------------------------------------------------------------
    def get_interaction_features(self) -> InteractionFeatureTables:
        """Aligned interaction context tables for the positive splits."""
        return self._cached(
            ("interaction_features",),
            lambda: queries.build_interaction_features(
                self.relevant_splits(), self._processor
            ),
        )

    # ------------------------------------------------------------------
    # Graph views
    # ------------------------------------------------------------------
    def get_knowledge_graph(self) -> HeteroData:
        """Static item knowledge graph (``HeteroData``), never user-item edges."""
        return self._cached(("knowledge_graph",), self._build_knowledge_graph)

    def get_bipartite_graph(self) -> torch.Tensor:
        """``(user, item)`` ``edge_index`` of shape ``[2, edges]`` from TRAIN only.

        Repeated events are preserved so models that weigh repetition keep the
        exact processed input.
        """
        return self._cached(
            ("bipartite_graph",),
            lambda: graphs.build_bipartite_graph(self.relevant_splits()["train"]),
        )

    # ------------------------------------------------------------------
    # Query views
    # ------------------------------------------------------------------
    def get_queries(
        self,
        split: str = "train",
        *,
        config: QueryConfig | None = None,
    ) -> RecSysDataset:
        """``RecSysDataset`` with strictly-past history and optional train negatives."""
        _check_split(split)
        query_config = config or QueryConfig()

        return self._cached(
            ("queries", split, query_config),
            lambda: self._build_queries(split, query_config),
        )

    # ------------------------------------------------------------------
    # RecBole export
    # ------------------------------------------------------------------
    def export_recbole(
        self,
        output_dir: Path,
        *,
        mode: str = "general",
        dataset_name: str | None = None,
        config: QueryConfig | None = None,
    ) -> dict[str, Path]:
        """Write RecBole atomic files for the processed splits.

        ``mode="general"`` writes the interaction, user and item files.
        ``mode="sequential"`` also writes the prebuilt ``*_seq`` benchmark
        splits, which require chronological histories. The exporter never
        rebuilds splits, positives or negatives.
        """
        if mode not in {"general", "sequential"}:
            raise ValueError("mode must be 'general' or 'sequential'.")
        if mode == "sequential" and not self.has_temporal_order:
            raise ValueError("Sequential atomic files require chronological histories.")
        if dataset_name is None:
            dataset_name = self.dataset_name
        if not dataset_name:
            raise ValueError(
                "dataset_name is required when the processed data has no manifest."
            )

        atomic_files = recbole.export_atomic_files(self.data, dataset_name, output_dir)
        if mode == "general":
            return atomic_files

        query_config = config or QueryConfig()
        views = {
            export_name: self.get_queries(split, config=query_config)
            for export_name, split in recbole.SEQUENTIAL_SPLIT_SOURCES.items()
        }
        atomic_files.update(
            recbole.export_sequential_benchmark(views, dataset_name, output_dir)
        )

        return atomic_files

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @property
    def _processor(self) -> DataProcessor:
        if self.data.data_processor is None:
            raise RuntimeError("Data processor is not available.")

        return self.data.data_processor

    def relevant_splits(self) -> dict[str, pd.DataFrame]:
        """Positive interactions per split, validated and cached."""
        if self._relevant_splits is None:
            self._relevant_splits = _positive_splits(self.data, self.dataset_name)

        return self._relevant_splits

    def _build_knowledge_graph(self) -> HeteroData:
        if self.data.item_features is None or self.data.i_static_feats is None:
            raise RuntimeError("Knowledge graph requires item data.")

        return graphs.build_knowledge_graph(
            self.data.item_features,
            self.data.i_static_feats,
            self._processor,
        )

    def _build_queries(self, split: str, config: QueryConfig) -> RecSysDataset:
        positives = self.relevant_splits()
        tables = self.get_interaction_features()
        history_items, history_mask, history_context = self._histories(config)[split]
        negatives = self._train_negatives(config) if split == "train" else None

        return RecSysDataset(
            interactions=positives[split],
            history_items=history_items,
            history_valid_mask=history_mask,
            history_context_index=history_context,
            interaction_dense=tables.dense,
            interaction_cat=tables.cat,
            interaction_timestamps=tables.timestamps,
            negative_item_ids=negatives,
        )

    def _histories(
        self, config: QueryConfig
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        enabled = (
            self.has_temporal_order
            if config.use_histories is None
            else config.use_histories
        )

        return self._cached(
            ("histories", config),
            lambda: queries.build_histories(
                self.relevant_splits(),
                max_history=config.max_history,
                enabled=enabled,
            ),
        )

    def _train_negatives(self, config: QueryConfig) -> np.ndarray | None:
        """Train negatives, or ``None`` for explicit feedback."""
        if self.is_explicit or config.negatives_per_positive == 0:
            return None

        return self._cached(
            ("train_negatives", config),
            lambda: queries.generate_negative_samples(
                interactions=self.relevant_splits()["train"],
                item_ids=np.arange(self.num_items),
                num_negatives=config.negatives_per_positive,
                random_state=config.random_state,
                observed_interactions=pd.concat(
                    self.relevant_splits().values(), ignore_index=True
                ),
            ),
        )

    def _cached[T](self, key: tuple[Any, ...], build: Callable[[], T]) -> T:
        if key not in self._cache:
            self._cache[key] = build()

        return cast(T, self._cache[key])


def _check_split(split: str) -> None:
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}.")


def _positive_splits(data: ProcessedData, label: str) -> dict[str, pd.DataFrame]:
    """Filter every split to positive rows and validate the canonical IDs."""
    name = label or "processed data"
    num_users = 0 if data.u_static_feats is None else data.u_static_feats.shape[0]
    num_items = 0 if data.i_static_feats is None else data.i_static_feats.shape[0]

    relevant: dict[str, pd.DataFrame] = {}
    for split, frame in data.splits().items():
        positive = frame.loc[frame[settings.RELEVANT_COL] > 0].reset_index(drop=True)
        if positive.empty:
            raise ValueError(
                f"{name}: {split} split has no positive interactions. "
                "Check filtering, relevance thresholds and split ratios."
            )
        for column, size in (
            (settings.USER_COL, num_users),
            (settings.ITEM_COL, num_items),
        ):
            if not positive[column].between(0, size - 1).all():
                raise ValueError(f"{name}: {split} contains invalid {column} values.")
        relevant[split] = positive

    return relevant
