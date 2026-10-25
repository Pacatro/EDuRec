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

from functools import cached_property
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from edurec.datasets.adapters import graphs, queries, recbole
from edurec.datasets.adapters.queries import InteractionFeatureTables, QueryConfig
from edurec.datasets.cache import ProcessedData
from edurec.datasets.pipeline.splits import SPLIT_ORDER
from edurec.datasets.torch_data import RecSysDataset


class DataAdapter:
    """Read-only views over ``data``.

    Construction is cheap: graphs, histories and atomic files are built on
    first request and then memoised. ``data`` must already be prepared; use
    :class:`~edurec.datasets.ElearningDataModule` to build or load it.
    """

    def __init__(self, data: ProcessedData):
        self.data = data
        self._queries: dict[tuple[str, QueryConfig], RecSysDataset] = {}
        self._histories_cache: dict[
            QueryConfig, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = {}
        self._negative_cache: dict[QueryConfig, np.ndarray | None] = {}

    @property
    def dataset_name(self) -> str:
        return self.data.dataset_name

    @cached_property
    def relevant_splits(self) -> dict[str, pd.DataFrame]:
        """Positive interactions per split, validated and cached."""
        return self.data.positive_splits()

    @cached_property
    def interaction_features(self) -> InteractionFeatureTables:
        """Aligned interaction context tables for the positive splits."""
        return queries.build_interaction_features(
            self.relevant_splits, self.data.transforms["inter"]
        )

    @cached_property
    def knowledge_graph(self) -> HeteroData:
        """Static item knowledge graph (``HeteroData``), never user-item edges."""
        return graphs.build_knowledge_graph(
            self.data.item_features,
            self.data.i_static_feats,
            self.data.schema["items"],
            self.data.graph_node_features,
        )

    @cached_property
    def bipartite_graph(self) -> torch.Tensor:
        """``(user, item)`` ``edge_index`` of shape ``[2, edges]`` from TRAIN only.

        Repeated events are preserved so models that weigh repetition keep the
        exact processed input.
        """
        return graphs.build_bipartite_graph(self.relevant_splits["train"])

    @cached_property
    def kg_node_counts(self) -> dict[str, int]:
        graph = self.knowledge_graph

        return {
            node: graph[node].num_nodes
            for node in graph.node_types
            if node not in {"user", "item"}
        }

    @cached_property
    def kg_edge_types(self) -> list[tuple[str, str, str]]:
        return list(self.knowledge_graph.edge_types)

    def get_queries(
        self,
        split: str = "train",
        config: QueryConfig | None = None,
    ) -> RecSysDataset:
        """``RecSysDataset`` with strictly-past history and optional train negatives."""
        _check_split(split)
        query_config = config or QueryConfig()
        key = (split, query_config)
        if key not in self._queries:
            self._queries[key] = self._build_queries(split, query_config)

        return self._queries[key]

    def export_recbole(
        self,
        output_dir: Path,
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
        if mode == "sequential" and not self.data.has_temporal_order:
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

    def _build_queries(self, split: str, config: QueryConfig) -> RecSysDataset:
        history_items, history_mask, history_context = self._histories(config)[split]
        negatives = self._train_negatives(config) if split == "train" else None

        return RecSysDataset(
            interactions=self.relevant_splits[split],
            history_items=history_items,
            history_valid_mask=history_mask,
            history_context_index=history_context,
            interaction_dense=self.interaction_features.dense,
            interaction_cat=self.interaction_features.cat,
            interaction_timestamps=self.interaction_features.timestamps,
            negative_item_ids=negatives,
        )

    def _histories(
        self, config: QueryConfig
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if config not in self._histories_cache:
            enabled = (
                self.data.has_temporal_order
                if config.use_histories is None
                else config.use_histories
            )
            self._histories_cache[config] = queries.build_histories(
                self.relevant_splits,
                max_history=config.max_history,
                enabled=enabled,
            )

        return self._histories_cache[config]

    def _train_negatives(self, config: QueryConfig) -> np.ndarray | None:
        """Train negatives, or ``None`` for explicit feedback."""
        if self.data.is_explicit or config.negatives_per_positive == 0:
            return None

        if config not in self._negative_cache:
            self._negative_cache[config] = queries.generate_negative_samples(
                interactions=self.relevant_splits["train"],
                item_ids=np.arange(self.data.num_items),
                num_negatives=config.negatives_per_positive,
                random_state=config.random_state,
                observed_interactions=pd.concat(
                    self.relevant_splits.values(), ignore_index=True
                ),
            )

        return cast(np.ndarray | None, self._negative_cache[config])


def _check_split(split: str) -> None:
    if split not in SPLIT_ORDER:
        raise ValueError(f"split must be one of {SPLIT_ORDER}, got {split!r}.")
