"""Adapter contract and factory signature for model-specific data preparation.

Concrete adapters live beside the model or baseline they serve.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from edurec.datasets import queries
from edurec.datasets.cache import ProcessedData
from edurec.datasets.preprocessing.splits import SPLIT_ORDER
from edurec.datasets.queries import (
    InteractionFeatureTables,
    QueryConfig,
)
from edurec.datasets.torch_data import RecSysDataset, RecSysQuery


@dataclass(frozen=True)
class AdapterMetadata:
    num_users: int
    num_items: int
    has_temporal_order: bool


class DataAdapter[Sample, ModelInput](ABC):
    """Model-specific structures over canonical data and a fixed evaluation protocol.

    Subclasses may return any map-style dataset and build arbitrary structures.
    Collaborative structures must use train only; query inputs must never use
    the target or future events. Cached structures must not be mutated in place.
    Training, checkpoints and score computation belong to the model integration.
    """

    requires_temporal_order = False
    collate_fn: Callable[[list[Sample]], Any] | None = None

    def __init__(self, data: ProcessedData, config: QueryConfig | None = None):
        self.data = data
        self.config = config or QueryConfig()
        if self.requires_temporal_order and not data.has_temporal_order:
            raise ValueError(f"{type(self).__name__} requires chronological histories.")
        self._queries: dict[tuple[str, QueryConfig], RecSysDataset] = {}
        self._histories_cache: dict[
            QueryConfig, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = {}
        self._negative_cache: dict[QueryConfig, np.ndarray | None] = {}

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            self.data.num_users, self.data.num_items, self.data.has_temporal_order
        )

    def prepare(self, stage: str | None = None) -> None:
        """Optionally build model structures for fit, validate, test or predict.

        Implementations must support repeated calls without changing splits.
        Structures can also be built lazily in ``dataset`` or ``evaluation_input``.
        """

    @abstractmethod
    def dataset(self, split: str) -> Dataset[Sample]:
        """Return model-specific samples for an existing canonical split."""

    def evaluation_queries(self, split: str = "test") -> RecSysDataset:
        """Common targets and histories; independent of training batch format."""
        return self.get_queries(split)

    @abstractmethod
    def evaluation_input(self, query: RecSysQuery) -> ModelInput:
        """Convert a common query batch to inputs, preserving its row order.

        The integration must return scores of shape [batch, num_items] in
        canonical item-ID order before passing them to the shared metrics.
        """

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

    def get_queries(
        self,
        split: str = "train",
        config: QueryConfig | None = None,
    ) -> RecSysDataset:
        """``RecSysDataset`` with strictly-past history and optional train negatives."""
        if split not in SPLIT_ORDER:
            raise ValueError(f"split must be one of {SPLIT_ORDER}, got {split!r}.")

        query_config = config or self.config
        key = (split, query_config)
        if key not in self._queries:
            self._queries[key] = self._build_queries(split, query_config)

        return self._queries[key]

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


#: Build a model adapter once canonical data is available.
AdapterFactory = Callable[[ProcessedData, QueryConfig], DataAdapter[Any, Any]]
