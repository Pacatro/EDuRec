"""Lightning data module: assemble queries and data loaders.

Build/load canonical data and delegate input preparation to a model adapter. The
selected adapter supplies model-specific datasets; evaluation queries retain a
common protocol regardless of the training sample format.
"""

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from edurec import settings
from edurec.datasets.adapter import AdapterFactory, DataAdapter
from edurec.datasets.cache import ProcessedData
from edurec.datasets.config import DataConfig
from edurec.datasets.preprocessing.build import (
    BuildResult,
    build_processed,
    split_strategy_for,
)
from edurec.datasets.queries import QueryConfig
from edurec.datasets.sources.registry import download_raw_data, load_raw_data


class ElearningDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: DataConfig,
        batch_size: int,
        use_processed_data: bool = False,
        max_history: int = settings.MAX_HISTORY_LEN,
        persistent_workers: bool = True,
        adapter_factory: AdapterFactory | None = None,
    ):
        super().__init__()
        # Lightning rejects frozen dataclasses when it serialises hparams for
        # logging, so store a plain mapping instead of the config instance.
        # The factory itself is not serialisable; keep its name as a label.
        adapter_name = (
            "proposed"
            if adapter_factory is None
            else getattr(adapter_factory, "__name__", type(adapter_factory).__name__)
        )
        self.save_hyperparameters(
            {
                "config": config.to_dict(),
                "batch_size": batch_size,
                "use_processed_data": use_processed_data,
                "max_history": max_history,
                "persistent_workers": persistent_workers,
                "adapter_name": adapter_name,
            }
        )
        self.config = config
        self.batch_size = batch_size
        self.use_processed_data = use_processed_data
        self.max_history = max_history
        self.persistent_workers = persistent_workers
        self.data_variant = config.dataset.value
        self.adapter_factory = adapter_factory
        self.adapter_name = adapter_name

        self.processed_folder = Path(settings.PROCESSED_FOLDER) / self.data_variant
        self.atomic_folder = Path(settings.ATOMICFILES_FOLDER) / self.data_variant

        self._data: ProcessedData | None = None
        self._build_result: BuildResult | None = None
        self._adapter: DataAdapter[Any, Any] | None = None

    @property
    def data(self) -> ProcessedData:
        """Canonical processed data; requires ``setup()`` or an explicit load."""
        if self._data is None:
            raise RuntimeError(
                f"{self.data_variant}: data is not prepared; call setup() first."
            )
        return self._data

    @property
    def build(self) -> BuildResult | None:
        """Build result from this run; ``None`` when the data came from a cache."""
        return self._build_result

    @property
    def adapter(self) -> DataAdapter[Any, Any]:
        """Adapter built from the injected factory after data preparation.

        ``None`` selects the proposed adapter lazily, keeping the datasets
        package independent from the model package at import time.
        """
        if self._adapter is None:
            factory = self.adapter_factory
            if factory is None:
                from edurec.recsys.recsys import EdurecDataAdapter

                factory = EdurecDataAdapter
            self._adapter = factory(self.data, self.query_config)
        return self._adapter

    def require_adapter[T: DataAdapter[Any, Any]](
        self, adapter_type: type[T]
    ) -> T:
        """Get model-specific structures, rejecting an incompatible integration."""
        adapter = self.adapter
        if not isinstance(adapter, adapter_type):
            raise TypeError(
                f"This integration requires {adapter_type.__name__}, "
                f"got {type(adapter).__name__}."
            )
        return adapter

    @property
    def query_config(self) -> QueryConfig:
        """History and negative policy used by every query view of this run."""
        return QueryConfig(
            max_history=self.max_history,
            negatives_per_positive=settings.TRAIN_NEGATIVES_PER_POSITIVE,
            random_state=self.config.random_state,
            use_histories=self.data.has_temporal_order,
        )

    def with_adapter(
        self, adapter_factory: AdapterFactory, batch_size: int | None = None
    ) -> "ElearningDataModule":
        """An independent model view sharing the prepared, read-only data."""
        view = ElearningDataModule(
            config=self.config,
            batch_size=self.batch_size if batch_size is None else batch_size,
            use_processed_data=self.use_processed_data,
            max_history=self.max_history,
            persistent_workers=self.persistent_workers,
            adapter_factory=adapter_factory,
        )
        view._data = self.data
        view._build_result = self.build
        return view

    def prepare_data(self) -> None:
        if self._data is not None:
            return
        if self.use_processed_data and self.processed_folder.exists():
            return

        download_raw_data(self.config.dataset)

    def setup(self, stage: str | None = None):
        self._ensure_data()

        adapter = self.adapter
        adapter.prepare(stage)
        if stage in ("fit", None):
            self.train_ds = adapter.dataset("train")
            self.val_ds = adapter.dataset("val")
        if stage == "validate":
            self.val_ds = adapter.dataset("val")
        if stage in ("test", None):
            self.test_ds = adapter.dataset("test")
        if stage == "predict":
            self.predict_ds = adapter.dataset("test")

    def validate_topks(self, topks: list[int]) -> None:
        """Reject impossible ranking cutoffs before any model is trained."""
        if not topks or any(k < 1 or k > self.data.num_items for k in topks):
            raise ValueError(
                f"{self.data_variant}: top-k values must be in [1, {self.data.num_items}], "
                f"got {topks}."
            )

    def _ensure_data(self) -> None:
        if self._data is not None:
            return

        if self.use_processed_data and self.processed_folder.exists():
            self._load_processed_data()
        else:
            self._process_raw_data()

    def _load_processed_data(self) -> None:
        try:
            data = ProcessedData.load(self.processed_folder)
        except Exception as exc:
            raise RuntimeError(
                f"{self.data_variant}: cannot load processed data from "
                f"{self.processed_folder}. It was written by an incompatible "
                "version; delete the folder and rebuild without "
                "--use-processed."
            ) from exc

        self._validate_manifest(data)
        self._data = data

    def _validate_manifest(self, data: ProcessedData) -> None:
        """Reject a cache built with other data options, loudly but actionably."""
        if data.manifest is None:
            warnings.warn(
                f"{self.data_variant}: processed cache has no manifest; "
                "compatibility with the current data options cannot be verified.",
                stacklevel=2,
            )
            return

        expected = self.config.manifest(split_strategy_for(data.train))
        issue = data.manifest.incompatibility(expected)
        if issue is not None:
            raise RuntimeError(
                f"{self.data_variant}: cannot reuse processed data in "
                f"{self.processed_folder}: {issue}. Delete the folder and rebuild "
                "without --use-processed, or pass the original data options."
            )

    def _process_raw_data(self) -> None:
        result = build_processed(load_raw_data(self.config.dataset), self.config)
        self._data = result.data
        self._build_result = result
        result.data.save(self.processed_folder)

    def _data_generator(self) -> torch.Generator | None:
        if self.config.random_state is None:
            return None

        generator = torch.Generator()
        generator.manual_seed(int(self.config.random_state))

        return generator

    def _dataloader(
        self,
        dataset: Dataset[Any],
        shuffle: bool,
        collate_fn: Callable | None = None,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=settings.NUM_WORKERS,
            shuffle=shuffle,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers and settings.NUM_WORKERS > 0,
            generator=self._data_generator() if shuffle else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(
            self.train_ds, shuffle=True, collate_fn=self.adapter.collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(
            self.val_ds, shuffle=False, collate_fn=self.adapter.collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(
            self.test_ds, shuffle=False, collate_fn=self.adapter.collate_fn
        )

    @property
    def is_processed(self) -> bool:
        return self._data is not None

    def predict_dataloader(self) -> DataLoader:
        return self._dataloader(
            self.predict_ds, shuffle=False, collate_fn=self.adapter.collate_fn
        )

    def evaluation_dataloader(self, split: str = "test") -> DataLoader:
        """Canonical queries, even when training uses a different sample type."""
        return self._dataloader(
            self.adapter.get_queries(split, self.query_config), shuffle=False
        )

    @property
    def has_history(self) -> bool:
        if not self.data.has_temporal_order or not hasattr(self, "train_ds"):
            return False
        queries = self.adapter.get_queries("train", self.query_config)
        return bool(queries.history_valid_mask.any().item())
