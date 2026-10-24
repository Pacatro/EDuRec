"""Lightning data module: assemble queries and data loaders.

All data views (interaction context, histories, negatives, knowledge graph and
RecBole exports) come from :class:`~edurec.datasets.adapters.DataAdapter`. This
module only builds/loads the canonical ``ProcessedData``, validates the cache
manifest and serves ``DataLoader`` instances.
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
from edurec.datasets.config import DataConfig
from edurec.datasets.pipeline.build import (
    BuildResult,
    build_processed,
    split_strategy_for,
)
from edurec.datasets.registry import download_raw_data, load_raw_data
from edurec.datasets.torch_data import RecSysDataset


class ElearningDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: DataConfig,
        batch_size: int,
        use_processed_data: bool = False,
        max_history: int = settings.MAX_HISTORY_LEN,
        persistent_workers: bool = True,
    ):
        super().__init__()
        # Lightning rejects frozen dataclasses when it serialises hparams for
        # logging, so store a plain mapping instead of the config instance.
        self.save_hyperparameters(
            {
                "config": config.to_dict(),
                "batch_size": batch_size,
                "use_processed_data": use_processed_data,
                "max_history": max_history,
                "persistent_workers": persistent_workers,
            }
        )
        self.config = config
        self.batch_size = batch_size
        self.use_processed_data = use_processed_data
        self.max_history = max_history
        self.persistent_workers = persistent_workers
        self.data_variant = config.dataset.value

        self.processed_folder = Path(settings.PROCESSED_FOLDER) / self.data_variant
        self.atomic_folder = Path(settings.ATOMICFILES_FOLDER) / self.data_variant

        self._data: ProcessedData | None = None
        self._build_result: BuildResult | None = None
        self._adapter: DataAdapter | None = None

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
    def adapter(self) -> DataAdapter:
        """Lazy facade over the processed artifacts."""
        if self._adapter is None:
            self._adapter = DataAdapter(self.data)
        return self._adapter

    @property
    def query_config(self) -> QueryConfig:
        """History and negative policy used by every query view of this run."""
        return QueryConfig(
            max_history=self.max_history,
            negatives_per_positive=settings.TRAIN_NEGATIVES_PER_POSITIVE,
            random_state=self.config.random_state,
            use_histories=self.data.has_temporal_order,
        )

    def prepare_data(self) -> None:
        if self.use_processed_data and self.processed_folder.exists():
            return

        download_raw_data(self.config.dataset)

    def setup(self, stage: str | None = None):
        self._ensure_data()

        config = self.query_config
        if stage in ("fit", None):
            self.train_ds = self.adapter.get_queries("train", config=config)
            self.val_ds = self.adapter.get_queries("val", config=config)
        if stage in ("test", None):
            self.test_ds = self.adapter.get_queries("test", config=config)

    def export_recbole(self, mode: str = "general") -> dict[str, Path]:
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
        return self._data is not None

    @property
    def has_history(self) -> bool:
        train_ds = getattr(self, "train_ds", None)

        return bool(
            self.data.has_temporal_order
            and train_ds is not None
            and train_ds.history_valid_mask.numel() > 0
            and train_ds.history_valid_mask.any().item()
        )
