"""Public data API: sources, canonical processing, model adapters and loaders."""

import edurec.datasets.sources  # noqa: F401  registers every dataset spec
from edurec.datasets.adapters import (
    AdapterMetadata,
    ModelDataAdapter,
    ProposedAdapter,
    create_adapter,
    register_adapter,
)
from edurec.datasets.cache import ProcessedData
from edurec.datasets.config import DataConfig, default_min_interactions
from edurec.datasets.datamodule import ElearningDataModule
from edurec.datasets.queries import QueryConfig
from edurec.datasets.sources.registry import (
    DATASETS,
    DatasetName,
    RawData,
    load_raw_data,
)
from edurec.datasets.torch_data import RecSysDataset, RecSysQuery

__all__ = [
    "DATASETS",
    "AdapterMetadata",
    "DataConfig",
    "DatasetName",
    "ElearningDataModule",
    "ModelDataAdapter",
    "ProcessedData",
    "ProposedAdapter",
    "QueryConfig",
    "RawData",
    "RecSysDataset",
    "RecSysQuery",
    "create_adapter",
    "default_min_interactions",
    "load_raw_data",
    "register_adapter",
]
