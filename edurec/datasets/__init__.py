"""Public data API: sources, canonical processing, model adapters and loaders."""

import edurec.datasets.sources  # noqa: F401  registers every dataset spec
from edurec.datasets import graphs, queries
from edurec.datasets.adapter import AdapterFactory, DataAdapter
from edurec.datasets.cache import ProcessedData
from edurec.datasets.config import DataConfig
from edurec.datasets.datamodule import ElearningDataModule
from edurec.datasets.queries import QueryConfig
from edurec.datasets.sources.registry import DATASETS, DatasetName, RawData
from edurec.datasets.torch_data import RecSysDataset, RecSysQuery

__all__ = [
    "DATASETS",
    "AdapterFactory",
    "DataAdapter",
    "DataConfig",
    "DatasetName",
    "ElearningDataModule",
    "ProcessedData",
    "QueryConfig",
    "RawData",
    "RecSysDataset",
    "RecSysQuery",
    "graphs",
    "queries",
]
