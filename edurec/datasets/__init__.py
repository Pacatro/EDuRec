import edurec.datasets.sources  # noqa: F401  registers every dataset spec
from edurec.datasets.adapters import DataAdapter
from edurec.datasets.adapters.queries import QueryConfig
from edurec.datasets.datamodule import ElearningDataModule
from edurec.datasets.features.tabular import DataProcessor, FeatureMetadata
from edurec.datasets.pipeline.prepare import select_dataset
from edurec.datasets.registry import (
    DatasetName,
    RawData,
    dataset_loaders,
    load_raw_data,
)
from edurec.datasets.torch_data import RecSysDataset, RecSysQuery

__all__ = [
    "DataAdapter",
    "DataProcessor",
    "DatasetName",
    "ElearningDataModule",
    "FeatureMetadata",
    "QueryConfig",
    "RawData",
    "RecSysDataset",
    "RecSysQuery",
    "dataset_loaders",
    "load_raw_data",
    "select_dataset",
]
