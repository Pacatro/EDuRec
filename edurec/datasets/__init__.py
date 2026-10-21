import edurec.datasets.sources  # noqa: F401  registers every dataset spec
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
    "DataProcessor",
    "DatasetName",
    "ElearningDataModule",
    "FeatureMetadata",
    "RawData",
    "RecSysDataset",
    "RecSysQuery",
    "dataset_loaders",
    "load_raw_data",
    "select_dataset",
]
