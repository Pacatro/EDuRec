from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd

if TYPE_CHECKING:
    from edurec.datasets.config import DataConfig


class DatasetName(StrEnum):
    EXPLICIT_MARS = "explicit_mars"
    IMPLICIT_MARS = "implicit_mars"
    ITM = "itm"
    DORIS = "doris"
    MOOCCUBEX = "mooccubex"
    COCO = "coco"


type Schema = dict[str, dict[str, Any]]


class RawData(NamedTuple):
    interactions: pd.DataFrame
    item_features: pd.DataFrame
    user_features: pd.DataFrame
    schema: Schema
    graph_entities: dict[str, pd.DataFrame] | None = None
    reports: dict[str, Any] | None = None


type PrepareFn = Callable[["DataConfig"], RawData]
type DownloadFn = Callable[[], Path]


@dataclass(frozen=True)
class DatasetSpec:
    """Source contract: download and prepare canonical tables before splitting."""

    name: DatasetName
    folder: str
    download: DownloadFn
    prepare: PrepareFn


DATASETS: dict[DatasetName, DatasetSpec] = {}


def register_spec(spec: DatasetSpec) -> None:
    DATASETS[spec.name] = spec


def get_spec(dataset_name: DatasetName) -> DatasetSpec:
    spec = DATASETS.get(dataset_name)
    if spec is None:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return spec


def download_raw_data(dataset_name: DatasetName) -> Path:
    return get_spec(dataset_name).download()


def prepare_raw_data(config: "DataConfig") -> RawData:
    """Run the source's explicit preparation sequence, including common filters."""
    return get_spec(config.dataset).prepare(config)
