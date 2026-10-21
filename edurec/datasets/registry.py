from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd


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
    selection_report: dict[str, Any] | None = None


type ExportFn = Callable[[], RawData]
type DownloadFn = Callable[[], Path]


@dataclass(frozen=True)
class DatasetSpec:
    """Source contract of one dataset: raw folder, download step and loader."""

    name: DatasetName
    folder: str
    download: DownloadFn
    load: ExportFn


dataset_loaders: dict[DatasetName, ExportFn] = {}
_specs: dict[DatasetName, DatasetSpec] = {}


def register_spec(spec: DatasetSpec) -> None:
    _specs[spec.name] = spec
    dataset_loaders[spec.name] = spec.load


def get_spec(dataset_name: DatasetName) -> DatasetSpec:
    spec = _specs.get(dataset_name)
    if spec is None:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return spec


def download_raw_data(dataset_name: DatasetName) -> Path:
    return get_spec(dataset_name).download()


def load_raw_data(dataset_name: DatasetName) -> RawData:
    return get_spec(dataset_name).load()
