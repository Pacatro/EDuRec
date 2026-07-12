from enum import StrEnum
from functools import wraps
from pathlib import Path
from typing import Callable, NamedTuple, Literal

import numpy as np
import pandas as pd

from .. import settings

RAW_DATA_FOLDER = Path(settings.DATA_FOLDER) / "raw"


class DatasetName(StrEnum):
    EXPLICIT_MARS = "explicit_mars"
    IMPLICIT_MARS = "implicit_mars"
    ITM = "itm"
    DORIS = "doris"


type Schema = dict[str, dict[str, list[str]]]


class RawData(NamedTuple):
    interactions: pd.DataFrame
    item_features: pd.DataFrame
    user_features: pd.DataFrame
    schema: Schema


type ExportFn = Callable[[], RawData]

dataset_loaders: dict[DatasetName, ExportFn] = {}


def register_dataset(ds_name: DatasetName) -> Callable[[ExportFn], ExportFn]:
    """Decorator for registering a dataset loader.

    This decorator associates a dataset loading function with a given
    `DatasetName` and stores it in the global `dataset_loaders` dictionary.

    Example:
        ```python
        @register_dataset(DatasetName.EXPLICIT_MARS)
        def load_explicit_mars() -> RawData:
            ...
        ```

    Args:
        ds_name (DatasetName): The dataset identifier to register.

    Returns:
        Callable[[ExportFn], ExportFn]: The decorator that registers the loader.
    """

    def decorator(fn: ExportFn) -> ExportFn:
        @wraps(fn)
        def wrapper() -> RawData:
            return fn()

        dataset_loaders[ds_name] = wrapper
        return wrapper

    return decorator


def load_mars(data_type: Literal["explicit", "implicit"]) -> RawData:
    """Load the MARS dataset.

    This loader combines English and French rating files and merges them with
    item metadata. Dataset-wide cleaning and filtering happen later in the
    datamodule preprocessing phase.

    Returns:
        pd.DataFrame: The MARS dataset.
    """
    mars_folder = RAW_DATA_FOLDER / "mars"
    items_en = pd.read_csv(mars_folder / "items_en.csv")
    items_fr = pd.read_csv(mars_folder / "items_fr.csv")
    users_en = pd.read_csv(mars_folder / "users_en.csv")
    users_fr = pd.read_csv(mars_folder / "users_fr.csv")
    ratings_en = pd.read_csv(mars_folder / f"{data_type}_ratings_en.csv")
    ratings_fr = pd.read_csv(mars_folder / f"{data_type}_ratings_fr.csv")

    interactions = pd.concat([ratings_en, ratings_fr], ignore_index=True)
    items = pd.concat([items_en, items_fr], ignore_index=True)
    users = pd.concat([users_en, users_fr], ignore_index=True)

    interactions.rename(
        columns={
            "user_id": settings.USER_COL,
            "item_id": settings.ITEM_COL,
            "rating": settings.RATING_COL,
            "created_at": settings.TIME_COL,
        },
        inplace=True,
    )

    items.rename(
        columns={"item_id": settings.ITEM_COL, "type": "item_type"},
        inplace=True,
    )

    users.rename(columns={"user_id": settings.USER_COL}, inplace=True)

    schema = {
        "users": {
            "bin": [],
            "num": [],
            "cat": ["job"],
            "text": [],
            "list": [],
        },
        "items": {
            "bin": [],
            "num": ["nb_views", "duration"],
            "cat": ["language", "difficulty", "item_type"],
            "text": ["name", "description"],
            "list": ["job", "software", "theme"],
        },
        "inter": {
            "bin": [],
            "num": ["watch_percentage"] if data_type == "explicit" else [],
            "cat": [],
            "text": [],
            "list": [],
        },
    }

    return RawData(
        interactions=interactions,
        item_features=items,
        user_features=users,
        schema=schema,
    )


@register_dataset(DatasetName.EXPLICIT_MARS)
def load_explicit_mars() -> RawData:
    return load_mars(data_type="explicit")


@register_dataset(DatasetName.IMPLICIT_MARS)
def load_implicit_mars() -> RawData:
    return load_mars(data_type="implicit")


@register_dataset(DatasetName.ITM)
def load_itm() -> RawData:
    """Load the ITM dataset.

    This loader merges ratings, items, and user information into a unified
    DataFrame. Dataset-wide cleaning and filtering happen later in the
    datamodule preprocessing phase.

    Returns:
        pd.DataFrame: The ITM dataset.
    """
    itm_folder = RAW_DATA_FOLDER / DatasetName.ITM.value
    ratings_df = pd.read_csv(itm_folder / "ratings.csv")
    items_df = pd.read_csv(itm_folder / "items.csv")
    users_df = pd.read_csv(itm_folder / "users.csv")

    ratings_df.rename(
        columns={
            "UserID": settings.USER_COL,
            "Item": settings.ITEM_COL,
            "Rating": settings.RATING_COL,
        },
        inplace=True,
    )
    # We add a time column to ensure that the data works for sequential models
    ratings_df[settings.TIME_COL] = np.arange(len(ratings_df), dtype=np.int64)
    items_df.rename(columns={"Item": settings.ITEM_COL}, inplace=True)
    users_df.rename(columns={"UserID": settings.USER_COL}, inplace=True)

    items_df = items_df.drop(["URL"], axis=1)

    schema = {
        "users": {
            "bin": ["gender", "married"],
            "num": [],
            "cat": ["age"],
            "text": [],
            "list": [],
        },
        "items": {
            "bin": [],
            "num": [],
            "cat": [],
            "text": ["title", "descriptions"],
            "list": [],
        },
        "inter": {
            "bin": [],
            "num": ["app", "data", "ease"],
            "cat": ["class", "semester", "lockdown"],
            "text": [],
            "list": [],
        },
    }

    return RawData(
        interactions=ratings_df,
        item_features=items_df,
        user_features=users_df,
        schema=schema,
    )


@register_dataset(DatasetName.DORIS)
def load_doris() -> RawData:
    doris_folder = RAW_DATA_FOLDER / DatasetName.DORIS.value
    ratings_df = pd.read_excel(doris_folder / "CourseSelectionTable.xlsx")
    items_df = pd.read_excel(doris_folder / "CourseInformationTable.xlsx")
    users_df = pd.read_excel(doris_folder / "StudentInformationTable.xlsx")

    ratings_df.rename(
        columns={
            "StudedntId": settings.USER_COL,
            "CourseId": settings.ITEM_COL,
            "Score": settings.RATING_COL,
            "AcademicYear": settings.TIME_COL,
        },
        inplace=True,
    )

    ratings_df[settings.RATING_COL] = pd.to_numeric(
        ratings_df[settings.RATING_COL],
        errors="coerce",
    )
    ratings_df = ratings_df.dropna(subset=[settings.RATING_COL]).reset_index(drop=True)

    start_year = (
        ratings_df[settings.TIME_COL]
        .astype("string")
        .str.extract(r"^(\d{2})-", expand=False)
    )
    ratings_df[settings.TIME_COL] = pd.to_numeric("20" + start_year, errors="coerce")

    ratings_df = ratings_df.dropna(subset=[settings.TIME_COL]).reset_index(drop=True)
    ratings_df[settings.TIME_COL] = ratings_df[settings.TIME_COL].astype(np.int64)
    items_df.rename(
        columns={"CourseId": settings.ITEM_COL, "type": "item_type"}, inplace=True
    )
    users_df.rename(columns={"StudentId": settings.USER_COL}, inplace=True)

    schema = {
        "users": {
            "bin": [],
            "num": [],
            "cat": ["enrollmentyear", "education", "major"],
            "text": [],
            "list": [],
        },
        "items": {
            "bin": [],
            "num": [],
            "cat": ["item_type", "grade", "prerequisite"],
            "text": ["introduction"],
            "list": [],
        },
        "inter": {
            "bin": [],
            "num": [],
            "cat": ["semester", "coursecollege"],
            "text": ["coursename"],
            "list": [],
        },
    }

    return RawData(
        interactions=ratings_df,
        item_features=items_df,
        user_features=users_df,
        schema=schema,
    )


def load_raw_data(dataset_name: DatasetName) -> RawData:
    """
    Load the specified dataset. If data was processed before, laod the data from disk.

    Args:
        dataset_name (DatasetName): The name of the dataset to load.

    Raises:
        ValueError: If the dataset name is not supported.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """

    loader = dataset_loaders.get(dataset_name)

    if loader is None:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    return loader()
