from enum import StrEnum
from functools import wraps
from pathlib import Path
from typing import Callable, NamedTuple

import pandas as pd

from .. import settings

RAW_DATA_FOLDER = Path(settings.DATA_FOLDER) / "raw"


class DatasetName(StrEnum):
    MARS = "mars"
    ITM = "itm"
    DORIS = "doris"


type Schema = dict[str, dict[str, list[str]]]


class RawDataset(NamedTuple):
    interactions: pd.DataFrame
    i_feats: pd.DataFrame
    u_feats: pd.DataFrame
    schema: Schema


type ExportFn = Callable[[], RawDataset]

dataset_loaders: dict[DatasetName, ExportFn] = {}


def register_dataset(ds_name: DatasetName) -> Callable[[ExportFn], ExportFn]:
    """Decorator for registering a dataset loader.

    This decorator associates a dataset loading function with a given
    `DatasetName` and stores it in the global `dataset_loaders` dictionary.

    Example:
        ```python
        @register_dataset(DatasetName.MARS)
        def load_mars() -> pd.DataFrame:
            ...
        ```

    Args:
        ds_name (DatasetName): The dataset identifier to register.

    Returns:
        Callable[[ExportFn], ExportFn]: The decorator that registers the loader.
    """

    def decorator(fn: ExportFn) -> ExportFn:
        @wraps(fn)
        def wrapper() -> RawDataset:
            return fn()

        dataset_loaders[ds_name] = wrapper
        return wrapper

    return decorator


@register_dataset(DatasetName.MARS)
def load_mars() -> RawDataset:
    """Load the MARS dataset.

    This loader combines English and French rating files and merges them with
    item metadata. Dataset-wide cleaning and filtering happen later in the
    datamodule preprocessing phase.

    Returns:
        pd.DataFrame: The MARS dataset.
    """
    mars_folder = RAW_DATA_FOLDER / DatasetName.MARS.value
    items_en = pd.read_csv(mars_folder / "items_en.csv")
    items_fr = pd.read_csv(mars_folder / "items_fr.csv")
    users_en = pd.read_csv(mars_folder / "users_en.csv")
    users_fr = pd.read_csv(mars_folder / "users_fr.csv")
    ratings_en = pd.read_csv(mars_folder / "explicit_ratings_en.csv")
    ratings_fr = pd.read_csv(mars_folder / "explicit_ratings_fr.csv")

    df_interactions = pd.concat([ratings_en, ratings_fr], ignore_index=True)
    df_items = pd.concat([items_en, items_fr], ignore_index=True)
    df_users = pd.concat([users_en, users_fr], ignore_index=True)

    df_interactions.rename(
        columns={
            "user_id": settings.USER_COL,
            "item_id": settings.ITEM_COL,
            "rating": settings.RATING_COL,
            "created_at": settings.TIME_COL,
        },
        inplace=True,
    )

    df_items.rename(
        columns={"item_id": settings.ITEM_COL, "type": "item_type"},
        inplace=True,
    )

    df_users.rename(columns={"user_id": settings.USER_COL}, inplace=True)

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
            "num": ["watch_percentage"],
            "cat": [],
            "text": [],
            "list": [],
        },
    }

    return RawDataset(
        interactions=df_interactions, i_feats=df_items, u_feats=df_users, schema=schema
    )


@register_dataset(DatasetName.ITM)
def load_itm() -> RawDataset:
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
    items_df.rename(columns={"Item": settings.ITEM_COL}, inplace=True)
    users_df.rename(columns={"UserID": settings.USER_COL}, inplace=True)

    items_df = items_df.drop(["URL"], axis=1)

    schema = {
        "users": {
            "bin": ["genre", "married"],
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

    return RawDataset(
        interactions=ratings_df,
        i_feats=items_df,
        u_feats=users_df,
        schema=schema,
    )


@register_dataset(DatasetName.DORIS)
def load_doris() -> RawDataset:
    doris_folder = RAW_DATA_FOLDER / DatasetName.DORIS.value
    ratings_df = pd.read_excel(doris_folder / "CourseSelectionTable.xlsx")
    items_df = pd.read_excel(doris_folder / "CourseInformationTable.xlsx")
    users_df = pd.read_excel(doris_folder / "StudentInformationTable.xlsx")

    ratings_df.rename(
        columns={
            "StudedntId": settings.USER_COL,
            "CourseId": settings.ITEM_COL,
            "Score": settings.RATING_COL,
        },
        inplace=True,
    )
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
            "cat": ["academicyear", "semester", "coursecollage"],
            "text": ["coursename"],
            "list": [],
        },
    }

    return RawDataset(
        interactions=ratings_df,
        i_feats=items_df,
        u_feats=users_df,
        schema=schema,
    )


def load_raw_data(dataset_name: DatasetName) -> RawDataset:
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
