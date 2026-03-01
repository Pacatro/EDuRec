from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
from typing import Callable

import numpy as np
import pandas as pd

from .. import config


class DatasetName(StrEnum):
    MARS = "mars"
    ITM = "itm"
    DORIS = "doris"


type Schema = dict[str, dict[str, list[str]]]


@dataclass
class RawDataset:
    interactions: pd.DataFrame
    i_feats: pd.DataFrame
    u_feats: pd.DataFrame
    schema: Schema


type ExportFn = Callable[[], RawDataset]

dataset_loaders: dict[DatasetName, ExportFn] = {}


def add_relevant_col(df: pd.DataFrame) -> None:
    """
    Add a boolean column to the dataframe indicating if an item is relevant.

    Relevance is determined by comparing each interaction's rating against a
    dynamic threshold:

    1. For users with enough activity (>= config.MIN_INTERACTIONS), the
       threshold is the user's mean rating.
    2. For users with fewer interactions, the threshold defaults to the
       global mean rating of the entire dataset.

    Args:
        df (pd.DataFrame): The interactions dataframe. It must contain the
            columns defined in config.USER_COL and config.RATING_COL.
    """
    if config.RELEVANT_COL in df.columns:
        return

    global_thershold = df[config.RATING_COL].mean()
    user_stats = df.groupby(config.USER_COL)[config.RATING_COL]

    mean_user_ratings = user_stats.transform("mean")
    count_user_raings = user_stats.transform("count")

    thresholds = np.where(
        count_user_raings < config.MIN_INTERACTIONS,
        global_thershold,
        mean_user_ratings,
    )

    df[config.RELEVANT_COL] = df[config.RATING_COL] >= thresholds


def clean_df(df: pd.DataFrame) -> None:
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )


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
    """Load and preprocess the MARS dataset.

    This loader combines English and French rating files and merges them with
    item metadata. It also standardizes column names and ensures consistent
    schema across sources.

    Returns:
        pd.DataFrame: The MARS dataset.
    """
    items_en = pd.read_csv(f"{config.DATA_FOLDER}/raw/mars/items_en.csv")
    items_fr = pd.read_csv(f"{config.DATA_FOLDER}/raw/mars/items_fr.csv")
    users_en = pd.read_csv(f"{config.DATA_FOLDER}/raw/mars/users_en.csv")
    users_fr = pd.read_csv(f"{config.DATA_FOLDER}/raw/mars/users_fr.csv")
    ratings_en = pd.read_csv(f"{config.DATA_FOLDER}/raw/mars/explicit_ratings_en.csv")
    ratings_fr = pd.read_csv(f"{config.DATA_FOLDER}/raw/mars/explicit_ratings_fr.csv")

    df_interactions = pd.concat([ratings_en, ratings_fr], ignore_index=True)
    df_items = pd.concat([items_en, items_fr], ignore_index=True)
    df_users = pd.concat([users_en, users_fr], ignore_index=True)

    df_interactions.rename(
        columns={
            "user_id": config.USER_COL,
            "item_id": config.ITEM_COL,
            "rating": config.RATING_COL,
            "created_at": config.TIME_COL,
        },
        inplace=True,
    )

    df_items.rename(
        columns={"item_id": config.ITEM_COL, "type": "item_type"},
        inplace=True,
    )

    df_users.rename(columns={"user_id": config.USER_COL})

    clean_df(df_interactions)
    clean_df(df_items)
    clean_df(df_users)

    add_relevant_col(df_interactions)

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
    """Load and preprocess the ITM dataset.

    This loader merges ratings, items, and user information into a unified
    DataFrame and normalizes column names for consistency.

    Returns:
        pd.DataFrame: The ITM dataset.
    """
    ratings_df = pd.read_csv(f"{config.DATA_FOLDER}/raw/itm/ratings.csv")
    items_df = pd.read_csv(f"{config.DATA_FOLDER}/raw/itm/items.csv")
    users_df = pd.read_csv(f"{config.DATA_FOLDER}/raw/itm/users.csv")

    ratings_df.rename(
        columns={
            "UserID": config.USER_COL,
            "Item": config.ITEM_COL,
            "Rating": config.RATING_COL,
        },
        inplace=True,
    )
    items_df.rename(columns={"Item": config.ITEM_COL}, inplace=True)
    users_df.rename(columns={"UserID": config.USER_COL}, inplace=True)

    items_df = items_df.drop(["URL"], axis=1)

    clean_df(ratings_df)
    clean_df(items_df)
    clean_df(users_df)

    add_relevant_col(ratings_df)

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


# TODO: Return df and some other info metadata
def load_data(dataset_name: DatasetName) -> RawDataset:
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
