from enum import Enum
from functools import wraps
from typing import Callable

import pandas as pd

from .. import config

type ExportFn = Callable[[], pd.DataFrame]


class DatasetName(str, Enum):
    MARS = "mars"
    ITM = "itm"
    ELEARNING_STUDENT = "elearning"


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
        def wrapper() -> pd.DataFrame:
            return fn()

        dataset_loaders[ds_name] = wrapper
        return wrapper

    return decorator


@register_dataset(DatasetName.MARS)
def load_mars() -> pd.DataFrame:
    """Load and preprocess the MARS dataset.

    This loader combines English and French rating files and merges them with
    item metadata. It also standardizes column names and ensures consistent
    schema across sources.

    Returns:
        pd.DataFrame: The MARS dataset.
    """
    explicit_df_en = pd.read_csv(
        f"{config.DATA_FOLDER}/raw/mars/explicit_ratings_en.csv"
    )
    explicit_df_fr = pd.read_csv(
        f"{config.DATA_FOLDER}/raw/mars/explicit_ratings_fr.csv"
    )

    items_en = pd.read_csv(f"{config.DATA_FOLDER}/raw/mars/items_en.csv")
    items_fr = pd.read_csv(f"{config.DATA_FOLDER}/raw/mars/items_fr.csv")

    df_explicit = pd.concat([explicit_df_en, explicit_df_fr], ignore_index=True)
    df_items = pd.concat([items_en, items_fr], ignore_index=True)

    df_items = df_items.drop(columns=["created_at"])

    df = pd.merge(df_explicit, df_items, on=config.ITEM_COL, how="inner")

    df.rename(
        columns={
            "user_id": config.USER_COL,
            "item_id": config.ITEM_COL,
            "rating": config.RATING_COL,
            "Difficulty": "difficulty",
            "type": "item_type",
            "created_at": config.TIME_COL,
        },
        inplace=True,
    )

    features = [
        config.USER_COL,
        config.ITEM_COL,
        "item_type",
        "difficulty",
        "nb_views",
        "watch_percentage",
        config.RATING_COL,
        config.TIME_COL,
    ]

    result_df = df[features]
    assert isinstance(result_df, pd.DataFrame)
    return result_df


@register_dataset(DatasetName.ITM)
def load_itm() -> pd.DataFrame:
    """Load and preprocess the ITM dataset.

    This loader merges ratings, items, and user information into a unified
    DataFrame and normalizes column names for consistency.

    Returns:
        pd.DataFrame: The ITM dataset.
    """
    ratings_df = pd.read_csv(f"{config.DATA_FOLDER}/raw/itm/ratings.csv")
    items_df = pd.read_csv(f"{config.DATA_FOLDER}/raw/itm/items.csv")
    users_df = pd.read_csv(f"{config.DATA_FOLDER}/raw/itm/users.csv")

    merged_df = pd.merge(left=items_df, right=ratings_df, how="inner", on="Item")
    merged_df = pd.merge(left=merged_df, right=users_df, how="inner", on="UserID")
    merged_df = merged_df.rename(
        columns={
            "UserID": config.USER_COL,
            "Item": config.ITEM_COL,
            "Rating": config.RATING_COL,
        }
    )

    features = [
        config.USER_COL,
        config.ITEM_COL,
        "Title",
        "Semester",
        "Class",
        "App",
        "Lockdown",
        "Ease",
        " Age",
        "Married",
        config.RATING_COL,
    ]

    result_df = merged_df[features]
    assert isinstance(result_df, pd.DataFrame)
    return result_df


@register_dataset(DatasetName.ELEARNING_STUDENT)
def load_elearning_student() -> pd.DataFrame:
    df = pd.read_csv(f"{config.DATA_FOLDER}/raw/elearning/elearning_dataset.csv")

    df = df.rename(
        columns={
            "UserID": config.USER_COL,
            "CourseName": config.ITEM_COL,
            "UserSatisfaction": config.RATING_COL,
        },
    )

    excluded_cols = ["SignUpDate", "LastActiveDate", "FeedbackComments"]

    features = [col for col in df.columns if col not in excluded_cols]

    result_df = df[features]
    assert isinstance(result_df, pd.DataFrame)
    return result_df


def load_data(dataset_name: DatasetName) -> pd.DataFrame:
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
