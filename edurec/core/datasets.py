from enum import Enum
import pandas as pd


class DatasetName(str, Enum):
    mars = "mars"
    itm = "itm"


def load_mars() -> pd.DataFrame:
    """Load the MARS dataset.

    Returns:
        pd.DataFrame: The MARS dataset.
    """
    explicit_df_en = pd.read_csv("./data/mars_dataset/explicit_ratings_en.csv")
    explicit_df_fr = pd.read_csv("./data/mars_dataset/explicit_ratings_fr.csv")

    items_en = pd.read_csv("./data/mars_dataset/items_en.csv")
    items_fr = pd.read_csv("./data/mars_dataset/items_fr.csv")

    df_explicit = pd.concat([explicit_df_en, explicit_df_fr], ignore_index=True)
    df_items = pd.concat([items_en, items_fr], ignore_index=True)

    df_explicit["created_at"] = pd.to_datetime(df_explicit["created_at"])
    df_items = df_items.drop(columns=["created_at"])

    df = pd.merge(df_explicit, df_items, on="item_id", how="inner")

    df.rename(
        columns={"Difficulty": "difficulty", "type": "item_type"},
        inplace=True,
    )

    features = [
        "user_id",
        "item_id",
        "item_type",
        "difficulty",
        "nb_views",
        "watch_percentage",
        "rating",
    ]

    result_df = df[features]
    assert isinstance(result_df, pd.DataFrame)
    return result_df


def load_itm() -> pd.DataFrame:
    """Load the ITM dataset.

    Returns:
        pd.DataFrame: The ITM dataset.
    """
    ratings_df = pd.read_csv("./data/itm_dataset/ratings.csv")
    items_df = pd.read_csv("./data/itm_dataset/items.csv")
    users_df = pd.read_csv("./data/itm_dataset/users.csv")

    merged_df = pd.merge(left=items_df, right=ratings_df, how="inner", on="Item")
    merged_df = pd.merge(left=merged_df, right=users_df, how="inner", on="UserID")
    merged_df = merged_df.rename(
        columns={"UserID": "user_id", "Item": "item_id", "Rating": "rating"}
    )

    features = [
        "user_id",
        "item_id",
        "Title",
        "Semester",
        "Class",
        "App",
        "Lockdown",
        "Ease",
        " Age",
        "Married",
        "rating",
    ]

    result_df = merged_df[features]
    assert isinstance(result_df, pd.DataFrame)
    return result_df


def load_data(dataset_name: DatasetName) -> pd.DataFrame:
    """Load the specified dataset.

    Args:
        dataset_name (DatasetName): The name of the dataset to load.

    Raises:
        ValueError: If the dataset name is not supported.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    match dataset_name:
        case DatasetName.mars:
            return load_mars()
        case DatasetName.itm:
            return load_itm()
