"""Source cleaning and sparsity filtering."""

import pandas as pd

from edurec import settings


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def deduplicate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated ``(user, item)`` pairs into a single interaction.

    Keeping duplicate pairs would let the same user-item interaction appear in
    more than one split (train/test leakage). When a timestamp is present the
    most recent event is kept, otherwise the last occurrence in file order is.
    """
    subset = [settings.USER_COL, settings.ITEM_COL]
    if not set(subset).issubset(df.columns):
        return df

    frame = df
    if settings.TIME_COL in frame.columns and frame[settings.TIME_COL].notna().any():
        frame = frame.sort_values(
            [settings.USER_COL, settings.TIME_COL], kind="mergesort"
        )
    return frame.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)


def filter_sparse(
    users: pd.DataFrame,
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    min_interactions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove users and items with fewer than ``min_interactions``.

    Interactions are expected to be normalized already: repeated pairs are
    collapsed once before filtering.
    """
    filtered = interactions.copy()

    while True:
        previous_size = len(filtered)

        user_counts = filtered[settings.USER_COL].value_counts(sort=False)
        item_counts = filtered[settings.ITEM_COL].value_counts(sort=False)

        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index

        filtered = filtered.loc[
            filtered[settings.USER_COL].isin(valid_users)
            & filtered[settings.ITEM_COL].isin(valid_items)
        ]

        if len(filtered) == previous_size:
            break

    valid_users = filtered[settings.USER_COL].unique()
    valid_items = filtered[settings.ITEM_COL].unique()

    return (
        users.loc[users[settings.USER_COL].isin(valid_users)].reset_index(drop=True),
        items.loc[items[settings.ITEM_COL].isin(valid_items)].reset_index(drop=True),
        filtered.reset_index(drop=True),
    )
