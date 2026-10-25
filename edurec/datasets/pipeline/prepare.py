"""Source cleaning, sparsity filtering and interaction budgeting."""

import hashlib

import pandas as pd

from edurec import settings

_BUDGET_SALT = "edurec-max-interactions:"


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


def limit_interactions(
    df: pd.DataFrame,
    max_interactions: int,
) -> pd.DataFrame:
    """Keep whole user histories until the interaction budget is exhausted.

    Users are visited in a stable hash order, so the selected subset is
    reproducible and independent from training seeds. A user that does not fit
    in the remaining budget is skipped; histories are never truncated.
    """
    if isinstance(max_interactions, bool) or max_interactions < 1:
        raise ValueError("max_interactions must be a positive integer.")

    user_col = settings.USER_COL
    if user_col not in df.columns:
        raise ValueError(
            f"Interaction budget requires the {user_col!r} column to select users."
        )
    if len(df) <= max_interactions:
        return df.reset_index(drop=True)

    sizes = df[user_col].value_counts(sort=False)
    order = sorted(
        sizes.index,
        key=lambda user: (
            hashlib.sha256(f"{_BUDGET_SALT}{user}".encode()).digest(),
            str(user),
        ),
    )

    selected: list[object] = []
    used = 0
    for user in order:
        size = int(sizes[user])
        if used + size <= max_interactions:
            selected.append(user)
            used += size
            if used == max_interactions:
                break

    if not selected:
        raise ValueError(
            "max_interactions is too small to retain any complete user history."
        )

    return df.loc[df[user_col].isin(selected)].reset_index(drop=True)


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
    user_support = (
        int(interactions[settings.USER_COL].value_counts().max())
        if len(interactions)
        else 0
    )
    item_support = (
        int(interactions[settings.ITEM_COL].value_counts().max())
        if len(interactions)
        else 0
    )

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

    if filtered.empty:
        raise ValueError(
            f"Sparsity filtering removed every interaction: min_interactions="
            f"{min_interactions} but the densest user has {user_support} "
            f"interactions and the most popular item has {item_support}. "
            "Lower settings.MIN_INTERACTIONS or pass --no-remove-sparse."
        )

    valid_users = filtered[settings.USER_COL].unique()
    valid_items = filtered[settings.ITEM_COL].unique()

    return (
        users.loc[users[settings.USER_COL].isin(valid_users)].reset_index(drop=True),
        items.loc[items[settings.ITEM_COL].isin(valid_items)].reset_index(drop=True),
        filtered.reset_index(drop=True),
    )
