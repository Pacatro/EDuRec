"""Source cleaning, eligibility filtering and dataset-independent user selection."""

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from edurec import settings
from edurec.datasets.registry import RawData


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
    deduplicate: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove users and items with fewer than ``min_interactions``.

    Repeated ``(user, item)`` events are only collapsed when explicitly
    requested or when no timestamp can order them (see ``split_data``).
    """

    has_time = settings.TIME_COL in interactions.columns
    if deduplicate or not has_time:
        filtered = deduplicate_interactions(interactions)
    else:
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


@dataclass(frozen=True)
class DatasetSelection:
    """A null user limit retains the complete eligible population."""

    max_users: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_users is not None and (
            isinstance(self.max_users, bool)
            or not isinstance(self.max_users, int)
            or self.max_users < 1
        ):
            raise ValueError("max_users must be a positive integer or None.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("selection seed must be an integer.")

    def metadata(self) -> dict[str, Any]:
        # Full selection does not depend on the seed.
        return {**asdict(self), "seed": self.seed if self.max_users else None}


def select_dataset(raw: RawData, selection: DatasetSelection) -> RawData:
    """Select users without inspecting ratings, time, item popularity or metadata.

    Call after source cleaning and eligibility filtering, before splitting or
    fitting features. No interactions of a selected user are removed here.
    Graph entity tables remain available; the processor resolves only entities
    referenced by the retained items.
    """
    events = raw.interactions
    user_col, item_col = settings.USER_COL, settings.ITEM_COL
    users = events[user_col].drop_duplicates().tolist()
    if not users:
        raise ValueError("Dataset has no eligible user histories to select.")
    if selection.max_users is not None and selection.max_users < len(users):

        def priority(user: object) -> tuple[str, str]:
            key = f"{type(user).__name__}:{user}"
            return hashlib.sha256(f"{selection.seed}:{key}".encode()).hexdigest(), key

        users = sorted(users, key=priority)[: selection.max_users]
    selected = events.loc[events[user_col].isin(users)].reset_index(drop=True)
    item_ids = selected[item_col].unique()
    report = {
        "policy": "random_users",
        "config": selection.metadata(),
        "eligible_users": int(events[user_col].nunique()),
        "eligible_items": int(events[item_col].nunique()),
        "eligible_interactions": len(events),
        "selected_users": len(users),
        "selected_items": len(item_ids),
        "selected_interactions": len(selected),
        "mean_history_retention": 1.0,
        "source_report": raw.selection_report,
    }
    return raw._replace(
        interactions=selected,
        user_features=raw.user_features.loc[
            raw.user_features[user_col].isin(users)
        ].reset_index(drop=True),
        item_features=raw.item_features.loc[
            raw.item_features[item_col].isin(item_ids)
        ].reset_index(drop=True),
        selection_report=report,
    )
