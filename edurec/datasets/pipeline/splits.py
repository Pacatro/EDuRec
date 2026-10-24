"""Temporal/random splits and relevance labels.

Histories, interaction context and negative sampling live in
:mod:`edurec.datasets.adapters.queries`: they are query views, not part of the
preprocessing protocol. Splitting and relevance stay here because they define
the canonical catalog, labels and partitions.
"""

import numpy as np
import pandas as pd

from edurec import settings

SPLIT_ORDER = ("train", "val", "test")


def split_data(
    df: pd.DataFrame,
    test_ratio: float,
    val_ratio: float,
    min_interactions: int,
    random_state: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Callers must normalize repeated pairs; build_processed deduplicates once
    # before filtering and this split.
    if not 0 < test_ratio < 1 or not 0 < val_ratio < 1 or test_ratio + val_ratio >= 1:
        raise ValueError(
            "Validation/test ratios must be positive and sum to less than 1."
        )
    if min_interactions < 3:
        raise ValueError(
            "min_interactions must be at least 3 to create train/val/test splits."
        )
    has_time = settings.TIME_COL in df.columns
    rng = np.random.default_rng(random_state)
    splits = {"train": [], "val": [], "test": []}

    for _, user_df in df.groupby(settings.USER_COL, sort=False):
        if has_time:
            user_df = user_df.sort_values(settings.TIME_COL, kind="mergesort")

        n = len(user_df)
        if n < min_interactions:
            splits["train"].append(user_df)
            continue

        n_test = max(1, int(np.floor(n * test_ratio)))
        n_val = max(1, int(np.floor(n * val_ratio)))
        if n_test + n_val >= n:
            n_test = n_val = 1

        if has_time:
            splits["train"].append(user_df.iloc[: -(n_test + n_val)])
            splits["val"].append(user_df.iloc[-(n_test + n_val) : -n_test])
            splits["test"].append(user_df.iloc[-n_test:])
        else:
            order = rng.permutation(n)
            splits["test"].append(user_df.iloc[order[:n_test]])
            splits["val"].append(user_df.iloc[order[n_test : n_test + n_val]])
            splits["train"].append(user_df.iloc[order[n_test + n_val :]])

    frames = [
        (
            pd.concat(splits[name], axis=0) if splits[name] else df.iloc[:0].copy()
        ).reset_index(drop=True)
        for name in SPLIT_ORDER
    ]
    return frames[0], frames[1], frames[2]


def get_relevance_threshold(train_df: pd.DataFrame) -> tuple[pd.Series, float] | None:
    """Fit per-user relevance thresholds using the training split only."""
    if settings.RATING_COL not in train_df.columns:
        return None

    user_mean = train_df.groupby(settings.USER_COL)[settings.RATING_COL].mean()
    global_mean = train_df[settings.RATING_COL].mean()

    return user_mean, global_mean


def add_relevance(
    df: pd.DataFrame,
    thresholds: tuple[pd.Series, float] | None,
) -> pd.DataFrame:
    """Normalize explicit ratings or implicit events to binary relevance."""
    df = df.copy()

    if settings.RATING_COL not in df.columns:
        df[settings.RELEVANT_COL] = 1
        return df.reset_index(drop=True)

    if thresholds is None:
        raise RuntimeError("Relevance thresholds are required for explicit feedback.")

    user_mean, global_mean = thresholds
    threshold = df[settings.USER_COL].map(user_mean).fillna(global_mean)
    df[settings.RELEVANT_COL] = df[settings.RATING_COL] >= threshold

    return df.reset_index(drop=True)
