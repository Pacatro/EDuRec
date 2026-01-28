from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .. import config


def global_preprocessing(df: pd.DataFrame, threshold: float) -> None:
    # We need to encode the user and item ids of all dataset
    df[config.USER_COL] = LabelEncoder().fit_transform(df[config.USER_COL])
    df[config.ITEM_COL] = LabelEncoder().fit_transform(df[config.ITEM_COL])

    # Process time column to timestamp format (nanoseconds)
    if config.TIME_COL in df.columns:
        df[config.TIME_COL] = (
            pd.to_datetime(df[config.TIME_COL]).astype(np.int64) // 10**9
        )

    if config.RELEVANT_COL not in df.columns:
        # An item is relevant if its rating is greater or equal than the threshold
        df[config.RELEVANT_COL] = (df[config.RATING_COL] >= threshold).astype(np.int64)


def process_chunk(
    chunk: pd.DataFrame,
    train_seen: dict[int, set[int]],
    all_items: np.ndarray,
    item_features: dict[int, dict[str, Any]],
    n_neg: int,
    neg_rating: float,
    rng_seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(rng_seed)
    new_rows: list[dict[str, Any]] = []

    records: list[dict[str, Any]] = cast(
        list[dict[str, Any]], chunk.to_dict(orient="records")
    )

    all_items_set: set[int] = set(all_items.tolist())

    for r in records:
        new_rows.append(r)

        u_id = int(r[config.USER_COL])
        i_pos_id = int(r[config.ITEM_COL])

        seen_u = train_seen.get(u_id, set())
        forbidden = seen_u | {i_pos_id}

        candidates_list = list(all_items_set - forbidden)

        if not candidates_list:
            continue

        candidates = np.array(candidates_list)
        replace = len(candidates) < n_neg
        neg_items = rng.choice(candidates, size=n_neg, replace=replace)

        for i_neg in neg_items:
            i_neg_int = int(i_neg)

            r_neg = r.copy()
            r_neg[config.ITEM_COL] = i_neg_int

            feat = item_features.get(i_neg_int)
            if feat:
                r_neg.update(feat)

            r_neg[config.RATING_COL] = neg_rating
            r_neg[config.RELEVANT_COL] = 0
            new_rows.append(r_neg)

    return new_rows


def get_column_types(
    df: pd.DataFrame, id_cols: list[str]
) -> tuple[list[str], dict[str, int]]:
    exclude_cols = id_cols + [config.RATING_COL, config.TIME_COL, config.RELEVANT_COL]
    numeric_cols = []
    categorical_lengths = {}

    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_lengths[col] = int(df[col].nunique())

    return numeric_cols, categorical_lengths
