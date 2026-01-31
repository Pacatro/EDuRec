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
        df[config.RELEVANT_COL] = df[config.RATING_COL] >= threshold


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
