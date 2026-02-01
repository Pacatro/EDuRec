import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .. import config


def global_preprocessing(df: pd.DataFrame) -> None:
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
        # The threshold is the mean of the ratings of the user
        mean_user_ratings = df[config.USER_COL].map(
            df.groupby(config.USER_COL)[config.RATING_COL].mean()
        )
        df[config.RELEVANT_COL] = df[config.RATING_COL] >= mean_user_ratings


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


def generate_neg_samples(df: pd.DataFrame, neg_samples: int) -> pd.DataFrame:
    print("Generating negative samples...")
    new_data = []
    user_item_set = df.groupby(config.USER_COL)[config.ITEM_COL].apply(set).to_dict()

    item_features_map = (
        df.drop_duplicates(config.ITEM_COL)
        .set_index(config.ITEM_COL)
        .to_dict(orient="index")
    )

    all_items = df[config.ITEM_COL].unique()
    columns = df.columns.tolist()

    user_idx = columns.index(config.USER_COL)
    item_idx = columns.index(config.ITEM_COL)
    rating_idx = columns.index(config.RATING_COL)
    rel_idx = columns.index(config.RELEVANT_COL)

    for row in df.itertuples(index=False):
        row_list = list(row)
        new_data.append(row_list)

        user_id = row_list[user_idx]
        negatives_found = 0

        while negatives_found < neg_samples:
            neg_id = np.random.choice(all_items)

            if neg_id not in user_item_set[user_id]:
                neg_row = row_list.copy()

                neg_row[item_idx] = neg_id
                neg_row[rating_idx] = 0.0
                neg_row[rel_idx] = False

                if neg_id in item_features_map:
                    item_attrs = item_features_map[neg_id]
                    for col_name, col_value in item_attrs.items():
                        if col_name in columns:
                            neg_row[columns.index(str(col_name))] = col_value

                new_data.append(neg_row)
                negatives_found += 1

    return pd.DataFrame(new_data, columns=columns)
