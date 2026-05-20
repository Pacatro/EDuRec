import numpy as np
import pandas as pd
import torch

from .. import settings
from .cache import ProcessedData
from .dataprocessor import DataProcessor


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def split_data(
    df: pd.DataFrame,
    test_ratio: float,
    val_ratio: float,
    min_interactions: int,
    random_state: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    has_time = settings.TIME_COL in df.columns
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

    train_split = pd.concat(splits["train"], axis=0).reset_index(drop=True)
    val_split = pd.concat(splits["val"], axis=0).reset_index(drop=True)
    test_split = pd.concat(splits["test"], axis=0).reset_index(drop=True)

    return train_split, val_split, test_split


def filter_sparse(
    users: pd.DataFrame,
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    min_interactions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove users with fewer than ``min_interactions`` in the full dataset."""
    user_counts = interactions[settings.USER_COL].value_counts(sort=False)
    valid_users = user_counts[user_counts >= min_interactions].index
    user_mask = users[settings.USER_COL].isin(valid_users)
    interaction_mask = interactions[settings.USER_COL].isin(valid_users)

    return (
        users.loc[user_mask].reset_index(drop=True),
        items.reset_index(drop=True),
        interactions.loc[interaction_mask].reset_index(drop=True),
    )


def get_relevance_threshold(train_df: pd.DataFrame) -> tuple[pd.Series, float] | None:
    if settings.RATING_COL not in train_df.columns:
        return None

    user_mean = train_df.groupby(settings.USER_COL)[settings.RATING_COL].mean()
    global_mean = train_df[settings.RATING_COL].mean()

    return user_mean, global_mean


def add_relevance(
    df: pd.DataFrame,
    thresholds: tuple[pd.Series, float] | None,
) -> pd.DataFrame:
    df = df.copy()

    if settings.RELEVANT_COL in df.columns:
        return df.reset_index(drop=True)

    if settings.RATING_COL not in df.columns:
        df[settings.RELEVANT_COL] = 1
        return df.reset_index(drop=True)

    if thresholds is None:
        raise RuntimeError("Relevance thresholds are required for rated interactions.")

    user_mean, global_mean = thresholds
    threshold = df[settings.USER_COL].map(user_mean).fillna(global_mean)
    df[settings.RELEVANT_COL] = df[settings.RATING_COL] >= threshold

    return df.reset_index(drop=True)


def preprocess(
    processor: DataProcessor,
    users: pd.DataFrame,
    items: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> ProcessedData:
    processor.fit(users_train=users, items_train=items, interactions_train=train)

    features = processor.transform(users=users, items=items)
    split_features = {
        "train": processor.transform(interactions=train),
        "val": processor.transform(interactions=val),
        "test": processor.transform(interactions=test),
    }

    if features.users is None or features.items is None:
        raise RuntimeError("User/item features were not processed.")

    user_frame = features.users
    item_frame = features.items
    if features.text_embeddings["users"] is not None:
        user_frame = pd.concat([user_frame, features.text_embeddings["users"]], axis=1)
    if features.text_embeddings["items"] is not None:
        item_frame = pd.concat([item_frame, features.text_embeddings["items"]], axis=1)

    split_dfs: dict[str, pd.DataFrame] = {}
    for name, processed in split_features.items():
        if processed.interactions is None:
            raise RuntimeError(f"{name} interactions were not processed.")
        split_dfs[name] = processed.interactions
        if processed.text_embeddings["inter"] is not None:
            split_dfs[name] = pd.concat(
                [split_dfs[name], processed.text_embeddings["inter"]],
                axis=1,
            )
        split_dfs[name] = split_dfs[name].reset_index(drop=True)

    static_feats = {}
    for name, df, prefix, id_col in (
        ("users", user_frame, "users", settings.USER_COL),
        ("items", item_frame, "items", settings.ITEM_COL),
    ):
        metadata = processor.feature_metadata[prefix]
        cols = (
            metadata.dense_cols
            + metadata.text_embedding_cols
            + metadata.categorical_cols
        )
        static_feats[name] = torch.as_tensor(
            df.sort_values(id_col)[cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

    return ProcessedData(
        train=split_dfs["train"],
        val=split_dfs["val"],
        test=split_dfs["test"],
        u_static_feats=static_feats["users"],
        i_static_feats=static_feats["items"],
        data_processor=processor,
    )
