import warnings

import numpy as np
import pandas as pd
import torch

from edurec import settings
from edurec.datasets.cache import ProcessedData
from edurec.datasets.dataprocessor import DataProcessor


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


def split_data(
    df: pd.DataFrame,
    test_ratio: float,
    val_ratio: float,
    min_interactions: int,
    random_state: int | None,
    deduplicate: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Preserve real, temporally distinct repeated interactions when a timestamp
    # is available. Without timestamps repeated pairs cannot be ordered, so they
    # are collapsed to avoid leakage across a random split.
    has_time = settings.TIME_COL in df.columns
    if deduplicate or not has_time:
        df = deduplicate_interactions(df)
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

    train_split = pd.concat(splits["train"], axis=0).reset_index(drop=True)
    val_split = pd.concat(splits["val"], axis=0).reset_index(drop=True)
    test_split = pd.concat(splits["test"], axis=0).reset_index(drop=True)

    return train_split, val_split, test_split


def filter_sparse(
    users: pd.DataFrame,
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    min_interactions: int,
    deduplicate: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove users and items with fewer than ``min_interactions``.

    Repeated ``(user, item)`` events are only collapsed when explicitly
    requested or when no timestamp can order them (see :func:`split_data`).
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


def generate_negative_samples(
    interactions: pd.DataFrame,
    item_ids: pd.Series | np.ndarray | list[int],
    num_negatives: int = 1,
    random_state: int | None = None,
    observed_interactions: pd.DataFrame | None = None,
) -> np.ndarray:
    """Precompute random negative item IDs aligned with positive interactions.

    Each output row corresponds to the interaction at the same input position.
    Samples are unique within a row and exclude every item observed by that
    user in ``observed_interactions`` (defaults to ``interactions``). Pass the
    union of all splits so held-out positives are never used as negatives.
    """
    required_cols = {settings.USER_COL, settings.ITEM_COL}
    missing_cols = required_cols.difference(interactions.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"Interactions are missing required columns: {missing}.")
    if num_negatives < 0:
        raise ValueError("num_negatives must be greater than or equal to zero.")

    negatives = np.empty((len(interactions), num_negatives), dtype=np.int64)
    if num_negatives == 0 or interactions.empty:
        return negatives

    items = pd.Index(pd.unique(np.asarray(item_ids, dtype=np.int64))).dropna()
    if len(items) < num_negatives:
        raise ValueError(
            f"At least {num_negatives} candidate items are required for sampling."
        )

    observed_source = (
        observed_interactions if observed_interactions is not None else interactions
    )
    observed_by_user = observed_source.groupby(settings.USER_COL, sort=False)[
        settings.ITEM_COL
    ].agg(set)
    rng = np.random.default_rng(random_state)
    candidates_by_user: dict[object, np.ndarray] = {}
    replace_by_user: dict[object, bool] = {}
    for user_id in observed_by_user.index:
        unseen_items = items.difference(observed_by_user[user_id], sort=False)
        if len(unseen_items) == 0:
            raise ValueError(
                f"User {user_id!r} has no unseen items to sample negatives from."
            )
        if len(unseen_items) < num_negatives:
            warnings.warn(
                f"User {user_id!r} has only {len(unseen_items)} unseen items; "
                f"sampling {num_negatives} negatives with replacement.",
                stacklevel=2,
            )
        candidates_by_user[user_id] = unseen_items.to_numpy(dtype=np.int64)
        replace_by_user[user_id] = len(unseen_items) < num_negatives

    for row_idx, user_id in enumerate(interactions[settings.USER_COL]):
        negatives[row_idx] = rng.choice(
            candidates_by_user[user_id],
            size=num_negatives,
            replace=replace_by_user[user_id],
        )

    return negatives


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
        cols = metadata.numeric_cols + metadata.text_embedding_cols
        static_feats[name] = torch.as_tensor(
            df.sort_values(id_col)[cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

    u_cat_feats = _categorical_codes(
        user_frame,
        processor.feature_metadata["users"].categorical_cols,
        id_col=settings.USER_COL,
    )

    return ProcessedData(
        train=split_dfs["train"],
        val=split_dfs["val"],
        test=split_dfs["test"],
        u_static_feats=static_feats["users"],
        u_cat_feats=u_cat_feats,
        i_static_feats=static_feats["items"],
        user_features=users.reset_index(drop=True),
        item_features=items.reset_index(drop=True),
        data_processor=processor,
    )


def _categorical_codes(
    frame: pd.DataFrame,
    categorical_cols: list[str],
    id_col: str,
) -> torch.Tensor:
    """Integer codes for the encoded categorical columns, ordered by mapped id.

    The processor already ordinal-encodes categorical fields, so the values are
    reused as-is. ``-1`` marks unknown categories and is preserved for the model
    to map to a padding embedding.
    """
    if not categorical_cols:
        return torch.zeros((len(frame), 0), dtype=torch.long)

    values = frame.sort_values(id_col)[categorical_cols].to_numpy(dtype=np.float32)
    values = np.nan_to_num(values, nan=-1.0)
    return torch.as_tensor(values.astype(np.int64), dtype=torch.long)
