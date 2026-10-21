"""Temporal/random splits, relevance labels, negative sampling and histories."""

import warnings
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch

from edurec import settings
from edurec.datasets.pipeline.prepare import deduplicate_interactions

SPLIT_ORDER = ("train", "val", "test")


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
    if not 0 < test_ratio < 1 or not 0 < val_ratio < 1 or test_ratio + val_ratio >= 1:
        raise ValueError(
            "Validation/test ratios must be positive and sum to less than 1."
        )
    if min_interactions < 3:
        raise ValueError(
            "min_interactions must be at least 3 to create train/val/test splits."
        )
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
        for name in ("train", "val", "test")
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
    if (
        settings.RELEVANT_COL in interactions.columns
        and not interactions[settings.RELEVANT_COL].gt(0).all()
    ):
        raise ValueError("Negative sampling expects only positive interactions.")

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
    for user_id in interactions[settings.USER_COL].unique():
        unseen_items = items.difference(
            observed_by_user.get(user_id, set()), sort=False
        )
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


def build_histories(
    splits: dict[str, pd.DataFrame],
    max_history: int,
    enabled: bool = True,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build fixed-size, strictly chronological histories for every split.

    For each split the returned tuple is ``(items, valid_mask, context_index)``,
    each with shape ``[n, max_history]``:

    * ``items``: past item IDs shifted by one so zero remains padding.
    * ``valid_mask``: ``True`` where a real past event is stored.
    * ``context_index``: global interaction row index of each past event, or
      ``-1`` for padding. The global index is the row position in the
      concatenation of the splits in ``SPLIT_ORDER`` (they map to
      ``[0, len(train))``, then val rows, then test rows).

    The alignment is strict: ``items[r, t]``, ``valid_mask[r, t]`` and
    ``context_index[r, t]`` all refer to the SAME past event. A row's own event
    is never part of its own history, and only events strictly before the
    current row may appear, which prevents target leakage. Earlier splits feed
    into later ones: while iterating rows in order, each row first records the
    current per-user history and only then appends its own event.

    When ``enabled`` is ``False`` every split yields zero-width tensors of the
    appropriate dtypes instead of histories.
    """
    if max_history < 0:
        raise ValueError("max_history must be non-negative.")

    split_frames: list[tuple[str, pd.DataFrame]] = []
    for split in SPLIT_ORDER:
        df = splits.get(split)
        if df is None:
            raise RuntimeError(f"Processed split {split} is not available.")
        split_frames.append((split, df))

    histories: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    if not enabled:
        for split, df in split_frames:
            n_rows = len(df)
            histories[split] = (
                torch.zeros((n_rows, 0), dtype=torch.long),
                torch.zeros((n_rows, 0), dtype=torch.bool),
                torch.zeros((n_rows, 0), dtype=torch.long),
            )
        return histories

    global_items: list[int] = []
    for _, df in split_frames:
        global_items.extend(int(item) for item in df[settings.ITEM_COL].tolist())

    user_history: defaultdict[int, deque[int]] = defaultdict(
        lambda: deque(maxlen=max_history)
    )

    global_offset = 0
    for split, df in split_frames:
        n_rows = len(df)
        items = torch.zeros((n_rows, max_history), dtype=torch.long)
        valid_mask = torch.zeros((n_rows, max_history), dtype=torch.bool)
        context_index = torch.full((n_rows, max_history), -1, dtype=torch.long)

        users = df[settings.USER_COL].tolist()
        for row_idx in range(n_rows):
            past = user_history[int(users[row_idx])]
            if past:
                past_indices = list(past)
                count = len(past_indices)
                items[row_idx, :count] = torch.tensor(
                    [global_items[g] + 1 for g in past_indices], dtype=torch.long
                )
                valid_mask[row_idx, :count] = True
                context_index[row_idx, :count] = torch.tensor(
                    past_indices, dtype=torch.long
                )
            past.append(global_offset + row_idx)

        histories[split] = (items, valid_mask, context_index)
        global_offset += n_rows

    return histories
