"""Query views: interaction context, histories and negatives.

Everything here is derived from the processed splits. Histories are strictly
chronological: a row's own event is never part of its own history and only
strictly earlier events (including earlier splits) may appear, which prevents
target leakage. Negative sampling keeps the historical policy so existing
results stay comparable; alternative policies must be separate experiment
changes, not adapter side effects.
"""

import warnings
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer

from edurec import settings
from edurec.datasets.preprocessing.splits import SPLIT_ORDER
from edurec.datasets.preprocessing.tabular import (
    categorical_cardinalities,
    feature_columns,
)


@dataclass(frozen=True)
class QueryConfig:
    """Parameters that change the content of a query view.

    Instances are hashable so :class:`~edurec.datasets.adapters.base.ModelDataAdapter`
    can memoise views by content. ``random_state`` only affects negative
    sampling; it never changes splits or labels. ``use_histories=None`` derives
    the setting from the data: histories are only meaningful when a chronology
    exists.
    """

    max_history: int = settings.MAX_HISTORY_LEN
    negatives_per_positive: int = settings.TRAIN_NEGATIVES_PER_POSITIVE
    random_state: int | None = None
    use_histories: bool | None = None

    def __post_init__(self) -> None:
        if self.max_history < 0:
            raise ValueError("max_history must be non-negative.")
        if self.negatives_per_positive < 0:
            raise ValueError("negatives_per_positive must be non-negative.")


@dataclass
class InteractionFeatureTables:
    """Aligned interaction features for every event across all splits.

    ``dense`` and ``cat`` hold the flattened, preprocessor-encoded
    interaction features, ``timestamps`` holds the event times and
    ``dense_cols``/``cat_cardinalities`` describe the dense column names and
    the cardinality of each categorical column respectively. Row ``g`` of every
    array corresponds to the same global event, where global order is the
    concatenation of the splits in ``SPLIT_ORDER`` (``[train; val; test]``).
    """

    dense: np.ndarray
    cat: np.ndarray
    timestamps: np.ndarray
    dense_cols: list[str]
    cat_cardinalities: list[int]


def build_interaction_features(
    splits: dict[str, pd.DataFrame],
    transformer: ColumnTransformer | None,
) -> InteractionFeatureTables:
    """Build aligned dense/categorical/timestamp interaction tables.

    These tables are used ONLY as historical interaction context (past events),
    never for the target event, so no future/target information can leak into a
    prediction. The vocabulary and scaler are fitted on the training split by
    the tabular transformer; val/test are merely transformed, so they cannot leak
    into the fit.

    Splits are processed in the fixed order ``("train", "val", "test")`` and
    missing splits are skipped. The returned arrays are concatenated in that
    same order, so the global row order is exactly ``[train; val; test]``.

    Dense columns are, in order, the rating column when present followed by the
    processor's numeric then text-embedding columns. Categorical columns are the
    processor's categorical columns, with missing values encoded as ``-1``.
    Timestamps fall back to zeros when the temporal column is unavailable. A
    ``RuntimeError`` is raised when an available split is missing an expected
    feature column.
    """
    available = [
        (name, splits[name]) for name in SPLIT_ORDER if splits.get(name) is not None
    ]

    dense_cols: list[str] = []
    if available and settings.RATING_COL in available[0][1].columns:
        dense_cols.append(settings.RATING_COL)
    dense_cols.extend(feature_columns(transformer, "num"))
    if available:
        dense_cols.extend(available[0][1].filter(regex=r"^text__").columns)
    cat_cols = feature_columns(transformer, "cat")

    dense_parts: list[np.ndarray] = []
    cat_parts: list[np.ndarray] = []
    timestamp_parts: list[np.ndarray] = []

    for name, df in available:
        missing = [col for col in (*dense_cols, *cat_cols) if col not in df.columns]
        if missing:
            raise RuntimeError(
                f"Processed {name} split is missing expected interaction "
                f"feature columns: {', '.join(sorted(missing))}."
            )

        dense = (
            np.nan_to_num(df[dense_cols].to_numpy(dtype=np.float32), nan=0.0)
            if dense_cols
            else np.zeros((len(df), 0), dtype=np.float32)
        )
        cat = (
            df[cat_cols].fillna(-1).to_numpy(dtype=np.int64)
            if cat_cols
            else np.zeros((len(df), 0), dtype=np.int64)
        )
        timestamps = (
            np.nan_to_num(df[settings.TIME_COL].to_numpy(dtype=np.float64), nan=0.0)
            if settings.TIME_COL in df.columns
            else np.zeros((len(df),), dtype=np.float64)
        )

        dense_parts.append(dense)
        cat_parts.append(cat)
        timestamp_parts.append(timestamps)

    if dense_parts:
        dense = np.concatenate(dense_parts, axis=0)
        cat = np.concatenate(cat_parts, axis=0)
        timestamps = np.concatenate(timestamp_parts, axis=0)
    else:
        dense = np.zeros((0, len(dense_cols)), dtype=np.float32)
        cat = np.zeros((0, len(cat_cols)), dtype=np.int64)
        timestamps = np.zeros((0,), dtype=np.float64)

    return InteractionFeatureTables(
        dense=np.ascontiguousarray(dense, dtype=np.float32),
        cat=np.ascontiguousarray(cat, dtype=np.int64),
        timestamps=np.ascontiguousarray(timestamps, dtype=np.float64),
        dense_cols=dense_cols,
        cat_cardinalities=categorical_cardinalities(transformer),
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
