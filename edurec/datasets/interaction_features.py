from dataclasses import dataclass

import numpy as np
import pandas as pd

from edurec import settings
from edurec.datasets.dataprocessor import DataProcessor

SPLIT_ORDER = ("train", "val", "test")


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
    processor: DataProcessor,
) -> InteractionFeatureTables:
    """Build aligned dense/categorical/timestamp interaction tables.

    These tables are used ONLY as historical interaction context (past events),
    never for the target event, so no future/target information can leak into a
    prediction. The vocabulary and scaler are fitted on the training split by
    the ``DataProcessor``; val/test are merely transformed, so they cannot leak
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
    metadata = processor.feature_metadata["inter"]
    available = [
        (name, splits[name]) for name in SPLIT_ORDER if splits.get(name) is not None
    ]

    dense_cols: list[str] = []
    if available and settings.RATING_COL in available[0][1].columns:
        dense_cols.append(settings.RATING_COL)
    dense_cols.extend(metadata.numeric_cols)
    dense_cols.extend(metadata.text_embedding_cols)
    cat_cols = list(metadata.categorical_cols)

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
            np.nan_to_num(df[cat_cols].to_numpy(dtype=np.float32), nan=-1.0).astype(
                np.int64
            )
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
        cat_cardinalities=[metadata.categorical_cardinalities[col] for col in cat_cols],
    )
