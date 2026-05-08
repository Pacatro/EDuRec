from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, TimeSeriesSplit

from .. import settings

DEFAULT_GENERAL_MODELS: tuple[str, ...] = ("BPR", "NeuMF", "LightGCN")
DEFAULT_SEQUENTIAL_MODELS: tuple[str, ...] = ("GRU4Rec", "NARM", "SASRec")


class CVStrategy(StrEnum):
    TEMPORAL = "temporal"
    GROUP = "group"


class InteractionMode(StrEnum):
    GENERAL = "general"
    SEQUENTIAL = "sequential"


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def has_temporal_feature(interactions: pd.DataFrame) -> bool:
    return build_temporal_series(interactions) is not None


def infer_cv_strategy(interactions: pd.DataFrame) -> CVStrategy:
    if has_temporal_feature(interactions):
        return CVStrategy.TEMPORAL
    return CVStrategy.GROUP


def infer_interaction_mode(interactions: pd.DataFrame) -> InteractionMode:
    if has_temporal_feature(interactions):
        return InteractionMode.SEQUENTIAL
    return InteractionMode.GENERAL


def default_recbole_models(mode: InteractionMode) -> tuple[str, ...]:
    if mode == InteractionMode.SEQUENTIAL:
        return DEFAULT_SEQUENTIAL_MODELS
    return DEFAULT_GENERAL_MODELS


def build_temporal_series(interactions: pd.DataFrame) -> pd.Series | None:
    if settings.TIME_COL not in interactions.columns:
        return None

    raw_values = interactions[settings.TIME_COL]
    numeric_values = pd.to_numeric(raw_values, errors="coerce")
    non_null_mask = raw_values.notna()

    if numeric_values[non_null_mask].notna().all():
        return numeric_values.astype(float)

    parsed_values = pd.to_datetime(raw_values, utc=True, errors="coerce")
    if not parsed_values.notna().any():
        return None

    temporal_values = pd.Series(np.nan, index=interactions.index, dtype=float)
    valid_mask = parsed_values.notna()
    temporal_values.loc[valid_mask] = (
        parsed_values.loc[valid_mask].astype("int64").astype(float) / 1_000_000_000.0
    )

    if temporal_values.isna().any():
        base_value = temporal_values.dropna().max()
        if np.isnan(base_value):
            base_value = 0.0
        missing_count = int(temporal_values.isna().sum())
        temporal_values.loc[temporal_values.isna()] = (
            base_value + np.arange(1, missing_count + 1, dtype=float)
        )

    return temporal_values


def sort_temporal_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    temporal_values = build_temporal_series(interactions)
    if temporal_values is None:
        raise ValueError("Temporal sorting requires a valid temporal feature.")

    working_df = interactions.copy()
    working_df["_cv_time"] = temporal_values
    working_df["_cv_order"] = np.arange(len(working_df), dtype=np.int64)
    ordered_df = working_df.sort_values(
        by=["_cv_time", "_cv_order"],
        kind="mergesort",
    ).reset_index(drop=True)

    return ordered_df.drop(columns=["_cv_time", "_cv_order"])


def generate_cv_folds(
    interactions: pd.DataFrame,
    n_splits: int,
    val_size: float,
    random_state: int | None = None,
    strategy: CVStrategy | None = None,
) -> list[FoldSplit]:
    if n_splits < 2:
        raise ValueError("Cross-validation requires at least 2 splits.")
    if not 0 < val_size < 1:
        raise ValueError("Validation size must be in the (0, 1) range.")

    chosen_strategy = strategy or infer_cv_strategy(interactions)

    if chosen_strategy == CVStrategy.TEMPORAL:
        return _generate_temporal_folds(
            interactions=interactions,
            n_splits=n_splits,
            val_size=val_size,
        )

    return _generate_group_folds(
        interactions=interactions,
        n_splits=n_splits,
        val_size=val_size,
        random_state=random_state,
    )


def _generate_temporal_folds(
    interactions: pd.DataFrame,
    n_splits: int,
    val_size: float,
) -> list[FoldSplit]:
    ordered_df = sort_temporal_interactions(interactions)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds: list[FoldSplit] = []

    for fold_idx, (train_valid_idx, test_idx) in enumerate(
        splitter.split(ordered_df),
        start=1,
    ):
        train_valid_df = ordered_df.iloc[train_valid_idx].reset_index(drop=True)
        test_df = ordered_df.iloc[test_idx].reset_index(drop=True)
        train_df, valid_df = _split_temporal_validation(train_valid_df, val_size)
        folds.append(
            FoldSplit(
                fold=fold_idx,
                train=train_df,
                valid=valid_df,
                test=test_df,
            )
        )

    return folds


def _split_temporal_validation(
    train_valid_df: pd.DataFrame,
    val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_valid = max(1, int(np.ceil(len(train_valid_df) * val_size)))
    if n_valid >= len(train_valid_df):
        raise ValueError("Temporal fold does not leave enough interactions for training.")

    train_df = train_valid_df.iloc[:-n_valid].reset_index(drop=True)
    valid_df = train_valid_df.iloc[-n_valid:].reset_index(drop=True)
    return train_df, valid_df


def _generate_group_folds(
    interactions: pd.DataFrame,
    n_splits: int,
    val_size: float,
    random_state: int | None,
) -> list[FoldSplit]:
    if settings.USER_COL not in interactions.columns:
        raise ValueError("Group CV requires a user identifier column.")

    groups = interactions[settings.USER_COL]
    unique_groups = int(groups.nunique())
    if unique_groups < n_splits:
        raise ValueError(
            f"Group CV needs at least {n_splits} unique users, found {unique_groups}."
        )

    splitter = GroupKFold(n_splits=n_splits)
    folds: list[FoldSplit] = []

    for fold_idx, (train_valid_idx, test_idx) in enumerate(
        splitter.split(interactions, groups=groups),
        start=1,
    ):
        train_valid_df = interactions.iloc[train_valid_idx].reset_index(drop=True)
        test_df = interactions.iloc[test_idx].reset_index(drop=True)
        train_df, valid_df = _split_group_validation(
            train_valid_df=train_valid_df,
            val_size=val_size,
            random_state=None if random_state is None else random_state + fold_idx,
        )
        folds.append(
            FoldSplit(
                fold=fold_idx,
                train=train_df,
                valid=valid_df,
                test=test_df,
            )
        )

    return folds


def _split_group_validation(
    train_valid_df: pd.DataFrame,
    val_size: float,
    random_state: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = train_valid_df[settings.USER_COL]
    unique_groups = int(groups.nunique())
    if unique_groups < 2:
        raise ValueError("Group validation split requires at least 2 unique users.")

    n_valid_groups = max(1, int(np.ceil(unique_groups * val_size)))
    if n_valid_groups >= unique_groups:
        n_valid_groups = 1

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=n_valid_groups,
        random_state=random_state,
    )
    train_idx, valid_idx = next(splitter.split(train_valid_df, groups=groups))

    train_df = train_valid_df.iloc[train_idx].reset_index(drop=True)
    valid_df = train_valid_df.iloc[valid_idx].reset_index(drop=True)
    return train_df, valid_df


def coerce_model_list(
    models: Sequence[str] | None,
    interactions: pd.DataFrame,
) -> list[str]:
    if models:
        return [model for model in models if model]
    return list(default_recbole_models(infer_interaction_mode(interactions)))


__all__ = [
    "CVStrategy",
    "FoldSplit",
    "InteractionMode",
    "build_temporal_series",
    "coerce_model_list",
    "default_recbole_models",
    "generate_cv_folds",
    "has_temporal_feature",
    "infer_cv_strategy",
    "infer_interaction_mode",
    "sort_temporal_interactions",
]
