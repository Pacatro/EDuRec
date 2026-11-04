"""Whole-user sampling after splitting and catalog-preserving concept pruning."""

from collections import Counter
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from edurec import settings
from edurec.datasets.config import DataConfig
from edurec.datasets.preprocessing.parsing import coerce_list_tokens
from edurec.datasets.sources.registry import RawData

# Inclusive upper bounds, measured exclusively on the training partition.
ACTIVITY_BOUNDS = (1, 4, 9, 19, 49, 100)


def sample_users(train: pd.DataFrame, config: DataConfig) -> set:
    """Select a stratified prefix using only training activity.

    Users are shuffled inside each activity stratum, then interleaved in
    proportion to stratum population. The training budget is the requested
    total times the nominal training fraction. Rounding in the original split
    means the final total is approximate; held-out lengths never drive selection.
    """
    counts = train.groupby(settings.USER_COL, sort=True).size()
    if config.max_interactions is None:
        return set(counts.index)
    budget = config.max_interactions * (1 - config.val_ratio - config.test_ratio)
    if budget >= int(counts.sum()):
        return set(counts.index)

    rng = np.random.default_rng(config.sampling_seed)
    strata = np.searchsorted(ACTIVITY_BOUNDS, counts.to_numpy(), side="left")
    candidates = []
    for stratum in np.unique(strata):
        group = counts.iloc[np.flatnonzero(strata == stratum)]
        group = group.iloc[rng.permutation(len(group))]
        candidates.append(
            pd.DataFrame(
                {
                    settings.USER_COL: group.index,
                    "cost": group.to_numpy(),
                    "priority": (np.arange(len(group)) + rng.random(len(group)))
                    / len(group),
                }
            )
        )
    if not candidates:
        raise ValueError("Cannot sample users from an empty training split.")
    ordered = pd.concat(candidates).sort_values("priority", kind="stable")
    selected = ordered.loc[ordered["cost"].cumsum() <= budget, settings.USER_COL]
    if selected.empty:
        raise ValueError(
            "max_interactions is too small to retain a complete user history; "
            "increase the budget."
        )
    return set(selected)


def prune_concepts(raw: RawData, limit: int, seed: int) -> tuple[RawData, dict]:
    """Preserve degree >=2 concepts and at most ``limit`` leaves per course.

    Degree is the number of distinct courses in the retained catalog, not
    interaction frequency. Hash ordering gives a reproducible, order-independent
    sample of exclusive concepts without pretending to rank pedagogical value.
    """
    if "concept" not in raw.item_features:
        return raw, {"before": 0, "after": 0, "edges_before": 0, "edges_after": 0}
    items = raw.item_features.copy()
    memberships = items["concept"].map(coerce_list_tokens)
    degrees = Counter(token for tokens in memberships for token in tokens)
    shared = {token for token, degree in degrees.items() if degree > 1}

    def reduce(tokens: list[str]) -> list[str]:
        exclusive = sorted(
            (token for token in tokens if token not in shared),
            key=lambda token: sha256(f"{seed}:{token}".encode()).digest(),
        )[:limit]
        retained = set(exclusive)
        return [token for token in tokens if token in shared or token in retained]

    items["concept"] = memberships.map(reduce)
    retained = {token for tokens in items["concept"] for token in tokens}
    entities = raw.graph_entities
    if entities is not None and "concept" in entities:
        entities = dict(entities)
        entities["concept"] = (
            entities["concept"]
            .loc[entities["concept"]["id"].isin(retained)]
            .reset_index(drop=True)
        )
    report = {
        "before": len(degrees),
        "after": len(retained),
        "shared_before": len(shared),
        "shared_after": len(shared & retained),
        "exclusive_after": len(retained - shared),
        "edges_before": sum(map(len, memberships)),
        "edges_after": sum(map(len, items["concept"])),
        "max_exclusive_per_course": limit,
    }
    return raw._replace(item_features=items, graph_entities=entities), report


def _coverage(train: pd.DataFrame, items: pd.DataFrame) -> dict[str, Any]:
    observed = items.loc[items[settings.ITEM_COL].isin(train[settings.ITEM_COL])]
    report: dict[str, Any] = {
        "courses_in_train": len(observed),
        "courses_without_train": len(items) - len(observed),
    }
    for column in ("field", "concept"):
        if column in items:
            catalog = {t for value in items[column] for t in coerce_list_tokens(value)}
            covered = {
                t for value in observed[column] for t in coerce_list_tokens(value)
            }
            report[column] = {
                "catalog": len(catalog),
                "covered_by_train": len(covered),
                "coverage": len(covered) / len(catalog) if catalog else 1.0,
            }
    return report


def reduce_dataset(
    raw: RawData,
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    config: DataConfig,
) -> tuple[RawData, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Reduce prepared data without resplitting histories or shrinking items."""
    if config.max_interactions is None and config.max_exclusive_concepts is None:
        return raw, splits
    train, val, test = splits
    report: dict[str, Any] = {"seed": config.sampling_seed}
    if config.max_interactions is not None:
        selected = sample_users(train, config)
        retained_splits = tuple(
            frame.loc[frame[settings.USER_COL].isin(selected)].reset_index(drop=True)
            for frame in splits
        )
        selected_train, val, test = retained_splits
        before_activity = train.groupby(settings.USER_COL).size()
        after_activity = selected_train.groupby(settings.USER_COL).size()
        report["sampling"] = {
            "requested_interactions": config.max_interactions,
            "budget_is_approximate": True,
            "users_before": len(before_activity),
            "users_after": len(selected),
            "interactions_before": sum(map(len, splits)),
            "interactions_after": sum(map(len, retained_splits)),
            "train_before": len(train),
            "train_after": len(selected_train),
            "activity_upper_bounds": list(ACTIVITY_BOUNDS),
            "strata_before": np.bincount(
                np.searchsorted(ACTIVITY_BOUNDS, before_activity),
                minlength=len(ACTIVITY_BOUNDS) + 1,
            ).tolist(),
            "strata_after": np.bincount(
                np.searchsorted(ACTIVITY_BOUNDS, after_activity),
                minlength=len(ACTIVITY_BOUNDS) + 1,
            ).tolist(),
            "coverage_before": _coverage(train, raw.item_features),
            "coverage_after": _coverage(selected_train, raw.item_features),
        }
        train = selected_train
        raw = raw._replace(
            user_features=raw.user_features.loc[
                raw.user_features[settings.USER_COL].isin(selected)
            ].reset_index(drop=True),
            interactions=raw.interactions.loc[
                raw.interactions[settings.USER_COL].isin(selected)
            ].reset_index(drop=True),
        )
    if config.max_exclusive_concepts is not None:
        raw, report["concepts"] = prune_concepts(
            raw,
            config.max_exclusive_concepts,
            config.sampling_seed,
        )
    raw = raw._replace(reports={**(raw.reports or {}), "reduction_report": report})
    return raw, (train, val, test)
