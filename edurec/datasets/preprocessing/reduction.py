"""Dataset reduction and graph pruning in one place.

The pipeline is intentionally split into two phases:

1. :func:`apply_common_filters` deduplicates interactions and removes sparse
   users/items before splitting.
2. :func:`reduce_dataset` samples whole users from TRAIN activity, restricts
   the splits and catalog, and prunes the item knowledge graph.

User sampling only reads the training partition, so held-out lengths and
targets never drive selection. Concept pruning only reads static course
metadata, so it cannot leak future information into the graph.
"""

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

type Splits = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase snake_case."""
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


def filter_sparse(
    users: pd.DataFrame,
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    min_interactions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove users and items with fewer than ``min_interactions``.

    Interactions are expected to be normalized already: repeated pairs are
    collapsed once before filtering.
    """
    filtered = interactions.copy()
    user_support = (
        int(interactions[settings.USER_COL].value_counts().max())
        if len(interactions)
        else 0
    )
    item_support = (
        int(interactions[settings.ITEM_COL].value_counts().max())
        if len(interactions)
        else 0
    )

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

    if filtered.empty:
        raise ValueError(
            f"Sparsity filtering removed every interaction: min_interactions="
            f"{min_interactions} but the densest user has {user_support} "
            f"interactions and the most popular item has {item_support}. "
            "Lower settings.MIN_INTERACTIONS or pass --no-remove-sparse."
        )

    valid_users = filtered[settings.USER_COL].unique()
    valid_items = filtered[settings.ITEM_COL].unique()

    return (
        users.loc[users[settings.USER_COL].isin(valid_users)].reset_index(drop=True),
        items.loc[items[settings.ITEM_COL].isin(valid_items)].reset_index(drop=True),
        filtered.reset_index(drop=True),
    )


def apply_common_filters(raw: RawData, config: DataConfig) -> RawData:
    """Clean, deduplicate and filter support; catalogs follow retained rows."""
    interactions = clean_cols(raw.interactions)
    users = clean_cols(raw.user_features)
    items = clean_cols(raw.item_features)

    if config.deduplicate or settings.TIME_COL not in interactions:
        interactions = deduplicate_interactions(interactions)

    if config.remove_sparse:
        users, items, interactions = filter_sparse(
            users,
            items,
            interactions,
            min_interactions=config.effective_min_interactions,
        )

    return raw._replace(
        interactions=interactions,
        user_features=users,
        item_features=items,
    )


def sample_users(train: pd.DataFrame, config: DataConfig) -> set:
    """Select a stratified prefix using only training activity.

    Users are shuffled inside each activity stratum, then interleaved in
    proportion to stratum population. The training budget is the requested
    total times the nominal training fraction. Rounding in the original split
    means the final total is approximate; held-out lengths never drive selection.
    """
    return select_users(train.groupby(settings.USER_COL, sort=True).size(), config)


def select_users(counts: pd.Series, config: DataConfig) -> set:
    """Select users from training counts without materializing their events."""
    counts = counts.sort_index()
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


def _restrict_splits(splits: Splits, selected: set) -> Splits:
    train, val, test = (
        frame.loc[frame[settings.USER_COL].isin(selected)].reset_index(drop=True)
        for frame in splits
    )
    return train, val, test


def _retain_users(raw: RawData, selected: set) -> RawData:
    """Keep only selected users in the user table and interaction log."""
    return raw._replace(
        user_features=raw.user_features.loc[
            raw.user_features[settings.USER_COL].isin(selected)
        ].reset_index(drop=True),
        interactions=raw.interactions.loc[
            raw.interactions[settings.USER_COL].isin(selected)
        ].reset_index(drop=True),
    )


def _retain_catalog(raw: RawData, splits: Splits) -> RawData:
    """Keep only items that appear in at least one retained split."""
    retained_items = pd.concat(splits, ignore_index=True)[settings.ITEM_COL]
    items = raw.item_features.loc[
        raw.item_features[settings.ITEM_COL].isin(retained_items)
    ].reset_index(drop=True)
    return raw._replace(item_features=items)


def _retain_entities(
    items: pd.DataFrame, entities: dict[str, pd.DataFrame] | None
) -> dict[str, pd.DataFrame] | None:
    """Discard metadata for attribute nodes absent from the reduced catalog."""
    if entities is None:
        return None
    result: dict[str, pd.DataFrame] = {}
    for field, frame in entities.items():
        if field not in items:
            continue
        retained = {t for value in items[field] for t in coerce_list_tokens(value)}
        result[field] = frame.loc[frame["id"].isin(retained)].reset_index(drop=True)
    return result


def _concept_pruning_active(config: DataConfig) -> bool:
    return (
        config.min_concept_support > 1
        or config.max_concept_support is not None
        or config.max_concepts_per_course is not None
    )


def _token_order(token: str, seed: int) -> bytes:
    """Deterministic tie-break for concepts with equal course support."""
    return sha256(f"{seed}:{token}".encode()).digest()


def _prune_concepts(items: pd.DataFrame, config: DataConfig) -> tuple[pd.DataFrame, dict]:
    """Apply global support filters and an optional per-course concept cap."""
    memberships = items["concept"].map(coerce_list_tokens)
    support = Counter(concept for concepts in memberships for concept in concepts)
    retained = {
        concept
        for concept, degree in support.items()
        if degree >= config.min_concept_support
        and (
            config.max_concept_support is None
            or degree <= config.max_concept_support
        )
    }

    def select(concepts: list[str]) -> list[str]:
        candidates = [concept for concept in concepts if concept in retained]
        limit = config.max_concepts_per_course
        if limit is None or len(candidates) <= limit:
            return candidates
        ranked = sorted(
            candidates,
            key=lambda concept: (
                -support[concept],
                _token_order(concept, config.sampling_seed),
            ),
        )
        selected = set(ranked[:limit])
        return [concept for concept in candidates if concept in selected]

    items = items.copy()
    items["concept"] = memberships.map(select)
    retained_after = {
        concept for concepts in items["concept"] for concept in concepts
    }
    return items, _concept_report(config, memberships, items, support, retained_after)


def _concept_report(
    config: DataConfig,
    memberships: pd.Series,
    items: pd.DataFrame,
    support: Counter[str],
    retained: set[str],
) -> dict[str, Any]:
    edges_after = sum(len(concepts) for concepts in items["concept"])
    return {
        "min_concept_support": config.min_concept_support,
        "max_concept_support": config.max_concept_support,
        "max_concepts_per_course": config.max_concepts_per_course,
        "seed": config.sampling_seed,
        "courses": len(items),
        "concepts_before": len(support),
        "concepts_after": len(retained),
        "edges_before": sum(support.values()),
        "edges_after": edges_after,
        "courses_without_concepts_before": sum(
            1 for concepts in memberships if not concepts
        ),
        "courses_without_concepts_after": sum(
            1 for concepts in items["concept"] if not concepts
        ),
    }


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


def _sampling_report(
    train: pd.DataFrame,
    original: Splits,
    retained: Splits,
    selected: set,
    items: pd.DataFrame,
    config: DataConfig,
) -> dict[str, Any]:
    selected_train = retained[0]
    before_activity = train.groupby(settings.USER_COL).size()
    after_activity = selected_train.groupby(settings.USER_COL).size()
    return {
        "requested_interactions": config.max_interactions,
        "budget_is_approximate": True,
        "users_before": len(before_activity),
        "users_after": len(selected),
        "interactions_before": sum(map(len, original)),
        "interactions_after": sum(map(len, retained)),
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
        "coverage_before": _coverage(train, items),
        "coverage_after": _coverage(selected_train, items),
    }


def _prune_graph(
    raw: RawData, config: DataConfig
) -> tuple[RawData, dict | None]:
    items = raw.item_features
    concept_report = None
    if _concept_pruning_active(config) and "concept" in items:
        items, concept_report = _prune_concepts(items, config)
    entities = _retain_entities(items, raw.graph_entities)
    return (
        raw._replace(item_features=items, graph_entities=entities),
        concept_report,
    )


def reduce_dataset(
    raw: RawData,
    splits: Splits,
    config: DataConfig,
) -> tuple[RawData, Splits]:
    """Reduce users, catalog and graph after the canonical split.

    Sampling uses only ``splits[0]`` (train). Sources with a streaming
    reduction path set ``RawData.source_reduced`` so their already-selected
    users are not sampled twice; graph pruning still applies.
    """
    sampling_active = config.max_interactions is not None and not raw.source_reduced
    pruning_active = _concept_pruning_active(config)
    if not sampling_active and not pruning_active:
        return raw, splits

    report = dict((raw.reports or {}).get("reduction_report", {}))
    report["seed"] = config.sampling_seed

    if sampling_active:
        train = splits[0]
        selected = sample_users(train, config)
        retained = _restrict_splits(splits, selected)
        report["sampling"] = _sampling_report(
            train, splits, retained, selected, raw.item_features, config
        )
        items_before = len(raw.item_features)
        raw = _retain_users(raw, selected)
        raw = _retain_catalog(raw, retained)
        report["catalog"] = {
            "items_before": items_before,
            "items_after": len(raw.item_features),
        }
        splits = retained

    raw, concept_report = _prune_graph(raw, config)
    if concept_report is not None:
        report["concepts"] = concept_report

    raw = raw._replace(
        reports={**(raw.reports or {}), "reduction_report": report}
    )
    return raw, splits
