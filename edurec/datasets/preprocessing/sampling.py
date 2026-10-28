"""Reproducible whole-user sampling and static concept graph reduction."""

import hashlib
from collections import Counter
from collections.abc import Mapping
from typing import Any

import pandas as pd

from edurec import settings

ACTIVITY_BOUNDS = (1, 4, 9, 19, 49, 100)


def _rank(value: str, seed: int) -> bytes:
    return hashlib.sha256(f"edurec-sampling-v1:{seed}:{value}".encode()).digest()


def _stratum(size: int) -> int:
    return next((bound for bound in ACTIVITY_BOUNDS if size <= bound), size)


def sample_users(
    histories: Mapping[str, int], max_interactions: int, seed: int
) -> tuple[set[str], dict[str, Any]]:
    """Select complete histories using proportional activity-stratum budgets.

    Each stratum receives the same target retention fraction of its events,
    approximating proportional user sampling without shortening histories.
    Whole-history rounding leaves some slack; a final hash-ordered pass fills
    it where possible. Very small budgets cannot represent every stratum.
    """
    if max_interactions < 1:
        raise ValueError("max_interactions must be positive.")
    if any(size < 1 for size in histories.values()):
        raise ValueError("History sizes must be positive.")
    total = sum(histories.values())
    selected: set[str] = set()
    groups: dict[int, list[str]] = {}
    for user, size in histories.items():
        groups.setdefault(_stratum(size), []).append(user)

    if total <= max_interactions:
        selected = set(histories)
    else:
        ranked = sorted(histories, key=lambda user: (_rank(user, seed), user))
        groups = {bound: [] for bound in groups}
        for user in ranked:
            groups[_stratum(histories[user])].append(user)
        used = 0
        for users in groups.values():
            quota = max_interactions * sum(histories[u] for u in users) // total
            group_used = 0
            for user in users:
                size = histories[user]
                if group_used + size <= quota:
                    selected.add(user)
                    group_used += size
            used += group_used
        for user in ranked:
            if user not in selected and used + histories[user] <= max_interactions:
                selected.add(user)
                used += histories[user]
        if not selected:
            raise ValueError(
                "max_interactions is too small to retain any complete user history."
            )

    return selected, {
        "selection": "whole_users_activity_stratified_sha256_v1",
        "seed": seed,
        "requested": max_interactions,
        "users_before": len(histories),
        "interactions_before": total,
        "users": len(selected),
        "interactions": sum(histories[u] for u in selected),
        "strata": {
            str(bound): {
                "users_before": len(users),
                "interactions_before": sum(histories[u] for u in users),
                "users": sum(u in selected for u in users),
                "interactions": sum(histories[u] for u in users if u in selected),
            }
            for bound, users in sorted(groups.items())
        },
    }


def reduce_concepts(
    items: pd.DataFrame, max_exclusive: int, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep shared concepts and cap exclusive concepts per retained course.

    Degrees use distinct courses in this static catalog, never interactions or
    held-out targets. Other item attributes and relation types are untouched.
    """
    if max_exclusive < 0:
        raise ValueError("max_exclusive must be nonnegative.")
    memberships = [set(values) for values in items["concept"]]
    degrees = Counter(concept for values in memberships for concept in values)
    reduced: list[list[str]] = []
    for course, values in zip(items[settings.ITEM_COL], memberships):
        shared = {value for value in values if degrees[value] > 1}
        exclusive = sorted(
            values - shared,
            key=lambda value: (_rank(f"{course}:{value}", seed), value),
        )[:max_exclusive]
        reduced.append(sorted(shared | set(exclusive)))
    result = items.copy()
    result["concept"] = reduced
    return result, {
        "selection": "shared_plus_capped_exclusive_sha256_v1",
        "seed": seed,
        "max_exclusive_per_course": max_exclusive,
        "concepts_before": len(degrees),
        "concepts": len({value for values in reduced for value in values}),
        "memberships_before": sum(map(len, memberships)),
        "memberships": sum(map(len, reduced)),
    }
