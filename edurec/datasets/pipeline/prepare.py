"""Source cleaning, eligibility filtering and community selection."""

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from edurec import settings
from edurec.datasets.features.parsing import coerce_list_tokens, single_token
from edurec.datasets.registry import RawData


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


def filter_sparse(
    users: pd.DataFrame,
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    min_interactions: int,
    deduplicate: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove users and items with fewer than ``min_interactions``.

    Repeated ``(user, item)`` events are only collapsed when explicitly
    requested or when no timestamp can order them (see ``split_data``).
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


def selection_metadata(
    max_users: int | None,
    seed: int,
    max_interactions: int | None = None,
    *,
    max_items: int | None = None,
    community_fields: list[str] | None = None,
    min_interactions: int = settings.MIN_INTERACTIONS,
    min_history_retention: float = 0.7,
) -> dict[str, Any]:
    """Community parameters shared by selection and cache identity."""
    for name, value in (
        ("max_users", max_users),
        ("max_interactions", max_interactions),
        ("max_items", max_items),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer or None.")
    if min_interactions < 1 or not 0 < min_history_retention <= 1:
        raise ValueError(
            "Require min_interactions >= 1 and history retention in (0, 1]."
        )
    return {
        "policy": "community",
        "max_users": max_users,
        "max_interactions": max_interactions,
        "max_items": max_items,
        "seed": seed,
        "community_fields": sorted(set(community_fields))
        if community_fields is not None
        else None,
        "min_interactions": min_interactions,
        "min_history_retention": min_history_retention,
    }


def _prune_community(
    events: pd.DataFrame, original_sizes: pd.Series, minimum: int, retention: float
) -> pd.DataFrame:
    """Iterate distinct-neighbor support and history retention to a fixed point."""
    user_col, item_col = settings.USER_COL, settings.ITEM_COL
    while not events.empty:
        previous_size = len(events)
        support = events.groupby(item_col)[user_col].nunique()
        events = events.loc[events[item_col].isin(support[support >= minimum].index)]
        support = events.groupby(user_col)[item_col].nunique()
        fraction = events.groupby(user_col).size() / original_sizes
        users = support[support >= minimum].index.intersection(
            fraction[fraction >= retention].index
        )
        events = events.loc[events[user_col].isin(users)]
        if len(events) == previous_size:
            break
    return events


def _stable_id(value: Any) -> str:
    return f"{type(value).__name__}:{value}"


def _item_links(raw: RawData, events: pd.DataFrame, fields: list[str]) -> dict:
    """Typed memberships connect items through semantics and shared users."""
    schema = raw.schema.get("items", {})
    lists = set(schema.get("list", [])) | set(schema.get("graph_list", []))
    links = defaultdict(set)
    for row in raw.item_features[[settings.ITEM_COL, *fields]].to_dict("records"):
        item = row[settings.ITEM_COL]
        for field in fields:
            parse = coerce_list_tokens if field in lists else single_token
            links[item].update(
                ("semantic", field, token) for token in parse(row[field])
            )
    for item, user in (
        events[[settings.ITEM_COL, settings.USER_COL]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        links[item].add(("user", "", _stable_id(user)))
    return dict(links)


def _connected_items(items: set, links: dict, start: Any) -> set:
    """Traverse memberships without constructing a dense item-item graph."""
    members = defaultdict(set)
    for item in items:
        for link in links.get(item, ()):
            members[link].add(item)
    reached = {start}
    stack = [start]
    while stack:
        for link in links.get(stack.pop(), ()):
            neighbors = members.pop(link, set()) - reached
            reached.update(neighbors)
            stack.extend(neighbors)
    return reached


def _grow_community(
    links: dict, support: pd.Series, limit: int | None, priority
) -> set:
    start = min(support.index, key=lambda item: (-support[item], priority(item)))
    if limit is None:
        return _connected_items(set(support.index), links, start)
    members = defaultdict(set)
    for item in support.index:
        for link in links.get(item, ()):
            members[link].add(item)
    weights = {
        link: math.log1p(len(support) / len(group)) for link, group in members.items()
    }
    chosen = set()
    scores = {start: 0.0}
    while scores and len(chosen) < limit:
        item = min(
            scores, key=lambda item: (-scores[item], -support[item], priority(item))
        )
        chosen.add(item)
        del scores[item]
        for link in sorted(links.get(item, ())):
            for neighbor in members[link] - chosen:
                scores[neighbor] = scores.get(neighbor, 0.0) + weights[link]
    return chosen


def select_dataset(
    raw: RawData,
    *,
    max_users: int | None = None,
    seed: int = 42,
    max_interactions: int | None = None,
    max_items: int | None = None,
    community_fields: list[str] | None = None,
    min_interactions: int = settings.MIN_INTERACTIONS,
    min_history_retention: float = 0.7,
) -> RawData:
    """Select one supported community using static semantics and shared users.

    Cohort selection precedes splitting and uses all eligible interactions, but
    never their ratings or timestamp values. Retention is relative to histories
    after the initial activity filter. No interaction edges are added to the KG.
    """
    metadata = selection_metadata(
        max_users,
        seed,
        max_interactions,
        max_items=max_items,
        community_fields=community_fields,
        min_interactions=min_interactions,
        min_history_retention=min_history_retention,
    )
    user_col, item_col = settings.USER_COL, settings.ITEM_COL
    events = _prune_community(
        raw.interactions, raw.interactions.groupby(user_col).size(), min_interactions, 0
    )
    if events.empty:
        raise ValueError("No users and items meet the minimum activity requirement.")
    original_sizes = events.groupby(user_col).size()
    schema = raw.schema.get("items", {})
    fields = metadata["community_fields"]
    if fields is None:
        fields = sorted(
            set(
                schema.get(
                    "community_fields",
                    [
                        *schema.get("cat", []),
                        *schema.get("list", []),
                        *schema.get("graph_list", []),
                    ],
                )
            )
        )
    links = _item_links(raw, events, fields)

    def priority(value):
        key = _stable_id(value)
        return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), key

    support = events.groupby(item_col)[user_col].nunique()
    catalog = _grow_community(links, support, max_items, priority)
    selected = _prune_community(
        events.loc[events[item_col].isin(catalog)],
        original_sizes,
        min_interactions,
        min_history_retention,
    )
    if max_users is not None or max_interactions is not None:
        users, total = [], 0
        sizes = selected.groupby(user_col).size()
        for user in sorted(sizes.index, key=priority):
            if max_users is not None and len(users) >= max_users:
                break
            size = int(sizes[user])
            if max_interactions is None or total + size <= max_interactions:
                users.append(user)
                total += size
        selected = selected.loc[selected[user_col].isin(users)]

    final_links = links
    # Removing users/items can remove bridges. Rebuild user links from the
    # surviving cohort, retain its main community and recheck all constraints.
    while not selected.empty:
        selected = _prune_community(
            selected, original_sizes, min_interactions, min_history_retention
        )
        if selected.empty:
            break
        final_links = _item_links(raw, selected, fields)
        support = selected.groupby(item_col)[user_col].nunique()
        start = min(support.index, key=lambda item: (-support[item], priority(item)))
        component = _connected_items(set(support.index), final_links, start)
        if len(component) == len(support):
            break
        selected = selected.loc[selected[item_col].isin(component)]
    if selected.empty:
        raise ValueError(
            "Community is empty after filtering; increase the budgets or lower support/retention thresholds."
        )

    selected = selected.reset_index(drop=True)
    users, items = selected[user_col].unique(), selected[item_col].unique()
    retention = selected.groupby(user_col).size() / original_sizes.reindex(users)
    semantic_support = Counter(
        link for item in items for link in final_links[item] if link[0] == "semantic"
    )
    report = {
        "policy": "community",
        "config": metadata,
        "community_fields": fields,
        "eligible_users": int(events[user_col].nunique()),
        "eligible_items": int(events[item_col].nunique()),
        "eligible_interactions": len(events),
        "selected_users": len(users),
        "selected_items": len(items),
        "selected_interactions": len(selected),
        "connected": True,
        "min_user_items": int(selected.groupby(user_col)[item_col].nunique().min()),
        "min_item_users": int(support.min()),
        "mean_history_retention": float(retention.mean()),
        "min_history_retention": float(retention.min()),
        "shared_semantic_entities": sum(
            count > 1 for count in semantic_support.values()
        ),
        "source_report": raw.selection_report,
    }
    return raw._replace(
        interactions=selected,
        user_features=raw.user_features.loc[
            raw.user_features[user_col].isin(users)
        ].reset_index(drop=True),
        item_features=raw.item_features.loc[
            raw.item_features[item_col].isin(items)
        ].reset_index(drop=True),
        selection_report=report,
    )
