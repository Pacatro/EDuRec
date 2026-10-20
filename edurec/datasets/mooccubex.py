"""Reproducible course communities and complete enrollment histories.

Only static course/concept links guide graph selection. Enrollment data guide
cohort coverage; no user edges are added to the recommendation graph. The
selection is a dataset-definition step before the chronological split, never a
search over validation/test recommendation scores.
"""

import hashlib
import heapq
import json
import math
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from edurec import settings


@dataclass(frozen=True)
class MooccubexSelection:
    """Budgets count all splits together; None means unlimited.

    Grow a course community through shared concepts, then select whole users.
    A null course limit includes all eligible courses.
    The selection seed is independent of model initialization seeds.
    """

    max_interactions: int | None = settings.MOOCCUBEX_MAX_INTERACTIONS
    max_courses: int | None = 100
    min_user_courses: int = 5
    min_course_users: int = 5
    min_history_retention: float = 0.7
    fields: tuple[str, ...] = ()
    seed_course: str | None = None
    seed: int = 42
    candidate_multiplier: int = 10
    encode_concept_text: bool = True
    min_birth_year: int = 1900
    max_birth_year: int = 2010

    def __post_init__(self) -> None:
        for name in ("max_interactions", "max_courses"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or null.")
        for name, minimum in (
            ("min_user_courses", 3),
            ("min_course_users", 1),
            ("candidate_multiplier", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in ("seed", "min_birth_year", "max_birth_year"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if not isinstance(self.fields, (list, tuple)) or not all(
            isinstance(value, str) for value in self.fields
        ):
            raise TypeError("fields must be a sequence of field labels.")
        object.__setattr__(self, "fields", tuple(sorted(set(self.fields))))
        if isinstance(self.min_history_retention, bool) or not isinstance(
            self.min_history_retention, (int, float)
        ):
            raise TypeError("min_history_retention must be numeric.")
        if not 0 < self.min_history_retention <= 1:
            raise ValueError("min_history_retention must be in (0, 1].")
        if not isinstance(self.encode_concept_text, bool):
            raise TypeError("encode_concept_text must be a boolean.")
        if self.min_birth_year > self.max_birth_year:
            raise ValueError("Invalid birth year bounds.")

    def metadata(self) -> dict[str, Any]:
        result = asdict(self)
        result["fields"] = list(self.fields)
        return result

    @classmethod
    def from_yaml(cls, path: Path) -> "MooccubexSelection":
        with path.open(encoding="utf-8") as source:
            values = yaml.safe_load(source)
        if not isinstance(values, dict):
            raise TypeError("MOOCCubeX selection YAML must contain a mapping.")
        values = dict(values)
        if "strategy" in values:
            raise ValueError(
                "MOOCCubeX selection always uses the community strategy; "
                "remove the 'strategy' field from the selection YAML."
            )
        fields = values.get("fields", [])
        if not isinstance(fields, list) or not all(
            isinstance(value, str) for value in fields
        ):
            raise ValueError("fields must be a YAML list of field labels.")
        values["fields"] = tuple(fields)
        return cls(**values)


def selection_config() -> MooccubexSelection:
    path = settings.MOOCCUBEX_CONFIG_PATH
    return (
        MooccubexSelection.from_yaml(path) if path is not None else MooccubexSelection()
    )


def _timestamp(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value.strip())
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return int(stamp.timestamp())
    except (ValueError, OverflowError, OSError):
        return None


def _enrollments(
    record: dict[str, Any], catalog: set[str], counts: Counter
) -> list[tuple[str, int]]:
    courses, times = record.get("course_order"), record.get("enroll_time")
    if (
        not isinstance(courses, list)
        or not isinstance(times, list)
        or len(courses) != len(times)
    ):
        counts["unaligned_users"] += 1
        return []
    # Enrollment duplicates are not repeat learning events. Keep the first
    # enrollment so that re-registration does not move a course into the future.
    first: dict[str, int] = {}
    for course, value in zip(courses, times):
        if course is None:
            counts["invalid_events"] += 1
            continue
        course_id = str(course)
        if not course_id.startswith("C_"):
            course_id = f"C_{course_id}"
        stamp = _timestamp(value)
        if course_id not in catalog or stamp is None:
            counts["invalid_events"] += 1
            continue
        if course_id in first:
            counts["duplicate_enrollments"] += 1
        first[course_id] = min(first.get(course_id, stamp), stamp)
    # Stable course-ID tie break is explicit; equal times do not establish a
    # pedagogical order and are reported in the selection diagnostics.
    return sorted(first.items(), key=lambda event: (event[1], event[0]))


def _course_community(
    items: pd.DataFrame,
    popularity: Counter,
    cfg: MooccubexSelection,
) -> set[str]:
    eligible = {
        str(course)
        for course in items[settings.ITEM_COL]
        if popularity[course] >= cfg.min_course_users
    }
    if not eligible:
        return set()
    if cfg.max_courses is None:
        if cfg.seed_course is not None and cfg.seed_course not in eligible:
            raise ValueError("seed_course is missing or has insufficient user support.")
        return eligible

    concepts = {
        str(row[settings.ITEM_COL]): set(row["concept"])
        for row in items.to_dict("records")
        if row[settings.ITEM_COL] in eligible
    }
    members: dict[str, set[str]] = defaultdict(set)
    for course, tokens in concepts.items():
        for token in tokens:
            members[token].add(course)
    # Inverse-frequency weighting stops ubiquitous concepts from dominating.
    weights = {
        token: math.log1p(len(eligible) / len(courses))
        for token, courses in members.items()
        if len(courses) > 1
    }
    seed = cfg.seed_course or min(
        eligible, key=lambda course: (-popularity[course], course)
    )
    if seed not in eligible:
        raise ValueError("seed_course is missing or has insufficient user support.")
    selected: set[str] = set()
    scores: dict[str, float] = defaultdict(float)
    current = seed
    while len(selected) < cfg.max_courses:
        selected.add(current)
        for token in sorted(concepts[current]):
            if token not in weights:
                continue
            for neighbor in sorted(members[token]):
                if neighbor not in selected:
                    scores[neighbor] += weights[token]
        scores.pop(current, None)
        if not scores:
            break
        current = min(
            scores, key=lambda course: (-scores[course], -popularity[course], course)
        )
    return selected


def _supported_core(
    events: pd.DataFrame,
    original_sizes: dict[str, int],
    cfg: MooccubexSelection,
) -> pd.DataFrame:
    """Iteratively enforce distinct-user/course support and history retention."""
    while not events.empty:
        size = len(events)
        item_counts = events.groupby(settings.ITEM_COL)[settings.USER_COL].nunique()
        events = events.loc[
            events[settings.ITEM_COL].isin(
                item_counts[item_counts >= cfg.min_course_users].index
            )
        ]
        user_counts = events.groupby(settings.USER_COL).size()
        valid = [
            user
            for user, count in user_counts.items()
            if count >= cfg.min_user_courses
            and count / original_sizes[str(user)] >= cfg.min_history_retention
        ]
        events = events.loc[events[settings.USER_COL].isin(valid)]
        if len(events) == size:
            break
    return events.reset_index(drop=True)


def _eligible_users(
    user_path: Path,
    catalog: set[str],
    field_catalog: set[str],
    cfg: MooccubexSelection,
    counts: Counter,
) -> Iterator[tuple[dict[str, Any], list[tuple[str, int]]]]:
    """Stream cleaned users, retaining only eligible IDs for deduplication."""
    seen: set[str] = set()
    with user_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            counts["scanned_users"] += 1
            history = _enrollments(record, catalog, counts)
            if len(history) < cfg.min_user_courses or not record.get("id"):
                continue
            retained = sum(course in field_catalog for course, _ in history)
            if (
                retained < cfg.min_user_courses
                or retained / len(history) < cfg.min_history_retention
            ):
                continue
            user = str(record["id"])
            if user in seen:
                counts["duplicate_user_records"] += 1
                continue
            seen.add(user)
            counts["eligible_source_users"] += 1
            yield record, history


@dataclass
class _Candidate:
    priority: tuple[str, str]
    profile: dict[str, Any]
    history: list[tuple[str, int]]
    original_size: int

    def __lt__(self, other: "_Candidate") -> bool:
        # heapq pops the greatest priority first, so the worst candidate can
        # be evicted without sorting or retaining every eligible history.
        return self.priority > other.priority


def _collect_candidates(
    user_path: Path,
    catalog: set[str],
    field_catalog: set[str],
    chosen: set[str],
    cfg: MooccubexSelection,
    budget: int | None,
    counts: Counter,
) -> list[_Candidate]:
    """Keep the same weighted priority prefix as sorting all eligible users.

    The first user overflowing the event capacity marks a permanent cutoff.
    This prevents later lower-priority, shorter histories from filling holes
    that a sorted prefix would leave empty. The heap holds at most capacity
    enrollment rows between records, plus the one current source history.
    """
    capacity = None if budget is None else budget * cfg.candidate_multiplier
    candidates: list[_Candidate] = []
    used = 0
    cutoff: tuple[str, str] | None = None
    # Cleaning diagnostics count source records once, during the first pass.
    for record, history in _eligible_users(
        user_path, catalog, field_catalog, cfg, Counter()
    ):
        retained = [(course, stamp) for course, stamp in history if course in chosen]
        if (
            len(retained) < cfg.min_user_courses
            or len(retained) / len(history) < cfg.min_history_retention
        ):
            continue
        if budget is not None and len(retained) > budget:
            counts["users_exceeding_budget"] += 1
            continue
        user = str(record["id"])
        priority = (hashlib.sha256(f"{cfg.seed}:{user}".encode()).hexdigest(), user)
        if cutoff is not None and priority >= cutoff:
            continue
        candidate = _Candidate(
            priority,
            {name: record.get(name) for name in ("gender", "school", "year_of_birth")},
            retained,
            len(history),
        )
        if capacity is None:
            candidates.append(candidate)
            continue
        heapq.heappush(candidates, candidate)
        used += len(retained)
        while used > capacity:
            removed = heapq.heappop(candidates)
            used -= len(removed.history)
            cutoff = (
                removed.priority if cutoff is None else min(cutoff, removed.priority)
            )
    return sorted(candidates, key=lambda candidate: candidate.priority)


def _field_catalog(
    items: pd.DataFrame, catalog: set[str], fields: tuple[str, ...]
) -> set[str]:
    """Restrict courses to any requested field, preserving raw field labels."""
    if not fields:
        return catalog
    requested = set(fields)
    matches = items["field"].map(
        lambda value: isinstance(value, (list, tuple))
        and bool(requested.intersection(value))
    )
    selected = set(items.loc[matches, settings.ITEM_COL])
    if not selected:
        raise ValueError(f"No MOOCCubeX courses match fields {fields}.")
    return selected


def _apply_budget(
    events: pd.DataFrame,
    original_sizes: dict[str, int],
    cfg: MooccubexSelection,
) -> pd.DataFrame:
    """Fit whole supported histories into the budget, then recheck support."""
    budget = cfg.max_interactions
    events = _supported_core(events, original_sizes, cfg)
    if budget is not None:
        sizes = events.groupby(settings.USER_COL, sort=False).size()
        accepted_users: list[str] = []
        used = 0
        for user, size in sizes.items():
            if used + size <= budget:
                accepted_users.append(str(user))
                used += size
        events = _supported_core(
            events.loc[events[settings.USER_COL].isin(accepted_users)],
            original_sizes,
            cfg,
        )

    return events


def _selection_report(
    events: pd.DataFrame,
    users: pd.DataFrame,
    original_sizes: dict[str, int],
    counts: Counter,
    user_path: Path,
    budget: int | None,
    *,
    catalog_size: int,
    candidate_users: int,
    candidate_interactions: int,
) -> dict[str, Any]:
    """Describe selection coverage, source provenance, and history retention."""
    counts["selected_users"] = len(users)
    counts["selected_courses"] = events[settings.ITEM_COL].nunique()
    counts["selected_interactions"] = len(events)
    counts["missing_or_invalid_birth_years"] = int(users["year_of_birth"].isna().sum())
    counts["tied_timestamps"] = int(
        events.duplicated([settings.USER_COL, settings.TIME_COL]).sum()
    )
    report: dict[str, Any] = dict(counts)
    report["selected_catalog_size_before_core"] = catalog_size
    report["candidate_pool_users"] = candidate_users
    report["candidate_pool_interactions"] = candidate_interactions
    report["budget_utilization"] = len(events) / budget if budget else None
    report["source_user_file_bytes"] = user_path.stat().st_size
    report["source_user_file_mtime_ns"] = user_path.stat().st_mtime_ns
    for name, sizes in (
        ("user_courses", events.groupby(settings.USER_COL).size()),
        ("course_users", events.groupby(settings.ITEM_COL).size()),
    ):
        report[name] = {
            "min": int(sizes.min()),
            "median": float(sizes.median()),
            "max": int(sizes.max()),
        }
    report["mean_history_retention"] = float(
        (
            events.groupby(settings.USER_COL).size()
            / pd.Series(original_sizes).reindex(users[settings.USER_COL])
        ).mean()
    )
    return report


def select_enrollments(
    user_path: Path,
    items: pd.DataFrame,
    cfg: MooccubexSelection,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select in two JSON passes without temporary storage.

    First count course support and choose the catalog, then keep a bounded
    priority heap of whole user histories. Only eligible user IDs are retained
    across source records for exact deduplication; their memory grows with the
    source population. Candidate histories are bounded by the event capacity in
    bounded mode; a null interaction budget retains all qualifying histories.
    """
    catalog = set(items[settings.ITEM_COL].astype(str))
    field_catalog = _field_catalog(items, catalog, cfg.fields)
    counts: Counter = Counter()
    popularity: Counter = Counter()
    budget = cfg.max_interactions
    for _, history in _eligible_users(user_path, catalog, field_catalog, cfg, counts):
        popularity.update(course for course, _ in history if course in field_catalog)
    chosen = _course_community(
        items.loc[items[settings.ITEM_COL].isin(field_catalog)],
        popularity,
        cfg,
    )
    if not chosen:
        raise ValueError("MOOCCubeX has no courses meeting the selection thresholds.")
    candidates = _collect_candidates(
        user_path, catalog, field_catalog, chosen, cfg, budget, counts
    )
    profiles = {candidate.priority[1]: candidate.profile for candidate in candidates}
    original_sizes = {
        candidate.priority[1]: candidate.original_size for candidate in candidates
    }
    rows = [
        (candidate.priority[1], course, stamp)
        for candidate in candidates
        for course, stamp in candidate.history
    ]
    del candidates
    events = pd.DataFrame(
        rows, columns=[settings.USER_COL, settings.ITEM_COL, settings.TIME_COL]
    )
    events = _apply_budget(events, original_sizes, cfg)

    if events.empty:
        raise ValueError(
            "MOOCCubeX selection is empty after support/retention filtering. Increase the budget, widen the catalog, or lower explicit thresholds."
        )
    events = events.sort_values(
        [settings.USER_COL, settings.TIME_COL, settings.ITEM_COL], kind="stable"
    ).reset_index(drop=True)
    user_ids = sorted(events[settings.USER_COL].unique())
    users = pd.DataFrame(
        [{settings.USER_COL: user, **profiles[user]} for user in user_ids]
    )
    birth = pd.to_numeric(users["year_of_birth"], errors="coerce")
    users["year_of_birth"] = birth.where(
        birth.between(cfg.min_birth_year, cfg.max_birth_year)
    )
    report = _selection_report(
        events,
        users,
        original_sizes,
        counts,
        user_path,
        budget,
        catalog_size=len(chosen),
        candidate_users=len(profiles),
        candidate_interactions=len(rows),
    )
    if budget is not None and len(events) < budget / 2:
        warnings.warn(
            f"MOOCCubeX retained {len(events)} of the {budget} event budget after filtering; inspect selection diagnostics or widen the catalog."
        )
    return events, users, report
