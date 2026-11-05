import json
import re
import warnings
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from edurec import settings
from edurec.datasets.preprocessing.reduction import select_users
from edurec.datasets.preprocessing.splits import split_sizes
from edurec.datasets.sources.common import ensure_files, stream_to_file
from edurec.datasets.sources.registry import DatasetName, DatasetSpec, RawData

if TYPE_CHECKING:
    from edurec.datasets.config import DataConfig

FOLDER = DatasetName.MOOCCUBEX.value

_COURSE_FIELDS = (
    settings.ITEM_COL,
    "name",
    "field",
    "prerequisites",
    "about",
    "resource",
)

_EXPERIMENTAL_COURSE = re.compile(
    settings.MOOCCUBEX_EXPERIMENTAL_COURSE_PATTERN,
    flags=re.IGNORECASE,
)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield nonempty JSON objects without loading the file into memory."""
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def _has_educational_resource(resource: object) -> bool:
    """Whether a course exposes at least one video or exercise resource."""
    if not isinstance(resource, list):
        return False
    prefixes = settings.MOOCCUBEX_EDUCATIONAL_RESOURCE_PREFIXES
    return any(
        isinstance(item, dict) and str(item.get("resource_id", "")).startswith(prefixes)
        for item in resource
    )


def _is_experimental_course(name: object) -> bool:
    """Detect placeholder course names without dropping real testing courses."""
    if not isinstance(name, str):
        return False
    if settings.MOOCCUBEX_EXCLUDE_DEPRECATED_COURSES and "废弃" in name:
        return True
    return bool(_EXPERIMENTAL_COURSE.search(name))


def _fetch_mooccubex(dest: Path) -> None:
    for relative_path in settings.MOOCCUBEX_REQUIRED_FILES:
        url = f"{settings.MOOCCUBEX_BASE_URL}/{relative_path}"
        destination = dest / relative_path
        if destination.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            stream_to_file(url, temporary, desc=relative_path)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def download_mooccubex() -> Path:
    dest = settings.RAW_DATA_FOLDER / FOLDER
    return ensure_files(
        dest,
        settings.MOOCCUBEX_REQUIRED_FILES,
        lambda: _fetch_mooccubex(dest),
    )


def _mooccubex_timestamp(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value.strip())
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return int(stamp.timestamp())
    except (ValueError, OverflowError, OSError):
        return None


def _mooccubex_enrollments(
    record: dict[str, Any], catalog: set[str]
) -> list[tuple[str, int]]:
    courses, times = record.get("course_order"), record.get("enroll_time")
    if (
        not isinstance(courses, list)
        or not isinstance(times, list)
        or len(courses) != len(times)
    ):
        return []
    # Enrollment duplicates are not repeat learning events. Keep the first
    # enrollment so that re-registration does not move a course into the future.
    first: dict[str, int] = {}
    for course, value in zip(courses, times):
        if course is None:
            continue
        course_id = str(course)
        if not course_id.startswith("C_"):
            course_id = f"C_{course_id}"
        stamp = _mooccubex_timestamp(value)
        if course_id not in catalog or stamp is None:
            continue
        first[course_id] = min(first.get(course_id, stamp), stamp)
    # Stable course-ID tie break is explicit; equal times do not establish a
    # pedagogical order.
    return sorted(first.items(), key=lambda event: (event[1], event[0]))


def _iter_mooccubex_users(
    user_path: Path,
    catalog: set[str],
    selected: set[str] | None = None,
) -> Iterator[tuple[str, dict[str, Any], list[tuple[str, int]]]]:
    """Stream normalized histories that pass the source-level filters.

    The user file has no index, so every pass scans it. Pass ``selected`` to
    avoid normalizing histories that will not be materialized.
    """
    seen: set[str] = set()
    for record in _iter_jsonl(user_path):
        if not record.get("id"):
            continue
        user = str(record["id"])
        if user in seen:
            continue
        seen.add(user)
        if selected is not None and user not in selected:
            continue
        history = _mooccubex_enrollments(record, catalog)
        if not history or len(history) > settings.MOOCCUBEX_MAX_USER_ENROLLMENTS:
            continue
        yield user, record, history


def _load_mooccubex_courses(path: Path) -> pd.DataFrame:
    """Stream courses, retaining only educational, non-placeholder metadata."""
    records: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for record in _iter_jsonl(path):
        if not record.get("id"):
            continue
        course = str(record["id"])
        # Match the old table semantics: the first occurrence owns the ID even
        # when it is later rejected by a course-level filter.
        if course in seen:
            continue
        seen.add(course)
        if not _has_educational_resource(record.get("resource")):
            continue
        if _is_experimental_course(record.get("name")):
            continue
        records[course] = {
            field: course if field == settings.ITEM_COL else record.get(field)
            for field in _COURSE_FIELDS
        }
    if not records:
        raise ValueError(
            "MOOCCubeX has no courses with educational resources after filtering."
        )

    return pd.DataFrame.from_records(list(records.values()), columns=_COURSE_FIELDS)


def _summarize_mooccubex_users(
    path: Path, catalog: set[str], config: "DataConfig"
) -> tuple[set[str], set[str], dict]:
    """Select users from streaming counts and find the source k-core.

    This is the first user-file pass: only identifiers and per-user training
    counts are retained, plus per-course support needed by the k-core. The
    catalog is rechecked until it stops shrinking; each extra pass still keeps
    summaries rather than enrollment histories.

    Histories are normalized against the original catalog on every pass so the
    extreme-enrollment cap never changes when sparse courses disappear.
    """
    retained = catalog.copy()
    while True:
        item_counts: Counter[str] = Counter()
        counts: dict[str, int] = {}
        for user, _, history in _iter_mooccubex_users(path, catalog):
            courses = [course for course, _ in history if course in retained]
            n = len(courses)
            if not n or (
                config.remove_sparse and n < config.effective_min_interactions
            ):
                continue
            item_counts.update(courses)
            # Below the split minimum the whole history becomes train-only.
            counts[user] = (
                n
                if n < config.effective_min_interactions
                else split_sizes(n, config.val_ratio, config.test_ratio)[0]
            )
        next_catalog = {
            course
            for course, count in item_counts.items()
            if not config.remove_sparse or count >= config.effective_min_interactions
        }
        if not next_catalog:
            raise ValueError(
                "Sparsity filtering removed every interaction; lower "
                "min_interactions or pass --no-remove-sparse."
            )
        if next_catalog == retained:
            break
        retained = next_catalog

    selected = select_users(pd.Series(counts, dtype="int64"), config)
    report = {
        "seed": config.sampling_seed,
        "sampling": {
            "requested_interactions": config.max_interactions,
            "budget_is_approximate": True,
            "users_before": len(counts),
            "users_after": len(selected),
            "interactions_before": sum(item_counts.values()),
            "train_before": sum(counts.values()),
            "train_after": sum(counts[user] for user in selected),
        },
        "catalog": {"items_before": len(retained)},
    }
    return selected, retained, report


def _load_mooccubex_enrollments(
    user_path: Path,
    catalog: set[str],
    selected: set[str] | None = None,
    retained_catalog: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Second user pass: materialize only selected complete histories."""
    rows: list[tuple[str, str, int]] = []
    profiles: list[dict[str, Any]] = []
    for user, record, history in _iter_mooccubex_users(
        user_path, catalog, selected=selected
    ):
        if retained_catalog is not None:
            history = [
                (course, stamp)
                for course, stamp in history
                if course in retained_catalog
            ]
            if not history:
                continue
        rows.extend((user, course, stamp) for course, stamp in history)
        profiles.append(
            {
                settings.USER_COL: user,
                **{
                    name: record.get(name)
                    for name in ("gender", "school", "year_of_birth")
                },
            }
        )
    events = pd.DataFrame(
        rows, columns=[settings.USER_COL, settings.ITEM_COL, settings.TIME_COL]
    )
    users = pd.DataFrame(
        profiles, columns=[settings.USER_COL, "gender", "school", "year_of_birth"]
    )
    birth = pd.to_numeric(users["year_of_birth"], errors="coerce")
    users["year_of_birth"] = birth.where(birth.between(1900, 2010))
    return events, users


def load_mooccubex(config: "DataConfig | None" = None) -> RawData:
    """Load valid enrollments after course and user source-level filtering."""
    if config is not None and config.dataset is not DatasetName.MOOCCUBEX:
        raise ValueError("Expected MOOCCubeX source configuration.")
    if settings.MOOCCUBEX_MAX_USER_ENROLLMENTS < 1:
        raise ValueError("MOOCCUBEX_MAX_USER_ENROLLMENTS must be positive.")
    entities_folder = settings.RAW_DATA_FOLDER / FOLDER / "entities"
    items = _load_mooccubex_courses(entities_folder / "course.json")

    relations_folder = entities_folder.parent / "relations"
    catalog = set(items[settings.ITEM_COL].astype(str))
    user_path = entities_folder / "user.json"
    report = None
    selected = retained = None
    if config is not None:
        selected, retained, report = _summarize_mooccubex_users(
            user_path, catalog, config
        )
    interactions, users = _load_mooccubex_enrollments(
        user_path,
        catalog,
        selected=selected,
        retained_catalog=retained,
    )

    items = items.loc[
        items[settings.ITEM_COL].isin(interactions[settings.ITEM_COL])
    ].copy()
    if items.empty or interactions.empty:
        raise ValueError(
            "MOOCCubeX has no interactions after source filtering; relax the "
            "enrollment cap or the course filters."
        )
    if report is not None:
        report["catalog"]["items_after"] = len(items)
        report["sampling"]["interactions_after"] = len(interactions)
    items["resource_count"] = items["resource"].map(len)
    items = items.drop(columns=["resource"])
    # Load static memberships only for the final retained catalog.
    for field, filename, course_column in (
        ("concept", "concept-course.txt", 1),
        ("teacher", "course-teacher.txt", 0),
        ("school", "course-school.txt", 0),
    ):
        items[field] = _load_course_relations(
            relations_folder / filename,
            items[settings.ITEM_COL],
            course_column=course_column,
        )
    concept_ids = {concept for concepts in items["concept"] for concept in concepts}
    concepts = (
        _load_concept_metadata(entities_folder / "concept.json", concept_ids)
        if settings.MOOCCUBEX_ENCODE_CONCEPT_TEXT
        else None
    )

    schema = {
        "users": {
            "bin": [],
            # A fully missing field carries no information and makes the
            # shared mean-imputation/scaling pipeline degenerate.
            "num": ["year_of_birth"] if users["year_of_birth"].notna().any() else [],
            "cat": ["gender", "school"],
            "text": [],
            "list": [],
        },
        "items": {
            "bin": [],
            "num": ["resource_count"],
            "cat": [],
            "text": ["name", "prerequisites", "about"],
            "list": ["field"],
            # IDs used only by the KG: never materialize a dense concept matrix.
            # Prerequisite descriptions remain text, not course references.
            "graph_list": ["concept", "teacher", "school"],
            "graph_text": {"concept": ["name", "context"]}
            if settings.MOOCCUBEX_ENCODE_CONCEPT_TEXT
            else {},
        },
        "inter": {
            "bin": [],
            "num": [],
            "cat": [],
            "text": [],
            "list": [],
        },
    }

    return RawData(
        interactions=interactions,
        item_features=items,
        user_features=users,
        schema=schema,
        graph_entities={"concept": concepts} if concepts is not None else None,
        reports={"reduction_report": report} if report is not None else None,
        source_reduced=report is not None,
    )


def _load_concept_metadata(path: Path, concept_ids: set[str]) -> pd.DataFrame:
    """Read linked concepts only; missing metadata falls back to the ID's name."""
    records: dict[str, dict[str, str]] = {}
    for record in _iter_jsonl(path):
        concept_id = record["id"]
        if concept_id in concept_ids:
            records[concept_id] = {
                "id": concept_id,
                "name": record.get("name")
                or concept_id.removeprefix("K_").rsplit("_", 1)[0],
                "context": record.get("context") or "",
            }
    missing = concept_ids - records.keys()
    if missing:
        warnings.warn(
            f"MOOCCubeX: {len(missing)} linked concepts lack metadata; using ID names."
        )
    for concept_id in missing:
        records[concept_id] = {
            "id": concept_id,
            "name": concept_id.removeprefix("K_").rsplit("_", 1)[0],
            "context": "",
        }
    return pd.DataFrame(
        [records[key] for key in sorted(records)], columns=["id", "name", "context"]
    )


def _load_course_relations(
    path: Path,
    course_ids: pd.Series,
    course_column: int,
) -> list[list[str]]:
    """Stream official TSV links, preserving entity IDs and all course memberships.

    Concept links are concept -> course; school/teacher links are course ->
    entity. Ignore courses outside the catalog and deduplicate each membership.
    """
    memberships: dict[str, set[str]] = {str(course): set() for course in course_ids}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2 or not all(part.strip() for part in parts):
                raise ValueError(f"{path}:{line_number}: expected two nonempty TSV IDs")
            course = parts[course_column].strip()
            if course in memberships:
                memberships[course].add(parts[1 - course_column].strip())
    return [sorted(memberships[str(course)]) for course in course_ids]


def prepare_mooccubex(config: "DataConfig") -> RawData:
    """Select users and catalog while streaming, then load graph relations."""
    # Source filters and sampling already ran before loading any KG relations.
    # Reapplying sparsity here would truncate the selected complete histories.
    return load_mooccubex(config)


SPEC = DatasetSpec(
    DatasetName.MOOCCUBEX, FOLDER, download_mooccubex, prepare_mooccubex
)
