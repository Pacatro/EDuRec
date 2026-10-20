import ast
import json
import warnings
from collections import Counter
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd

from edurec import settings
from edurec.datasets.mooccubex import (
    MooccubexSelection,
    select_enrollments,
    selection_config,
)


class DatasetName(StrEnum):
    EXPLICIT_MARS = "explicit_mars"
    IMPLICIT_MARS = "implicit_mars"
    ITM = "itm"
    DORIS = "doris"
    MOOCCUBEX = "mooccubex"
    COCO = "coco"


type Schema = dict[str, dict[str, Any]]


class RawData(NamedTuple):
    interactions: pd.DataFrame
    item_features: pd.DataFrame
    user_features: pd.DataFrame
    schema: Schema
    graph_entities: dict[str, pd.DataFrame] | None = None
    selection_report: dict[str, Any] | None = None


type ExportFn = Callable[[], RawData]

dataset_loaders: dict[DatasetName, ExportFn] = {}


def register_dataset[Loader: ExportFn](
    ds_name: DatasetName,
) -> Callable[[Loader], Loader]:
    """Decorator for registering a dataset loader.

    This decorator associates a dataset loading function with a given
    `DatasetName` and stores it in the global `dataset_loaders` dictionary.

    Example:
        ```python
        @register_dataset(DatasetName.EXPLICIT_MARS)
        def load_explicit_mars() -> RawData:
            ...
        ```

    Args:
        ds_name (DatasetName): The dataset identifier to register.

    Returns:
        Callable[[ExportFn], ExportFn]: The decorator that registers the loader.
    """

    def decorator(fn: Loader) -> Loader:
        dataset_loaders[ds_name] = fn
        return fn

    return decorator


def _to_epoch_seconds(values: pd.Series) -> pd.Series:
    """Convert a datetime-like column to integer seconds, using NaN for NaT."""
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    seconds = parsed.astype("int64") // 10**9
    return seconds.where(parsed.notna())


def load_mars(data_type: Literal["explicit", "implicit"]) -> RawData:
    """Load the MARS dataset.

    Combines the English and French rating files and merges them with item
    metadata. Dataset-wide cleaning and filtering happen later in the
    datamodule preprocessing phase.
    """
    mars_folder = settings.RAW_DATA_FOLDER / "mars"
    items_en = pd.read_csv(mars_folder / "items_en.csv")
    items_fr = pd.read_csv(mars_folder / "items_fr.csv")
    users_en = pd.read_csv(mars_folder / "users_en.csv")
    users_fr = pd.read_csv(mars_folder / "users_fr.csv")
    ratings_en = pd.read_csv(mars_folder / f"{data_type}_ratings_en.csv")
    ratings_fr = pd.read_csv(mars_folder / f"{data_type}_ratings_fr.csv")

    interactions = pd.concat([ratings_en, ratings_fr], ignore_index=True)
    items = pd.concat([items_en, items_fr], ignore_index=True)
    users = pd.concat([users_en, users_fr], ignore_index=True)

    interactions.rename(
        columns={
            "user_id": settings.USER_COL,
            "item_id": settings.ITEM_COL,
            "rating": settings.RATING_COL,
            "created_at": settings.TIME_COL,
        },
        inplace=True,
    )

    if settings.TIME_COL in interactions.columns:
        timestamp_seconds = _to_epoch_seconds(interactions[settings.TIME_COL])
        valid_timestamps = timestamp_seconds.notna() & timestamp_seconds.ge(0)
        num_invalid = int((~valid_timestamps).sum())
        if num_invalid:
            warnings.warn(
                f"Dropping {num_invalid} MARS interactions with invalid "
                "created_at timestamps.",
                stacklevel=2,
            )
        interactions = interactions.loc[valid_timestamps].copy()
        interactions[settings.TIME_COL] = timestamp_seconds.loc[
            valid_timestamps
        ].astype(np.int64)
        interactions = interactions.reset_index(drop=True)

    items.rename(
        columns={"item_id": settings.ITEM_COL, "type": "item_type"},
        inplace=True,
    )

    users.rename(columns={"user_id": settings.USER_COL}, inplace=True)

    schema = {
        "users": {
            "bin": [],
            "num": [],
            "cat": ["job"],
            "text": [],
            "list": [],
        },
        "items": {
            "bin": [],
            "num": ["nb_views", "duration"],
            "cat": ["language", "difficulty", "item_type"],
            "text": ["name", "description"],
            "list": ["job", "software", "theme"],
            "graph_text": {
                field: ["name"]
                for field in ("job", "software", "theme", "item_type", "difficulty")
            },
        },
        "inter": {
            "bin": [],
            "num": ["watch_percentage"] if data_type == "explicit" else [],
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
    )


@register_dataset(DatasetName.EXPLICIT_MARS)
def load_explicit_mars() -> RawData:
    return load_mars(data_type="explicit")


@register_dataset(DatasetName.IMPLICIT_MARS)
def load_implicit_mars() -> RawData:
    return load_mars(data_type="implicit")


@register_dataset(DatasetName.ITM)
def load_itm() -> RawData:
    """Load the ITM dataset.

    Merges ratings, items, and user information into a unified structure.
    Dataset-wide cleaning and filtering happen later in the datamodule
    preprocessing phase.
    """
    itm_folder = settings.RAW_DATA_FOLDER / DatasetName.ITM.value
    ratings_df = pd.read_csv(itm_folder / "ratings.csv")
    items_df = pd.read_csv(itm_folder / "items.csv")
    users_df = pd.read_csv(itm_folder / "users.csv")

    ratings_df.rename(
        columns={
            "UserID": settings.USER_COL,
            "Item": settings.ITEM_COL,
            "Rating": settings.RATING_COL,
        },
        inplace=True,
    )
    # App, Data and Ease are post-interaction multi-criteria ratings, not
    # contextual variables available when producing a recommendation. Remove
    # them at the data boundary to prevent target leakage downstream.
    ratings_df = ratings_df.drop(columns=["App", "Data", "Ease"])
    items_df.rename(columns={"Item": settings.ITEM_COL}, inplace=True)
    users_df.rename(columns={"UserID": settings.USER_COL}, inplace=True)

    items_df = items_df.drop(["URL"], axis=1)

    schema = {
        "users": {
            "bin": ["gender", "married"],
            "num": [],
            "cat": ["age"],
            "text": [],
            "list": [],
        },
        "items": {
            "bin": [],
            "num": [],
            "cat": [],
            "text": ["title", "descriptions"],
            "list": [],
        },
        "inter": {
            "bin": [],
            "num": [],
            "cat": ["class", "semester", "lockdown"],
            "text": [],
            "list": [],
        },
    }

    return RawData(
        interactions=ratings_df,
        item_features=items_df,
        user_features=users_df,
        schema=schema,
    )


@register_dataset(DatasetName.DORIS)
def load_doris() -> RawData:
    doris_folder = settings.RAW_DATA_FOLDER / DatasetName.DORIS.value
    ratings_df = pd.read_excel(doris_folder / "CourseSelectionTable.xlsx")
    items_df = pd.read_excel(doris_folder / "CourseInformationTable.xlsx")
    users_df = pd.read_excel(doris_folder / "StudentInformationTable.xlsx")

    ratings_df.rename(
        columns={
            "StudedntId": settings.USER_COL,
            "CourseId": settings.ITEM_COL,
            "Score": settings.RATING_COL,
            "AcademicYear": settings.TIME_COL,
        },
        inplace=True,
    )

    ratings_df[settings.RATING_COL] = pd.to_numeric(
        ratings_df[settings.RATING_COL],
        errors="coerce",
    )
    ratings_df = ratings_df.dropna(subset=[settings.RATING_COL]).reset_index(drop=True)

    start_year = (
        ratings_df[settings.TIME_COL]
        .astype("string")
        .str.extract(r"^(\d{2})-", expand=False)
    )
    ratings_df[settings.TIME_COL] = pd.to_numeric("20" + start_year, errors="coerce")

    ratings_df = ratings_df.dropna(subset=[settings.TIME_COL]).reset_index(drop=True)
    ratings_df[settings.TIME_COL] = ratings_df[settings.TIME_COL].astype(np.int64)
    items_df.rename(
        columns={"CourseId": settings.ITEM_COL, "Type": "item_type"}, inplace=True
    )
    users_df.rename(columns={"StudentId": settings.USER_COL}, inplace=True)

    schema = {
        "users": {
            "bin": [],
            "num": [],
            "cat": ["enrollmentyear", "education", "major"],
            "text": [],
            "list": [],
        },
        "items": {
            "bin": [],
            "num": [],
            "cat": ["item_type", "grade", "coursecollege", "department"],
            "text": ["coursename", "introduction", "prerequisite"],
            "list": [],
            "refs": {"prerequisite": "coursename"},
            "graph_text": {
                field: ["name"]
                for field in ("item_type", "coursecollege", "department")
            },
        },
        "inter": {
            "bin": [],
            "num": [],
            "cat": ["semester", "coursecollege"],
            "text": ["coursename"],
            "list": [],
        },
    }

    return RawData(
        interactions=ratings_df,
        item_features=items_df,
        user_features=users_df,
        schema=schema,
    )


@register_dataset(DatasetName.MOOCCUBEX)
def load_mooccubex(config: MooccubexSelection | None = None) -> RawData:
    """Select complete enrollment histories and preserve static course links."""
    config = config or selection_config()
    entities_folder = settings.RAW_DATA_FOLDER / "mooccubex" / "entities"
    items = pd.read_json(entities_folder / "course.json", lines=True)
    items.rename(columns={"id": settings.ITEM_COL}, inplace=True)
    items = items.drop_duplicates(settings.ITEM_COL).copy()
    # Concepts are needed before selecting a coherent course community. Read
    # all links, while text encoding remains restricted to the final catalog.
    relations_folder = entities_folder.parent / "relations"
    items["concept"] = _load_course_relations(
        relations_folder / "concept-course.txt",
        items[settings.ITEM_COL],
        course_column=1,
    )
    interactions, users, report = select_enrollments(
        entities_folder / "user.json",
        items,
        config,
    )
    items = items.loc[
        items[settings.ITEM_COL].isin(interactions[settings.ITEM_COL])
    ].copy()
    items["resource_count"] = items["resource"].map(len)
    items = items.drop(columns=["resource"])
    for field, filename in (
        ("teacher", "course-teacher.txt"),
        ("school", "course-school.txt"),
    ):
        items[field] = _load_course_relations(
            relations_folder / filename,
            items[settings.ITEM_COL],
            course_column=0,
        )
    concept_ids = {concept for concepts in items["concept"] for concept in concepts}
    concepts = (
        _load_concept_metadata(entities_folder / "concept.json", concept_ids)
        if config.encode_concept_text
        else None
    )

    concept_support = Counter(
        concept for values in items["concept"] for concept in values
    )
    report["graph"] = {
        "concept_nodes": len(concept_support),
        "shared_concept_nodes": sum(count > 1 for count in concept_support.values()),
        "courses_without_shared_concepts": sum(
            not any(concept_support[concept] > 1 for concept in values)
            for values in items["concept"]
        ),
        "memberships": {
            field: int(items[field].map(len).sum())
            for field in ("field", "concept", "teacher", "school")
        },
    }
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
            if config.encode_concept_text
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
        selection_report=report,
    )


def _load_concept_metadata(path: Path, concept_ids: set[str]) -> pd.DataFrame:
    """Read linked concepts only; missing metadata falls back to the ID's name."""
    records: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
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
    *,
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


COCO_TEXT_COLS = (
    "short_description",
    "objectives",
    "requirements",
    "target_audience",
    "long_description",
)
COCO_LESSON_CLASSES = {
    "lecture": "num_lectures",
    "chapter": "num_chapters",
    "quiz": "num_quizzes",
    "practice": "num_practices",
}
COCO_INSTRUCTOR_COLS = (
    "num_instructors",
    "instructor_enrollments",
    "instructor_reviews",
)


def _coco_instructor_features(coco_folder: Path) -> pd.DataFrame:
    """Aggregate instructor statistics per course."""
    teach = pd.read_csv(
        coco_folder / "teach_latest.csv",
        usecols=["course_id", "instructor_id"],
    )
    instructors = pd.read_csv(coco_folder / "instructor_latest.csv")
    merged = teach.merge(instructors, on="instructor_id", how="left")

    return merged.groupby("course_id").agg(
        num_instructors=("instructor_id", "nunique"),
        instructor_enrollments=("total_enrollments", "sum"),
        instructor_reviews=("total_reviews", "sum"),
    )


def _coco_curriculum_features(coco_folder: Path) -> pd.DataFrame:
    """Count lessons, chapters, quizzes, and practices per course."""
    curriculum = pd.read_csv(
        coco_folder / "curriculum_lesson_chapter_latest.csv",
        usecols=["course_id", "class"],
    )
    counts = curriculum.groupby("course_id")["class"].value_counts().unstack()
    counts = counts.reindex(columns=list(COCO_LESSON_CLASSES), fill_value=0)
    counts.rename(columns=COCO_LESSON_CLASSES, inplace=True)

    return counts


def _coco_instructor_graph(
    coco_folder: Path,
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """Keep course-instructor memberships and available static profile text.

    IDs are read as strings in both files to avoid float coercion from missing
    values. No enrollment counts, review statistics or learner comments enter
    the entity text.
    """
    teach = pd.read_csv(
        coco_folder / "teach_latest.csv",
        usecols=["course_id", "instructor_id"],
        dtype={"instructor_id": "string"},
    ).dropna(subset=["course_id", "instructor_id"])
    memberships = teach.groupby("course_id")["instructor_id"].agg(
        lambda ids: sorted({value.strip() for value in ids if value.strip()})
    )
    instructors = pd.read_csv(
        coco_folder / "instructor_latest.csv", dtype={"instructor_id": "string"}
    ).rename(columns={"instructor_id": "id"})
    instructors = instructors.dropna(subset=["id"]).copy()
    instructors["id"] = instructors["id"].str.strip()
    instructors = instructors.drop_duplicates("id")
    text_columns = [
        column
        for column in (
            "name",
            "display_name",
            "title",
            "job_title",
            "headline",
            "description",
            "biography",
        )
        if column in instructors
    ]
    return memberships, instructors[["id", *text_columns]], text_columns


def _flatten_list_text(value: object) -> object:
    """Turn stringified lists into plain text for the embedding model."""
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return value

    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return value

    if isinstance(parsed, (list, tuple)):
        return " ".join(str(item).strip() for item in parsed if str(item).strip())

    return value


@register_dataset(DatasetName.COCO)
def load_coco() -> RawData:
    """Load the COCO course-review dataset.

    Ratings live in ``evaluate_latest.csv``; the free-text reviewer comment is
    dropped because it is written after the rating and would leak the relevance
    target. Course metadata is enriched with curriculum and instructor
    aggregates so the item encoder receives course-level signals.
    """
    coco_folder = settings.RAW_DATA_FOLDER / DatasetName.COCO.value

    interactions = pd.read_csv(
        coco_folder / "evaluate_latest.csv",
        usecols=["learner_id", "course_id", "learner_rating", "learner_timestamp"],
        nrows=settings.COCO_MAX_INTERACTIONS,
    )
    interactions.dropna(inplace=True)
    interactions.rename(
        columns={
            "learner_id": settings.USER_COL,
            "course_id": settings.ITEM_COL,
            "learner_rating": settings.RATING_COL,
            "learner_timestamp": settings.TIME_COL,
        },
        inplace=True,
    )
    interactions[settings.ITEM_COL] = interactions[settings.ITEM_COL].astype(np.int64)
    interactions[settings.RATING_COL] = interactions[settings.RATING_COL].astype(
        np.float32
    )

    timestamps = pd.to_datetime(
        interactions[settings.TIME_COL], errors="coerce", utc=True
    )
    interactions = interactions.loc[timestamps.notna()].copy()
    interactions[settings.TIME_COL] = (
        timestamps.loc[timestamps.notna()].astype("int64") // 10**9
    )
    interactions = interactions.reset_index(drop=True)

    items = pd.read_csv(coco_folder / "course_latest.csv")
    items.rename(columns={"course_id": settings.ITEM_COL}, inplace=True)
    items = items.drop(columns=["short_url"], errors="ignore")
    for col in COCO_TEXT_COLS:
        if col in items:
            items[col] = items[col].map(_flatten_list_text)

    items = items.merge(
        _coco_instructor_features(coco_folder),
        left_on=settings.ITEM_COL,
        right_index=True,
        how="left",
    )
    items = items.merge(
        _coco_curriculum_features(coco_folder),
        left_on=settings.ITEM_COL,
        right_index=True,
        how="left",
    )
    numeric_cols = [
        *COCO_INSTRUCTOR_COLS,
        *COCO_LESSON_CLASSES.values(),
    ]
    items[numeric_cols] = items[numeric_cols].fillna(0).astype(np.float32)
    items = items.reset_index(drop=True)
    instructor_links, instructor_entities, instructor_text = _coco_instructor_graph(
        coco_folder
    )
    items["instructor"] = items[settings.ITEM_COL].map(instructor_links)
    items["instructor"] = items["instructor"].map(
        lambda value: value if isinstance(value, list) else []
    )

    users = interactions[[settings.USER_COL]].drop_duplicates().reset_index(drop=True)

    schema = {
        "users": {
            "bin": [],
            "num": [],
            "cat": [],
            "text": [],
            "list": [],
        },
        "items": {
            "bin": [],
            "num": list(numeric_cols),
            "cat": [
                "language",
                "first_level_category",
                "second_level_category",
                "instructional_level",
            ],
            "text": list(COCO_TEXT_COLS),
            "list": ["subtitles"],
            "graph_list": ["instructor"],
            "graph_text": {
                "first_level_category": ["name"],
                "second_level_category": ["name"],
                "instructional_level": ["name"],
                **({"instructor": instructor_text} if instructor_text else {}),
            },
            "cooc": [
                ("second_level_category", "first_level_category"),
            ],
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
        graph_entities={"instructor": instructor_entities},
    )


def load_raw_data(dataset_name: DatasetName) -> RawData:
    """Load the specified dataset through its registered loader."""
    loader = dataset_loaders.get(dataset_name)

    if loader is None:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    return loader()
