import ast
from pathlib import Path

import numpy as np
import pandas as pd

from edurec import settings
from edurec.datasets.sources.common import ensure_files
from edurec.datasets.sources.registry import DatasetName, DatasetSpec, RawData

FOLDER = DatasetName.COCO.value

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


def _report_missing_coco(dest: Path) -> None:
    raise FileNotFoundError(
        "The COCO dataset is not distributed through an automated download. "
        f"Place its CSV files in {dest}: {', '.join(settings.COCO_REQUIRED_FILES)}."
    )


def download_coco() -> Path:
    dest = settings.RAW_DATA_FOLDER / FOLDER
    return ensure_files(
        dest,
        settings.COCO_REQUIRED_FILES,
        lambda: _report_missing_coco(dest),
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


def load_coco() -> RawData:
    """Load the COCO course-review dataset.

    Ratings live in ``evaluate_latest.csv``; the free-text reviewer comment is
    dropped because it is written after the rating and would leak the relevance
    target. Course metadata is enriched with curriculum and instructor
    aggregates so the item encoder receives course-level signals.
    """
    coco_folder = settings.RAW_DATA_FOLDER / FOLDER

    interactions = pd.read_csv(
        coco_folder / "evaluate_latest.csv",
        usecols=["learner_id", "course_id", "learner_rating", "learner_timestamp"],
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


SPEC = DatasetSpec(DatasetName.COCO, FOLDER, download_coco, load_coco)
