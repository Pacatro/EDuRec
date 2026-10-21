import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from edurec import settings
from edurec.datasets.registry import DatasetName, DatasetSpec, RawData
from edurec.datasets.sources.common import ensure_files, extract_zip

FOLDER = DatasetName.DORIS.value


def _fetch_doris(dest: Path) -> None:
    extract_zip(settings.DORIS_ZIP_URL, dest, desc="DORIS dataset (ZIP)")

    subdir = dest / "datasetInEnglish"
    if subdir.is_dir():
        for f in subdir.iterdir():
            shutil.move(str(f), dest / f.name)
        shutil.rmtree(subdir)

    macosx = dest / "__MACOSX"
    if macosx.is_dir():
        shutil.rmtree(macosx)


def download_doris() -> Path:
    dest = settings.RAW_DATA_FOLDER / FOLDER
    return ensure_files(
        dest,
        settings.DORIS_REQUIRED_FILES,
        lambda: _fetch_doris(dest),
    )


def load_doris() -> RawData:
    doris_folder = settings.RAW_DATA_FOLDER / FOLDER
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


SPEC = DatasetSpec(DatasetName.DORIS, FOLDER, download_doris, load_doris)
