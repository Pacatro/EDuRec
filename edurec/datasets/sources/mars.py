import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from edurec import settings
from edurec.datasets.registry import DatasetName, DatasetSpec, RawData
from edurec.datasets.sources.common import ensure_files, extract_zip

FOLDER = "mars"


def _to_epoch_seconds(values: pd.Series) -> pd.Series:
    """Convert a datetime-like column to integer seconds, using NaN for NaT."""
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    seconds = parsed.astype("int64") // 10**9
    return seconds.where(parsed.notna())


def download_mars() -> Path:
    dest = settings.RAW_DATA_FOLDER / FOLDER
    return ensure_files(
        dest,
        settings.MARS_REQUIRED_FILES,
        lambda: extract_zip(settings.MARS_ZIP_URL, dest, desc="MARS dataset (ZIP)"),
    )


def load_mars(data_type: Literal["explicit", "implicit"]) -> RawData:
    """Load the MARS dataset.

    Combines the English and French rating files and merges them with item
    metadata. Dataset-wide cleaning and filtering happen later in the
    pipeline preprocessing phase.
    """
    mars_folder = settings.RAW_DATA_FOLDER / FOLDER
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


def load_explicit_mars() -> RawData:
    return load_mars(data_type="explicit")


def load_implicit_mars() -> RawData:
    return load_mars(data_type="implicit")


SPECS = (
    DatasetSpec(DatasetName.EXPLICIT_MARS, FOLDER, download_mars, load_explicit_mars),
    DatasetSpec(DatasetName.IMPLICIT_MARS, FOLDER, download_mars, load_implicit_mars),
)
