import shutil
from pathlib import Path

import kagglehub
import pandas as pd

from edurec import settings
from edurec.datasets.registry import DatasetName, DatasetSpec, RawData
from edurec.datasets.sources.common import ensure_files

FOLDER = DatasetName.ITM.value


def _fetch_itm(dest: Path) -> None:
    src = Path(kagglehub.dataset_download(settings.KAGGLE_ITM_DATASET))
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name in settings.ITM_REQUIRED_FILES:
            shutil.copy2(f, dest / f.name)


def download_itm() -> Path:
    dest = settings.RAW_DATA_FOLDER / FOLDER
    return ensure_files(
        dest,
        settings.ITM_REQUIRED_FILES,
        lambda: _fetch_itm(dest),
    )


def load_itm() -> RawData:
    """Load the ITM dataset.

    Merges ratings, items, and user information into a unified structure.
    Dataset-wide cleaning and filtering happen later in the pipeline
    preprocessing phase.
    """
    itm_folder = settings.RAW_DATA_FOLDER / FOLDER
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


SPEC = DatasetSpec(DatasetName.ITM, FOLDER, download_itm, load_itm)
