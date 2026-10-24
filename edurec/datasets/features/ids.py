"""Canonical integer identifiers, aligned with user/item feature table rows."""

import numpy as np
import pandas as pd

from edurec import settings


def encode_ids(
    users: pd.DataFrame,
    items: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
) -> tuple[
    pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict[object, int]]
]:
    """Map the selected catalog in table order and reject missing/duplicate IDs.

    Output entity row i has canonical ID i. Source IDs remain available in the
    maps, while every output table uses int64 canonical IDs.
    """
    id_maps: dict[str, dict[object, int]] = {}
    tables = {}
    for name, frame, column in (
        ("users", users, settings.USER_COL),
        ("items", items, settings.ITEM_COL),
    ):
        if frame[column].isna().any() or frame[column].duplicated().any():
            raise ValueError(f"{name}: {column} must be non-null and unique.")
        id_maps[name] = {value: idx for idx, value in enumerate(frame[column])}
        tables[name] = frame.reset_index(drop=True).copy()
        tables[name][column] = np.arange(len(frame), dtype=np.int64)

    mapped_splits = {}
    for name, frame in splits.items():
        mapped = frame.reset_index(drop=True).copy()
        for entity, column in (
            ("users", settings.USER_COL),
            ("items", settings.ITEM_COL),
        ):
            values = mapped[column].map(id_maps[entity])
            if values.isna().any():
                raise ValueError(
                    f"{name}: {column} contains IDs outside the {entity} catalog."
                )
            mapped[column] = values.astype(np.int64)
        mapped_splits[name] = mapped
    return tables["users"], tables["items"], mapped_splits, id_maps
