from pathlib import Path
from typing import cast

import pandas as pd

from .. import settings
from .cache import ProcessedData


def save_atomic_files(
    artifacts: ProcessedData,
    dataset_name: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Write the RecBole atomic files used by the SOTA baselines.

    Only ``user_id`` and ``item_id`` are exported. The RecBole configurations
    load identifier fields exclusively (see
    :func:`edurec.evaluation.sota._build_config_dict`), so user/item side
    features are intentionally omitted.
    """
    if not artifacts.is_ready:
        raise RuntimeError("Data must be processed before saving atomic files.")
    if artifacts.data_processor is None:
        raise RuntimeError("Data processor is not available.")

    processor = artifacts.data_processor
    output_dir.mkdir(parents=True, exist_ok=True)

    split_frames = {
        "train": artifacts.train,
        "valid": artifacts.val,
        "test": artifacts.test,
    }
    if any(df is None for df in split_frames.values()):
        raise RuntimeError("Processed interaction splits are not available.")

    atomic_files: dict[str, Path] = {}

    combined = pd.concat(
        [_positive_interactions(df) for df in split_frames.values() if df is not None],
        axis=0,
        ignore_index=True,
    )

    atomic_files["inter"] = output_dir / f"{dataset_name}.inter"
    _inter_id_frame(combined).to_csv(atomic_files["inter"], sep="\t", index=False)

    for split_name, split_df in split_frames.items():
        if split_df is None:
            continue

        path = output_dir / f"{dataset_name}.{split_name}.inter"
        atomic_files[f"{split_name}.inter"] = path
        _inter_id_frame(_positive_interactions(split_df)).to_csv(
            path, sep="\t", index=False
        )

    atomic_files["user"] = output_dir / f"{dataset_name}.user"
    _id_frame(
        artifacts.user_features,
        processor.user_id_map,
        settings.USER_COL,
    ).to_csv(atomic_files["user"], sep="\t", index=False)

    atomic_files["item"] = output_dir / f"{dataset_name}.item"
    _id_frame(
        artifacts.item_features,
        processor.item_id_map,
        settings.ITEM_COL,
    ).to_csv(atomic_files["item"], sep="\t", index=False)

    return atomic_files


def _positive_interactions(df: pd.DataFrame) -> pd.DataFrame:
    if settings.RELEVANT_COL not in df.columns:
        return df.reset_index(drop=True)

    return df.loc[df[settings.RELEVANT_COL] > 0].reset_index(drop=True)


def _inter_id_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction identifiers as RecBole tokens, one row per interaction."""
    missing = {settings.USER_COL, settings.ITEM_COL}.difference(df.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise RuntimeError(f"Interactions are missing identifier columns: {names}.")

    return pd.DataFrame(
        {
            f"{settings.USER_COL}:token": _token_column(
                cast(pd.Series, df[settings.USER_COL])
            ),
            f"{settings.ITEM_COL}:token": _token_column(
                cast(pd.Series, df[settings.ITEM_COL])
            ),
        }
    )


def _id_frame(
    frame: pd.DataFrame | None,
    id_map: dict[object, int],
    id_col: str,
) -> pd.DataFrame:
    if frame is None:
        raise RuntimeError(f"{id_col} features are not available.")

    mapped = cast(pd.Series, frame[id_col]).map(id_map.get)
    ids = cast(pd.Series, mapped.dropna().astype("int64")).sort_values()
    return pd.DataFrame({f"{id_col}:token": ids.astype(str).to_numpy()})


def _token_column(values: pd.Series) -> pd.Series:
    numeric = cast(pd.Series, pd.to_numeric(values, errors="coerce"))
    return pd.Series(
        numeric.to_numpy(dtype="float64").round().astype("int64").astype(str),
        index=values.index,
    )
