"""Export views to RecBole atomic files.

The exporter only serialises rows that already exist in the processed splits:
it never rebuilds splits nor changes positives/negatives. RecBole may remap
tokens internally, so callers must verify its IDs after loading (see
:mod:`edurec.evaluation.baselines.recbole`).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pandas as pd

from edurec import settings
from edurec.datasets.cache import ProcessedData
from edurec.datasets.torch_data import RecSysDataset

#: Benchmark split names read by RecBole for the general interaction protocol.
BENCHMARK_SPLITS = ("train", "valid", "test")

#: Benchmark split names read by RecBole for prebuilt sequential histories.
SEQUENTIAL_BENCHMARK_SPLITS = ("train_seq", "valid_seq", "test_seq")

#: Exported sequential file -> processed split that supplies its rows.
SEQUENTIAL_SPLIT_SOURCES = {
    "train_seq": "train",
    "valid_seq": "val",
    "test_seq": "test",
}


def export_atomic_files(
    data: ProcessedData,
    dataset_name: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Write the RecBole atomic files used by the RecBole baselines.

    Only ``user_id`` and ``item_id`` are exported. The RecBole configurations
    load identifier fields exclusively (see
    :func:`edurec.evaluation.baselines.recbole._build_config_dict`), so user/item
    side features are intentionally omitted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    positives = data.positive_splits()
    split_frames = {
        "train": positives["train"],
        "valid": positives["val"],
        "test": positives["test"],
    }

    atomic_files: dict[str, Path] = {}

    combined = pd.concat(split_frames.values(), axis=0, ignore_index=True)

    atomic_files["inter"] = output_dir / f"{dataset_name}.inter"
    _inter_id_frame(combined).to_csv(atomic_files["inter"], sep="\t", index=False)

    for split_name, split_df in split_frames.items():
        path = output_dir / f"{dataset_name}.{split_name}.inter"
        atomic_files[f"{split_name}.inter"] = path
        _inter_id_frame(split_df).to_csv(path, sep="\t", index=False)

    atomic_files["user"] = output_dir / f"{dataset_name}.user"
    _id_frame(
        data.user_features,
        settings.USER_COL,
    ).to_csv(atomic_files["user"], sep="\t", index=False)

    atomic_files["item"] = output_dir / f"{dataset_name}.item"
    _id_frame(
        data.item_features,
        settings.ITEM_COL,
    ).to_csv(atomic_files["item"], sep="\t", index=False)

    return atomic_files


def export_sequential_benchmark(
    views: Mapping[str, RecSysDataset],
    dataset_name: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Write split-preserving sequence rows for RecBole sequential models.

    ``views`` maps each exported split name (``train_seq``, ``valid_seq``,
    ``test_seq``) to its query dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_files: dict[str, Path] = {}

    for split_name in SEQUENTIAL_BENCHMARK_SPLITS:
        view = views.get(split_name)
        if view is None:
            raise RuntimeError(f"Sequential benchmark split {split_name!r} is missing.")
        path = output_dir / f"{dataset_name}.{split_name}.inter"
        build_sequential_frame(split_name, view).to_csv(path, sep="\t", index=False)
        atomic_files[split_name] = path

    return atomic_files


def build_sequential_frame(split_name: str, view: RecSysDataset) -> pd.DataFrame:
    """RecBole rows with the split's target item and its strictly-past items."""
    history_mask = view.history_valid_mask.bool()
    user_ids = view.user_ids
    target_item_ids = view.target_item_ids
    history_items = view.history_items

    if split_name == "train_seq":
        keep = history_mask.any(dim=1)
        keep_array = keep.cpu().numpy()
        user_ids = user_ids[keep_array]
        target_item_ids = target_item_ids[keep_array]
        history_items = history_items[keep]
        history_mask = history_mask[keep]

    if len(user_ids) == 0:
        raise RuntimeError(
            f"Sequential benchmark split {split_name!r} has no usable queries."
        )

    histories = [
        " ".join(str(int(item_id) - 1) for item_id in items[mask].tolist())
        for items, mask in zip(history_items, history_mask, strict=True)
    ]
    return pd.DataFrame(
        {
            f"{settings.USER_COL}:token": [str(int(value)) for value in user_ids],
            f"{settings.ITEM_COL}:token": [
                str(int(value)) for value in target_item_ids
            ],
            f"{settings.ITEM_COL}_list:token_seq": histories,
        }
    )


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
    id_col: str,
) -> pd.DataFrame:
    if frame is None:
        raise RuntimeError(f"{id_col} features are not available.")

    ids = frame[id_col].astype("int64").sort_values()
    return pd.DataFrame({f"{id_col}:token": ids.astype(str).to_numpy()})


def _token_column(values: pd.Series) -> pd.Series:
    return values.astype("int64").astype(str)
