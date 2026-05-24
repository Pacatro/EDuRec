import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import settings
from .cache import ProcessedData
from .dataprocessor import FeatureMetadata

RESERVED_FIELDS = {
    settings.USER_COL,
    settings.ITEM_COL,
    settings.RATING_COL,
    settings.RELEVANT_COL,
    settings.TIME_COL,
}


def save_atomic_files(
    artifacts: ProcessedData,
    dataset_name: str,
    output_dir: Path,
) -> dict[str, Path]:
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

    inter_categorical_cols = set(
        processor.feature_metadata.get("inter", FeatureMetadata()).categorical_cols
    )
    user_categorical_cols = set(
        processor.feature_metadata.get("users", FeatureMetadata()).categorical_cols
    )
    item_categorical_cols = set(
        processor.feature_metadata.get("items", FeatureMetadata()).categorical_cols
    )

    atomic_files: dict[str, Path] = {}

    inter_df = pd.concat(
        [_positive_interactions(df) for df in split_frames.values() if df is not None],
        axis=0,
        ignore_index=True,
    )
    inter_df = inter_df[_ordered_inter_columns(inter_df)]

    atomic_inter, _ = format_atomic_frame(
        inter_df,
        categorical_cols=inter_categorical_cols,
        namespace="inter",
    )

    atomic_files["inter"] = output_dir / f"{dataset_name}.inter"
    atomic_inter.to_csv(atomic_files["inter"], sep="\t", index=False)

    for split_name, split_df in split_frames.items():
        if split_df is None:
            continue

        split_df = _positive_interactions(split_df)
        split_df = split_df[_ordered_inter_columns(split_df)]

        atomic_split, _ = format_atomic_frame(
            split_df,
            categorical_cols=inter_categorical_cols,
            namespace="inter",
        )

        atomic_files[f"{split_name}.inter"] = (
            output_dir / f"{dataset_name}.{split_name}.inter"
        )
        atomic_split.to_csv(atomic_files[f"{split_name}.inter"], sep="\t", index=False)

    user_df = static_feature_frame(
        artifacts.u_static_feats,
        processor.feature_metadata.get("users"),
        prefix="users",
        id_col=settings.USER_COL,
    )
    atomic_user, _ = format_atomic_frame(
        user_df,
        categorical_cols=user_categorical_cols,
        namespace="user",
    )
    atomic_files["user"] = output_dir / f"{dataset_name}.user"
    atomic_user.to_csv(atomic_files["user"], sep="\t", index=False)

    item_df = static_feature_frame(
        artifacts.i_static_feats,
        processor.feature_metadata.get("items"),
        prefix="items",
        id_col=settings.ITEM_COL,
    )
    atomic_item, _ = format_atomic_frame(
        item_df,
        categorical_cols=item_categorical_cols,
        namespace="item",
    )
    atomic_files["item"] = output_dir / f"{dataset_name}.item"
    atomic_item.to_csv(atomic_files["item"], sep="\t", index=False)

    return atomic_files


def _positive_interactions(df: pd.DataFrame) -> pd.DataFrame:
    if settings.RELEVANT_COL not in df.columns:
        return df.reset_index(drop=True)

    return df.loc[df[settings.RELEVANT_COL] > 0].reset_index(drop=True)


def _ordered_inter_columns(df: pd.DataFrame) -> list[str]:
    front_cols = [
        settings.USER_COL,
        settings.ITEM_COL,
        settings.RATING_COL,
        settings.RELEVANT_COL,
        settings.TIME_COL,
    ]
    return [col for col in front_cols if col in df.columns] + [
        col for col in df.columns if col not in front_cols
    ]


def format_atomic_frame(
    df: pd.DataFrame,
    categorical_cols: set[str],
    namespace: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy().reset_index(drop=True)
    name_map = _build_column_name_map(df.columns.tolist(), namespace=namespace)

    data: dict[str, pd.Series] = {}
    load_cols: list[str] = []

    for col in df.columns:
        recbole_name = name_map[col]
        recbole_type = infer_field_type(col, categorical_cols)
        atomic_col = f"{recbole_name}:{recbole_type}"
        load_cols.append(recbole_name)

        if recbole_type == "token":
            values = df[col]
            numeric_values = pd.to_numeric(values, errors="coerce")

            if numeric_values.notna().all():
                data[atomic_col] = numeric_values.round().astype("int64").astype(str)
            else:
                data[atomic_col] = (
                    values.fillna("Undefined")
                    .astype(str)
                    .str.strip()
                    .replace("", "Undefined")
                )
        else:
            data[atomic_col] = (
                pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)
            )

    formatted = pd.DataFrame(data, index=df.index)

    return formatted, load_cols


def _build_column_name_map(
    columns: list[str],
    namespace: str | None = None,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()

    for col in columns:
        base = sanitize_field_name(col)
        if col not in RESERVED_FIELDS and namespace is not None:
            base = f"{namespace}_{base}"

        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1

        mapping[col] = candidate
        used.add(candidate)

    return mapping


def sanitize_field_name(name: object) -> str:
    field = str(name).strip().lower()
    field = re.sub(r"[^0-9a-zA-Z_]+", "_", field)
    field = re.sub(r"_+", "_", field).strip("_")
    if not field:
        field = "field"
    if field[0].isdigit():
        field = f"f_{field}"
    return field


def infer_field_type(col: str, categorical_cols: set[str]) -> str:
    if col in {settings.USER_COL, settings.ITEM_COL}:
        return "token"
    if col in categorical_cols:
        return "token"
    return "float"


def static_feature_frame(
    tensor: torch.Tensor | None,
    metadata: FeatureMetadata | None,
    prefix: str,
    id_col: str,
) -> pd.DataFrame:
    if tensor is None:
        raise RuntimeError(f"{prefix} static features are not available.")
    if metadata is None:
        raise RuntimeError(f"{prefix} feature metadata is not available.")

    feature_cols = (
        metadata.dense_cols + metadata.text_embedding_cols + metadata.categorical_cols
    )

    values = tensor.detach().cpu().numpy()

    if values.shape[1] != len(feature_cols):
        raise RuntimeError(
            f"Expected {len(feature_cols)} {prefix} feature columns, "
            f"got tensor with shape {values.shape}."
        )

    id_df = pd.DataFrame({id_col: np.arange(values.shape[0], dtype=np.int64)})
    feature_df = pd.DataFrame(values, columns=feature_cols)

    return pd.concat([id_df, feature_df], axis=1)
