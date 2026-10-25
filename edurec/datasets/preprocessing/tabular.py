"""Fit tabular encoders and transform one DataFrame at a time."""

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from edurec import settings
from edurec.datasets.preprocessing import text
from edurec.datasets.preprocessing.parsing import is_missing
from edurec.datasets.sources.registry import Schema

FEATURE_TYPES = {"num": "numeric", "cat": "categorical", "list": "list", "text": "text"}
RESERVED_COLUMNS = {
    settings.USER_COL,
    settings.ITEM_COL,
    settings.RATING_COL,
    settings.RELEVANT_COL,
    settings.TIME_COL,
}
FEATURE_PREFIXES = ("num__", "cat__", "text__")


def resolve_schema(schema: Schema, frames: dict[str, pd.DataFrame]) -> Schema:
    """Resolve enabled, available features once and persist that schema."""
    active = set(settings.PREPROCESS_FEATURE_TYPES)
    invalid = active - set(FEATURE_TYPES.values()) - {"time"}
    if invalid:
        raise ValueError(
            f"Unsupported preprocess feature types: {', '.join(sorted(invalid))}"
        )

    resolved: Schema = {}
    for prefix, frame in frames.items():
        if any(str(col).startswith(FEATURE_PREFIXES) for col in frame.columns):
            raise ValueError(f"{prefix}: source columns use reserved feature prefixes.")
        group = dict(schema.get(prefix, {}))
        group["cat"] = [*group.get("bin", []), *group.get("cat", [])]
        group.pop("bin", None)
        for kind, feature_type in FEATURE_TYPES.items():
            group[kind] = list(
                dict.fromkeys(
                    col
                    for col in group.get(kind, [])
                    if feature_type in active
                    and col in frame
                    and col not in RESERVED_COLUMNS
                )
            )
        group["graph_list"] = [
            col for col in group.get("graph_list", []) if col in frame
        ]
        if "text" not in active:
            group["graph_text"] = {}
        assigned = [col for kind in FEATURE_TYPES for col in group[kind]]
        if len(assigned) != len(set(assigned)):
            raise ValueError(f"{prefix}: a column belongs to multiple feature types.")
        resolved[prefix] = group
    return resolved


def clean_tabular(
    frame: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> pd.DataFrame:
    """Use consistent numeric and object dtypes, including entirely empty fields."""
    result = frame[[*numeric, *categorical]].copy()
    for col in numeric:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    for col in categorical:
        values = result[col].map(
            lambda value: np.nan if is_missing(value) else str(value).strip()
        )
        result[col] = values.mask(values == "", np.nan).astype(object)
    return result


def fit_tabular(
    frame: pd.DataFrame, schema: dict[str, Any]
) -> ColumnTransformer | None:
    """Fit numeric/categorical preprocessing; interactions must use TRAIN only."""
    numeric = schema.get("num", [])
    categorical = [*schema.get("bin", []), *schema.get("cat", [])]
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="mean", keep_empty_features=True),
                        ),
                        ("scaler", MinMaxScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="Undefined",
                                keep_empty_features=True,
                            ),
                        ),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        return None
    transformer = ColumnTransformer(transformers, sparse_threshold=0)
    transformer.fit(clean_tabular(frame, numeric, categorical))
    return transformer


def feature_columns(transformer: ColumnTransformer | None, kind: str) -> list[str]:
    """Encoded column names in the same order as their fitted transformer."""
    if transformer is None:
        return []
    return [
        str(col)
        for col in transformer.get_feature_names_out()
        if col.startswith(f"{kind}__")
    ]


def categorical_cardinalities(transformer: ColumnTransformer | None) -> list[int]:
    """Include row zero for unknown codes, which models shift from -1 to zero."""
    if transformer is None or "cat" not in transformer.named_transformers_:
        return []
    encoder = transformer.named_transformers_["cat"].named_steps["encoder"]
    return [len(categories) + 1 for categories in encoder.categories_]


def transform_features(
    frame: pd.DataFrame,
    transformer: ColumnTransformer | None,
    schema: dict[str, Any],
) -> pd.DataFrame:
    """Replace declared features with num__/cat__/text__ columns; keep IDs intact."""
    numeric = schema.get("num", [])
    categorical = [*schema.get("bin", []), *schema.get("cat", [])]
    text_columns = schema.get("text", [])
    result = frame.drop(columns=[*numeric, *categorical, *text_columns]).copy()
    for col in (settings.RATING_COL, settings.RELEVANT_COL):
        if col in result:
            result[col] = (
                pd.to_numeric(result[col], errors="coerce").fillna(0).astype(np.float32)
            )
    if settings.TIME_COL in result:
        timestamps = pd.to_numeric(result[settings.TIME_COL], errors="coerce")
        if not np.isfinite(timestamps.to_numpy(dtype=np.float64)).all():
            raise ValueError(
                f"Temporal column {settings.TIME_COL!r} contains invalid timestamps."
            )
        result[settings.TIME_COL] = timestamps.astype(np.float64)

    if transformer is not None:
        columns = transformer.get_feature_names_out()
        values = (
            np.asarray(
                transformer.transform(clean_tabular(frame, numeric, categorical)),
                dtype=np.float64,
            )
            if len(frame)
            else np.empty((0, len(columns)), dtype=np.float64)
        )
        encoded = pd.DataFrame(values, columns=columns, index=frame.index)
        num_columns = feature_columns(transformer, "num")
        cat_columns = feature_columns(transformer, "cat")
        encoded[num_columns] = encoded[num_columns].astype(np.float32)
        encoded[cat_columns] = encoded[cat_columns].astype(np.int64)
        result = pd.concat([result, encoded], axis=1)

    if text_columns:
        vectors = text.encode_documents(text.build_documents(frame, text_columns))
        embeddings = pd.DataFrame(
            vectors,
            columns=[f"text__embedding_{idx:03d}" for idx in range(vectors.shape[1])],
            index=frame.index,
        )
        result = pd.concat([result, embeddings], axis=1)
    return result


def dense_tensor(frame: pd.DataFrame) -> torch.Tensor:
    """Numeric features followed by text, as required by the content encoders."""
    columns = [
        *frame.filter(regex=r"^num__").columns,
        *frame.filter(regex=r"^text__").columns,
    ]
    return torch.from_numpy(frame[columns].to_numpy(dtype=np.float32))


def categorical_tensor(frame: pd.DataFrame) -> torch.Tensor:
    return torch.from_numpy(frame.filter(regex=r"^cat__").to_numpy(dtype=np.int64))
