from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from edurec.datasets.loaders import Schema

from .. import config

set_config(transform_output="pandas")


# @staticmethod
# def to_torch_feature_matrix(
#     df: pd.DataFrame,
#     id_col: str,
#     device: str | torch.device | None = None,
# ) -> torch.Tensor:
#     feature_cols = [c for c in df.columns if c != id_col]
#     features_np = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
#     feature_matrix = torch.as_tensor(features_np, dtype=torch.float32)
#
#     if device is not None:
#         feature_matrix = feature_matrix.to(device)
#
#     return feature_matrix

# def build_entity_tensors(
#     self,
#     processed_df: pd.DataFrame,
#     device: str | torch.device | None = None,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     user_df, item_df = self.split_entity_feature_frames(processed_df)
#     return (
#         self.to_torch_feature_matrix(
#             user_df, id_col=config.USER_COL, device=device
#         ),
#         self.to_torch_feature_matrix(
#             item_df, id_col=config.ITEM_COL, device=device
#         ),
#     )


@dataclass
class ProcessedFeatures:
    users: pd.DataFrame
    items: pd.DataFrame
    interactions: pd.DataFrame
    preprocessors: dict[str, ColumnTransformer | None]


class DataProcessor:
    def __init__(
        self,
        schema: Schema,
        dense_output: bool = True,
        tfidf_max_features: int = 50_000,
        handle_unknown_ohe: str = "ignore",
        ct_sparse_threshold: float = 0.0,
    ):
        self.schema = schema
        self.dense_output = dense_output
        self.tfidf_max_features = tfidf_max_features
        self.handle_unknown_ohe = handle_unknown_ohe
        self.ct_sparse_threshold = ct_sparse_threshold

        self.preprocessors: dict[str, ColumnTransformer | None] = {
            "users": None,
            "items": None,
            "inter": None,
        }

    def fit(
        self,
        users_train: pd.DataFrame,
        items_train: pd.DataFrame,
        interactions_train: pd.DataFrame,
    ) -> Self:
        self._fit_ct_feats(users_train, "users")
        self._fit_ct_feats(items_train, "items")
        self._fit_ct_feats(interactions_train, "inter")

        return self

    def _fit_ct_feats(self, df: pd.DataFrame, prefix: str) -> None:
        _, num_cols, cat_cols, text_cols, list_cols = _get_column_types(
            self.schema, prefix
        )
        id_cols = [c for c in [config.USER_COL, config.ITEM_COL] if c in df.columns]
        time_col = config.TIME_COL if config.TIME_COL in df.columns else None

        preprocessor = self._build_ct(
            id_cols=id_cols,
            num_cols=num_cols,
            cat_cols=cat_cols,
            text_cols=text_cols,
            time_col=time_col,
        )
        preprocessor.fit(df)
        self.preprocessors[prefix] = preprocessor

    def _build_ct(
        self,
        id_cols: list[str],
        # bin_cols: list[str],
        num_cols: list[str],
        cat_cols: list[str],
        text_cols: list[str],
        time_col: str | None = None,
    ) -> ColumnTransformer:
        transformers = []

        if time_col and time_col in num_cols:
            num_cols = [c for c in num_cols if c != time_col]

        if num_cols:
            num_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", MinMaxScaler()),
                ]
            )
            transformers.append(("num", num_pipe, num_cols))

        if cat_cols:
            cat_pipe = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="Undefined"),
                    ),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    ),
                ]
            )
            transformers.append(("cat", cat_pipe, cat_cols))

        if text_cols:
            text_pipe = Pipeline(
                [
                    ("concat", TextConcatenator(text_cols)),
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=self.tfidf_max_features, ngram_range=(1, 2)
                        ),
                    ),
                ]
            )
            transformers.append(("text", text_pipe, text_cols))

        if time_col:
            time_pipe = Pipeline(
                [
                    ("time_feats", TimeFeaturesTransformer()),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", MinMaxScaler()),
                ]
            )
            transformers.append(("time", time_pipe, [time_col]))

        if id_cols:
            ids_pipe = Pipeline(
                [
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    ),
                    ("to_long", FunctionTransformer(lambda x: x.astype("int64"))),
                ]
            )
            transformers.append(("ids", ids_pipe, id_cols))

        return ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            sparse_threshold=self.ct_sparse_threshold,
            verbose_feature_names_out=False,
        )

    def transform(
        self,
        users: pd.DataFrame,
        items: pd.DataFrame,
        interactions: pd.DataFrame | None = None,
    ) -> ProcessedFeatures:
        if not self.preprocessors["users"] or not self.preprocessors["items"]:
            raise RuntimeError("DataProcessor not fitted")

        user_processed = self._get_clean_df("users", users)
        item_processed = self._get_clean_df("items", items)
        inter_processed = self._get_clean_df("inter", interactions)

        return ProcessedFeatures(
            users=user_processed,
            items=item_processed,
            interactions=inter_processed,
            preprocessors=self.preprocessors,
        )

    def _get_clean_df(self, key: str, original_df: pd.DataFrame | None) -> Any:
        ct = self.preprocessors[key]
        if original_df is None or ct is None:
            raise RuntimeError(f"DataProcessor not fitted for {key}")

        processed_data = ct.transform(original_df)

        if hasattr(processed_data, "toarray"):
            df = pd.DataFrame(
                np.array(processed_data),
                index=original_df.index,
                columns=ct.get_feature_names_out(),
            )
        else:
            df = processed_data

        return df


def _get_column_types(
    schema: Schema, prefix: str
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    bin_cols = schema[prefix]["bin"]
    num_cols = schema[prefix]["num"]
    cat_cols = schema[prefix]["cat"]
    text_cols = schema[prefix]["text"]
    list_cols = schema[prefix]["list"]

    return bin_cols, num_cols, cat_cols, text_cols, list_cols


class TimeFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Entrada: una columna TIME_COL con strings/datetime.
    Salida: DataFrame con features numéricas (float) listas para imputar/escalar.
    """

    def fit(self, X, y=None) -> Self:
        _ = X, y
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        X_df = pd.DataFrame(X)
        s = X_df.iloc[:, 0]

        dt = pd.to_datetime(s, errors="coerce", utc=True)

        # timestamp in secods
        ts = (dt.astype(np.int64) // 10**9).astype("float64")

        # hour = dt.dt.hour.astype("float64")
        # dow = dt.dt.dayofweek.astype("float64")  # 0=lunes
        # month = dt.dt.month.astype("float64")
        #
        # # Circular features to preserve temporal continuity.
        # hour_sin = np.sin(2 * np.pi * (hour / 24.0))
        # hour_cos = np.cos(2 * np.pi * (hour / 24.0))
        # dow_sin = np.sin(2 * np.pi * (dow / 7.0))
        # dow_cos = np.cos(2 * np.pi * (dow / 7.0))

        return pd.DataFrame(
            {
                "time_ts": ts,
                # "time_hour": hour,
                # "time_dow": dow,
                # "time_month": month,
                # "time_hour_sin": hour_sin,
                # "time_hour_cos": hour_cos,
                # "time_dow_sin": dow_sin,
                # "time_dow_cos": dow_cos,
            },
            index=X_df.index,
        )


class TextConcatenator(BaseEstimator, TransformerMixin):
    """Concatena varias columnas textuales en un único string por fila (para TF-IDF)."""

    def __init__(self, cols: list[str]):
        self.cols = cols

    def fit(self, X, y=None) -> Self:
        _ = X, y
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        Xdf = (
            pd.DataFrame(X, columns=self.cols)
            if not isinstance(X, pd.DataFrame)
            else X[self.cols]
        )
        return np.array(Xdf.fillna("").astype(str).agg(" ".join, axis=1).values)
