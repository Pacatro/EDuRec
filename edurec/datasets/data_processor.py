from typing import Self

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass

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


# TODO: ADAPTAR ESTO PARA QUE SEA UN DATAFRAME EN LUGAR DE UN NDARRAY
@dataclass
class ProcessedFeatures:
    X_users: np.ndarray
    X_items: np.ndarray
    X_interactions: np.ndarray | None
    preprocessors: dict[str, ColumnTransformer | None]

    # used_users_cols: dict[str, list[str]]
    # used_items_cols: dict[str, list[str]]
    # used_inter_cols: dict[str, list[str]]


class DataProcessor:
    def __init__(
        self,
        dense_output: bool = True,
        tfidf_max_features: int = 50_000,
        handle_unknown_ohe: str = "ignore",
        ct_sparse_threshold: float = 0.0,
    ):
        self.dense_output = dense_output
        self.tfidf_max_features = tfidf_max_features
        self.handle_unknown_ohe = handle_unknown_ohe
        self.ct_sparse_threshold = ct_sparse_threshold

        self.preprocessors: dict[str, ColumnTransformer | None] = {
            "users": None,
            "items": None,
            "inter": None,
        }

        # self.used_users_cols: dict[str, list[str]] = {}
        # self.used_items_cols: dict[str, list[str]] = {}
        # self.used_inter_cols: dict[str, list[str]] = {}

    def _build_ct(
        self,
        num_cols: list[str],
        cat_cols: list[str],
        text_cols,
        time_col: str | None = None,
    ) -> ColumnTransformer:
        transformers = []

        if time_col is not None and time_col in num_cols:
            # Exclude TIME_COL from num_cols
            num_cols = [c for c in num_cols if c != time_col]

        if num_cols:
            num_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", MinMaxScaler()),
                ]
            )
            transformers.append(("num", num_pipe, num_cols))

        # cat
        if cat_cols:
            cat_pipe = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="Undefined"),
                    ),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                    # # Shift to 1-based indexing
                    # (
                    #     "shift_plus1",
                    #     FunctionTransformer(lambda x: (x + 1).astype(np.int64)),
                    # ),
                ]
            )
            transformers.append(("cat", cat_pipe, cat_cols))

        # text
        if text_cols:
            text_pipe = Pipeline(
                steps=[
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

        if time_col is not None and time_col:
            time_pipe = Pipeline(
                steps=[
                    ("time_feats", TimeFeaturesTransformer()),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", MinMaxScaler()),
                ]
            )
            transformers.append(("time", time_pipe, [time_col]))

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=self.ct_sparse_threshold,
        )

    def _fit_ct_feats(self, df: pd.DataFrame, prefix: str) -> None:
        num_cols, cat_lens, _, text_cols = get_column_types(df)
        cat_cols = list(cat_lens.keys())
        preprocessor = self._build_ct(num_cols, cat_cols, text_cols)
        preprocessor.fit(df)
        self.preprocessors[prefix] = preprocessor

    def fit(
        self,
        users_train: pd.DataFrame,
        items_train: pd.DataFrame,
        interactions_train: pd.DataFrame | None = None,
    ) -> Self:
        self._fit_ct_feats(users_train, "users")
        self._fit_ct_feats(items_train, "items")

        if interactions_train is not None:
            inum, icat_lens, _, itext = get_column_types(interactions_train)
            icat = list(icat_lens.keys())

            num_inter_feats = (
                len(inum) + len(icat) + len(itext) + (1 if config.TIME_COL else 0)
            )

            if num_inter_feats > 0:
                inter_preprocessor = self._build_ct(inum, icat, itext, config.TIME_COL)
                inter_preprocessor.fit(interactions_train)
                self.preprocessors["inter"] = inter_preprocessor
            else:
                self.preprocessors["inter"] = None
        else:
            self.preprocessors["inter"] = None

        return self

    def transform(
        self,
        users: pd.DataFrame,
        items: pd.DataFrame,
        interactions: pd.DataFrame | None = None,
    ) -> ProcessedFeatures:
        if self.preprocessors["users"] is None or self.preprocessors["items"] is None:
            raise RuntimeError("DataProcessor not fitted")

        X_user = self.preprocessors["users"].transform(users)
        X_item = self.preprocessors["items"].transform(items)
        X_inter = np.array(
            self.preprocessors["inter"].transform(interactions)
            if interactions is not None and self.preprocessors["inter"] is not None
            else None
        )

        return ProcessedFeatures(
            X_users=np.array(X_user),
            X_items=np.array(X_item),
            X_interactions=X_inter,
            preprocessors=self.preprocessors,
        )


def get_column_types(
    df: pd.DataFrame,
) -> tuple[list[str], dict[str, int], list[str], list[str]]:
    exclude_cols = [config.RATING_COL, config.RELEVANT_COL, config.TIME_COL]
    id_cols = [config.USER_COL, config.ITEM_COL]
    numeric_cols, list_cols, text_cols = [], [], []
    categorical_lenghts: dict[str, int] = {}

    for col in df.columns:
        if col in exclude_cols:
            continue

        if col in id_cols:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
            continue

        non_null_series = df[col].dropna()
        if non_null_series.empty:
            continue
        sample_value = non_null_series.iloc[0]

        if isinstance(sample_value, list) or (
            isinstance(sample_value, str) and sample_value.startswith("[")
        ):
            list_cols.append(col)
        elif isinstance(sample_value, str):
            num_unique = df[col].nunique()
            avg_words = non_null_series.str.split().str.len().mean()
            if avg_words > 4 or (num_unique / len(df)) > 0.5:
                text_cols.append(col)
            else:
                categorical_lenghts[col] = df[col].nunique()
        else:
            categorical_lenghts[col] = df[col].nunique()

    return numeric_cols, categorical_lenghts, list_cols, text_cols


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

        hour = dt.dt.hour.astype("float64")
        dow = dt.dt.dayofweek.astype("float64")  # 0=lunes
        month = dt.dt.month.astype("float64")

        # Circular features to preserve temporal continuity.
        hour_sin = np.sin(2 * np.pi * (hour / 24.0))
        hour_cos = np.cos(2 * np.pi * (hour / 24.0))
        dow_sin = np.sin(2 * np.pi * (dow / 7.0))
        dow_cos = np.cos(2 * np.pi * (dow / 7.0))

        return pd.DataFrame(
            {
                "time_ts": ts,
                "time_hour": hour,
                "time_dow": dow,
                "time_month": month,
                "time_hour_sin": hour_sin,
                "time_hour_cos": hour_cos,
                "time_dow_sin": dow_sin,
                "time_dow_cos": dow_cos,
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
