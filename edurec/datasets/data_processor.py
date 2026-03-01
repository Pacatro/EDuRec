from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from edurec.datasets.loaders import Schema

from .. import config

set_config(transform_output="pandas")


@dataclass
class ProcessedFeatures:
    users: pd.DataFrame | None
    items: pd.DataFrame | None
    interactions: pd.DataFrame | None
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

        self.user_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self.item_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )

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
        all_user_ids = pd.concat(
            [users_train[[config.USER_COL]], interactions_train[[config.USER_COL]]]
        ).drop_duplicates()
        self.user_encoder.fit(all_user_ids)

        all_item_ids = pd.concat(
            [items_train[[config.ITEM_COL]], interactions_train[[config.ITEM_COL]]]
        ).drop_duplicates()
        self.item_encoder.fit(all_item_ids)

        self._fit_ct_feats(users_train, "users")
        self._fit_ct_feats(items_train, "items")
        self._fit_ct_feats(interactions_train, "inter")

        return self

    def _fit_ct_feats(self, df: pd.DataFrame, prefix: str) -> None:
        _, num_cols, cat_cols, text_cols, list_cols = _get_column_types(
            self.schema, prefix
        )
        time_col = config.TIME_COL if config.TIME_COL in df.columns else None

        # FIXME: THIS IS ONLY FOR TESTING, REMOVE WHEN THE LIST AND TEXT COLS PROCESSING ARE IMPLEMENTED
        df = df.drop(columns=list_cols + text_cols)

        preprocessor = self._build_ct(
            num_cols=num_cols,
            cat_cols=cat_cols,
            text_cols=[],
            time_col=time_col,
        )
        preprocessor.fit(df)
        self.preprocessors[prefix] = preprocessor

    def _build_ct(
        self,
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
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
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

        return ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            sparse_threshold=self.ct_sparse_threshold,
            verbose_feature_names_out=False,
        )

    def transform(
        self,
        users: pd.DataFrame | None = None,
        items: pd.DataFrame | None = None,
        interactions: pd.DataFrame | None = None,
    ) -> ProcessedFeatures:
        if not self.preprocessors["users"] or not self.preprocessors["items"]:
            raise RuntimeError("DataProcessor not fitted")

        user_processed = self._transform("users", users) if users is not None else None
        item_processed = self._transform("items", items) if items is not None else None
        inter_processed = (
            self._transform("inter", interactions) if interactions is not None else None
        )
        return ProcessedFeatures(
            users=user_processed,
            items=item_processed,
            interactions=inter_processed,
            preprocessors=self.preprocessors,
        )

    def _transform(self, key: str, df: pd.DataFrame | None) -> pd.DataFrame:
        ct = self.preprocessors[key]

        if ct is None:
            raise RuntimeError(f"DataProcessor not fitted for {key}")

        processed_df = ct.transform(df)

        assert isinstance(processed_df, pd.DataFrame)

        if config.USER_COL in processed_df.columns:
            processed_df[config.USER_COL] = self.user_encoder.transform(
                processed_df[[config.USER_COL]]
            ).astype("int64")

        if config.ITEM_COL in processed_df.columns:
            processed_df[config.ITEM_COL] = self.item_encoder.transform(
                processed_df[[config.ITEM_COL]]
            ).astype("int64")

        return processed_df


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

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Xdf = (
            pd.DataFrame(X, columns=self.cols)
            if not isinstance(X, pd.DataFrame)
            else X[self.cols]
        )
        return np.array(Xdf.fillna("").astype(str).agg(" ".join, axis=1).values)
