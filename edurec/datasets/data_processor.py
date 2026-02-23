from typing import Self

import numpy as np
import pandas as pd
import torch
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OrdinalEncoder

from .. import config

set_config(transform_output="pandas")


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

        # TODO: Add time features in the future --> only on interactions?
        # hour = dt.dt.hour.astype("float64")
        # dow = dt.dt.dayofweek.astype("float64")  # 0=lunes
        # month = dt.dt.month.astype("float64")
        #
        # # Circular features
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


class DataProcessor:
    def __init__(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
        text_cols: list[str],
        list_cols: list[str],
        id_cols: list[str],
        has_time: bool = False,
        max_length: int = 128,
    ):
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.text_cols = text_cols
        self.list_cols = list_cols
        self.id_cols = id_cols
        self.has_time = has_time
        self.max_length = max_length

        self.pipeline = self._build_pipeline()

    @staticmethod
    def _prefix_feature_columns(
        df: pd.DataFrame, id_col: str, prefix: str
    ) -> pd.DataFrame:
        rename_map = {
            col: f"{prefix}{col}"
            for col in df.columns
            if col != id_col and not col.startswith(prefix)
        }
        return df.rename(columns=rename_map)

    @classmethod
    def merge_raw_features(
        cls,
        interactions_df: pd.DataFrame,
        users_df: pd.DataFrame,
        items_df: pd.DataFrame,
    ) -> pd.DataFrame:
        users_prefixed = cls._prefix_feature_columns(
            users_df.copy(), config.USER_COL, "user_"
        )
        items_prefixed = cls._prefix_feature_columns(
            items_df.copy(), config.ITEM_COL, "item_"
        )

        merged = interactions_df.merge(users_prefixed, on=config.USER_COL, how="left")
        merged = merged.merge(items_prefixed, on=config.ITEM_COL, how="left")
        return merged

    def _build_pipeline(self) -> Pipeline:
        transformers = []

        if self.has_time:
            transformers.append(
                (
                    "time",
                    Pipeline(
                        [
                            ("time_feats", TimeFeaturesTransformer()),
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", MinMaxScaler()),
                            (
                                "to_f32",
                                FunctionTransformer(
                                    lambda x: x.astype(np.float32),
                                    feature_names_out="one-to-one",
                                ),
                            ),
                        ]
                    ),
                    [config.TIME_COL],
                )
            )

        if self.numeric_cols:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("scaler", MinMaxScaler()),
                            (
                                "to_f32",
                                FunctionTransformer(
                                    lambda x: x.astype(np.float32),
                                    feature_names_out="one-to-one",
                                ),
                            ),
                        ]
                    ),
                    self.numeric_cols,
                )
            )

        if self.categorical_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant", fill_value="Undefined"
                                ),
                            ),
                            (
                                "encoder",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                ),
                            ),
                            (
                                "shift_plus1",
                                FunctionTransformer(
                                    lambda x: (x + 1).astype(np.int64),
                                    feature_names_out="one-to-one",
                                ),
                            ),
                        ]
                    ),
                    self.categorical_cols,
                )
            )

        if self.id_cols:
            transformers.append(
                (
                    "id",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(strategy="constant", fill_value=-1),
                            ),
                            (
                                "encoder",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                ),
                            ),
                            (
                                "shift_plus1",
                                FunctionTransformer(
                                    lambda x: (x + 1).astype(np.int64),
                                    feature_names_out="one-to-one",
                                ),
                            ),
                        ]
                    ),
                    self.id_cols,
                )
            )

        if self.text_cols:
            pass

        transformers.append(("rating_raw", "passthrough", [config.RATING_COL]))
        transformers.append(("relevant_raw", "passthrough", [config.RELEVANT_COL]))

        ct = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)

        return Pipeline(steps=[("preprocessor", ct)])

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        # TODO: Remove this in the furure, the idea is to use all features
        feats = (
            self.numeric_cols
            + self.categorical_cols
            # + self.text_cols
            + self.id_cols
            + [config.RATING_COL, config.RELEVANT_COL]
        )

        if self.has_time:
            feats.append(config.TIME_COL)

        train_processed = self.pipeline.fit_transform(train_df[feats])
        val_processed = self.pipeline.transform(val_df[feats])

        test_processed = None
        if test_df is not None:
            test_processed = self.pipeline.transform(test_df[feats])

        for df_p in [train_processed, val_processed, test_processed]:
            if df_p is not None:
                df_p.columns = [c.split("__")[-1] for c in df_p.columns]
                df_p[config.RATING_COL] = df_p[config.RATING_COL].astype(np.float32)
                df_p[config.RELEVANT_COL] = df_p[config.RELEVANT_COL].astype(bool)

        assert isinstance(train_processed, pd.DataFrame)
        assert isinstance(val_processed, pd.DataFrame)
        if test_processed is not None:
            assert isinstance(test_processed, pd.DataFrame)

        return train_processed, val_processed, test_processed

    def split_entity_feature_frames(
        self, processed_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        user_feat_cols = [
            c
            for c in processed_df.columns
            if c.startswith("user_") and c != config.USER_COL
        ]
        item_feat_cols = [
            c
            for c in processed_df.columns
            if c.startswith("item_") and c != config.ITEM_COL
        ]

        user_df = (
            processed_df[[config.USER_COL] + user_feat_cols]
            .drop_duplicates(subset=[config.USER_COL])
            .sort_values(config.USER_COL)
            .reset_index(drop=True)
        )
        item_df = (
            processed_df[[config.ITEM_COL] + item_feat_cols]
            .drop_duplicates(subset=[config.ITEM_COL])
            .sort_values(config.ITEM_COL)
            .reset_index(drop=True)
        )

        return user_df, item_df

    @staticmethod
    def to_torch_feature_matrix(
        df: pd.DataFrame,
        id_col: str,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        feature_cols = [c for c in df.columns if c != id_col]
        features_np = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        feature_matrix = torch.as_tensor(features_np, dtype=torch.float32)

        if device is not None:
            feature_matrix = feature_matrix.to(device)

        return feature_matrix

    def build_entity_tensors(
        self,
        processed_df: pd.DataFrame,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_df, item_df = self.split_entity_feature_frames(processed_df)
        return (
            self.to_torch_feature_matrix(
                user_df, id_col=config.USER_COL, device=device
            ),
            self.to_torch_feature_matrix(
                item_df, id_col=config.ITEM_COL, device=device
            ),
        )
