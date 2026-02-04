import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OrdinalEncoder

from .. import config


def global_preprocessing(df: pd.DataFrame) -> None:
    # We need to encode the user and item ids of all dataset
    df[config.USER_COL] = LabelEncoder().fit_transform(df[config.USER_COL])
    df[config.ITEM_COL] = LabelEncoder().fit_transform(df[config.ITEM_COL])

    # Process time column to timestamp format (nanoseconds)
    if config.TIME_COL in df.columns:
        df[config.TIME_COL] = (
            pd.to_datetime(df[config.TIME_COL]).astype(np.int64) // 10**9
        )

    if config.RELEVANT_COL not in df.columns:
        # An item is relevant if its rating is greater or equal than the threshold
        # The threshold is the mean of the ratings of the user
        mean_user_ratings = df[config.USER_COL].map(
            df.groupby(config.USER_COL)[config.RATING_COL].mean()
        )
        df[config.RELEVANT_COL] = df[config.RATING_COL] >= mean_user_ratings


def get_column_types(
    df: pd.DataFrame, id_cols: list[str]
) -> tuple[list[str], dict[str, int]]:
    exclude_cols = id_cols + [config.RATING_COL, config.TIME_COL, config.RELEVANT_COL]
    numeric_cols = []
    categorical_lengths = {}

    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_lengths[col] = int(df[col].nunique())

    return numeric_cols, categorical_lengths


class DataProcessor:
    def __init__(
        self, numeric_cols: list[str], categorical_cols: list[str], id_cols: list[str]
    ):
        self.numeric_cols: list[str] = numeric_cols
        self.id_cols: list[str] = id_cols
        self.categorical_cols: list[str] = categorical_cols
        self.id_cols = [config.USER_COL, config.ITEM_COL]

    @property
    def pipeline(self) -> Pipeline:
        transformers = []

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
                                FunctionTransformer(lambda x: x.astype(np.float32)),
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
                            # Shift to 1-based indexing
                            (
                                "shift_plus1",
                                FunctionTransformer(lambda x: (x + 1).astype(np.int64)),
                            ),
                        ]
                    ),
                    self.categorical_cols,
                )
            )

        return Pipeline(
            steps=[
                ("preprocessor", ColumnTransformer(transformers, remainder="drop")),
            ]
        )

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        # TODO: Set val_df as possible None instead of test_df
        self.train_df = train_df.copy()
        self.val_df = val_df.copy()
        self.test_df = test_df.copy() if test_df is not None else None
        pipeline = self.pipeline

        feats = np.array(self.numeric_cols + self.categorical_cols, dtype=np.str_)

        pipeline.fit(self.train_df[feats])

        train_features = pipeline.transform(self.train_df[feats])
        val_features = pipeline.transform(self.val_df[feats])
        # La salida es num + cat (en ese orden) y 1:1 columnas
        train_processed = pd.DataFrame(
            train_features, columns=feats, index=self.train_df.index
        )

        val_processed = pd.DataFrame(
            val_features, columns=feats, index=self.val_df.index
        )
        self.train_df = self._merge_features(self.train_df, train_processed)
        self.val_df = self._merge_features(self.val_df, val_processed)

        if self.test_df is not None:
            test_features = pipeline.transform(self.test_df[feats])

            test_processed = pd.DataFrame(
                test_features,
                columns=feats,
                index=self.test_df.index,
            )

            self.test_df = self._merge_features(self.test_df, test_processed)

        return self.train_df, self.val_df, self.test_df

    def _merge_features(
        self, original: pd.DataFrame, processed: pd.DataFrame
    ) -> pd.DataFrame:
        result = original.copy()
        for col in self.numeric_cols:
            result[col] = processed[col].astype(np.float32)
        for col in self.categorical_cols:
            result[col] = processed[col].astype(np.int64)
        return result
