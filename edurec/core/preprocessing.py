import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest

from . import config


class Preprocessor:
    def __init__(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        self.train_df = train_df.copy()
        self.val_df = val_df.copy()
        self.test_df = test_df.copy()
        self.numeric_cols = []
        self.categorical_cols = []
        self.categorical_lengths = {}
        self.id_cols = [config.USER_COL, config.ITEM_COL]
        self.preprocessor: Pipeline | None = None

    def _get_column_types(self) -> tuple[list[str], list[str], list[str]]:
        numeric_cols = []
        categorical_cols = []
        id_cols = self.id_cols
        exclude_cols = id_cols + [config.TARGET_COL, config.TIME_COL]

        for col in self.train_df.columns:
            if col in exclude_cols:
                continue
            if pd.api.types.is_numeric_dtype(self.train_df[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
                self.categorical_lengths[col] = self.train_df[col].unique()

        return numeric_cols, categorical_cols, id_cols

    def _build_preprocessor(self) -> Pipeline:
        self.numeric_cols, self.categorical_cols, self.id_cols = (
            self._get_column_types()
        )

        transformers = []

        if self.numeric_cols:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("scaler", MinMaxScaler()),
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
                                SimpleImputer(strategy="most_frequent"),
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
                    self.categorical_cols,
                )
            )

        column_transformer = ColumnTransformer(transformers, remainder="passthrough")

        return Pipeline(
            steps=[
                ("preprocessor", column_transformer),
                ("feature_selector", SelectKBest(k=config.SELECTED_K)),
            ]
        )

    def _update_df_features(
        self, df: pd.DataFrame, features: np.ndarray
    ) -> pd.DataFrame:
        for i, col in enumerate(self.numeric_cols + self.categorical_cols):
            df[col] = features[:, i]
        return df

    def fit_transform(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.preprocessor = self._build_preprocessor()
        feats = self.numeric_cols + self.categorical_cols

        self.preprocessor.fit(self.train_df[feats])

        train_features = self.preprocessor.transform(self.train_df[feats])
        val_features = self.preprocessor.transform(self.val_df[feats])
        test_features = self.preprocessor.transform(self.test_df[feats])

        self.train_df = self._update_df_features(
            self.train_df, np.array(train_features)
        )
        self.val_df = self._update_df_features(self.val_df, np.array(val_features))
        self.test_df = self._update_df_features(self.test_df, np.array(test_features))

        for col in feats:
            self.train_df[col] = self.train_df[col].astype(np.float32)
            self.val_df[col] = self.val_df[col].astype(np.float32)
            self.test_df[col] = self.test_df[col].astype(np.float32)

        return self.train_df, self.val_df, self.test_df
