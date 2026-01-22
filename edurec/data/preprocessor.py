import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OrdinalEncoder

from .. import config


class Preprocessor:
    def __init__(self):
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.categorical_lengths: dict[str, int] = {}
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
                self.categorical_lengths[col] = int(self.train_df[col].nunique())

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
        self.train_df = train_df.copy()
        self.val_df = val_df.copy()
        self.test_df = test_df.copy() if test_df is not None else None

        self.preprocessor = self._build_preprocessor()
        feats = np.array(self.numeric_cols + self.categorical_cols, dtype=np.str_)

        self.preprocessor.fit(self.train_df[feats])

        train_features = self.preprocessor.transform(self.train_df[feats])
        val_features = self.preprocessor.transform(self.val_df[feats])
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
            test_features = self.preprocessor.transform(self.test_df[feats])

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
