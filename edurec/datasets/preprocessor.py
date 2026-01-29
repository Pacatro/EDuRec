import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OrdinalEncoder

from .. import config


class Preprocessor:
    def __init__(
        self, numeric_cols: list[str], categorical_cols: list[str], id_cols: list[str]
    ):
        self.numeric_cols: list[str] = numeric_cols
        self.id_cols: list[str] = id_cols
        self.categorical_cols: list[str] = categorical_cols
        self.id_cols = [config.USER_COL, config.ITEM_COL]
        self.preprocessor: Pipeline | None = None

    def _build_preprocessor(self) -> Pipeline:
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

    def _generate_neg_samples(
        self, df: pd.DataFrame, neg_samples: int, min_rating: float
    ) -> pd.DataFrame:
        new_data = []
        user_item_set = (
            df.groupby(config.USER_COL)[config.ITEM_COL].apply(set).to_dict()
        )

        # TODO: We shoud have a way to differenciate between ITEM features and USER features
        # for now, we assome that all other features are ITEM features
        item_features_map = (
            df.drop_duplicates(config.ITEM_COL)
            .set_index(config.ITEM_COL)
            .to_dict(orient="index")
        )

        all_items = df[config.ITEM_COL].unique()
        columns = df.columns.tolist()

        user_idx = columns.index(config.USER_COL)
        item_idx = columns.index(config.ITEM_COL)
        rating_idx = columns.index(config.RATING_COL)

        for row in df.itertuples(index=False):
            row_list = list(row)
            new_data.append(row_list)

            user_id = row_list[user_idx]
            negatives_found = 0

            while negatives_found < neg_samples:
                neg_id = np.random.choice(all_items)

                if neg_id not in user_item_set[user_id]:
                    neg_row = row_list.copy()

                    neg_row[item_idx] = neg_id
                    neg_row[rating_idx] = min_rating

                    if neg_id in item_features_map:
                        item_attrs = item_features_map[neg_id]
                        for col_name, col_value in item_attrs.items():
                            if col_name in columns:
                                neg_row[columns.index(str(col_name))] = col_value

                    new_data.append(neg_row)
                    negatives_found += 1

        return pd.DataFrame(new_data, columns=columns)

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame | None,
        min_rating: float,
        neg_samples: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        if neg_samples:
            self.train_df = self._generate_neg_samples(
                train_df, config.TRAIN_NEG_SAMPLES, min_rating
            )
            self.val_df = self._generate_neg_samples(
                val_df, config.VAL_NEG_SAMPLES, min_rating
            )
            self.test_df = (
                self._generate_neg_samples(test_df, config.TEST_NEG_SAMPLES, min_rating)
                if test_df is not None
                else None
            )
        else:
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
