import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import joblib
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

PREFIXES = ("users", "items", "inter")
ACTIVE_FEATURE_TYPES = ("numeric", "categorical")
INTERACTION_FEATURE_TYPES = (*ACTIVE_FEATURE_TYPES, "text", "list", "time")


@dataclass
class ProcessedFeatures:
    """
    Container for processed DataFrames and the fitted preprocessors used
    to transform them.
    """

    users: pd.DataFrame | None
    items: pd.DataFrame | None
    interactions: pd.DataFrame | None
    preprocessors: dict[str, ColumnTransformer | None]


@dataclass
class FeatureMetadata:
    binary_cols: list[str] = field(default_factory=list)
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    categorical_cardinalities: dict[str, int] = field(default_factory=dict)
    text_cols: list[str] = field(default_factory=list)
    list_cols: list[str] = field(default_factory=list)
    time_cols: list[str] = field(default_factory=list)
    pending_cols: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SchemaColumns:
    binary_cols: list[str] = field(default_factory=list)
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    text_cols: list[str] = field(default_factory=list)
    list_cols: list[str] = field(default_factory=list)

    @property
    def declared_categorical_cols(self) -> list[str]:
        return [*self.binary_cols, *self.categorical_cols]


class DataProcessor:
    """
    Handles the end-to-end preprocessing pipeline for e-learning datasets.

    This class manages separate Scikit-Learn pipelines for users, items, and
    interaction features. It ensures consistent ID encoding across all datasets
    and handles numerical scaling, categorical encoding, and temporal transformations.
    """

    def __init__(
        self,
        schema: Schema,
        ct_sparse_threshold: float = 0.0,
        tfidf_max_features: int = 50_000,
    ):
        self.schema = schema
        self.ct_sparse_threshold = ct_sparse_threshold
        self.tfidf_max_features = tfidf_max_features

        self.user_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self.item_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )

        self._initialize_runtime_state()

    def fit(
        self,
        users_train: pd.DataFrame,
        items_train: pd.DataFrame,
        interactions_train: pd.DataFrame,
    ) -> Self:
        """
        Fits ID encoders and feature pipelines using training data.

        Args:
            users_train: DataFrame containing raw user features.
            items_train: DataFrame containing raw item features.
            interactions_train: DataFrame containing user-item interaction history.

        Returns:
            The fitted DataProcessor instance.
        """
        self._initialize_runtime_state(reset_fitted_state=True)

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

    def _initialize_runtime_state(self, reset_fitted_state: bool = False) -> None:
        self.schema_columns = {
            prefix: self._normalize_schema(prefix) for prefix in PREFIXES
        }

        if reset_fitted_state or not hasattr(self, "preprocessors"):
            self.preprocessors: dict[str, ColumnTransformer | None] = {
                prefix: None for prefix in PREFIXES
            }

        if reset_fitted_state or not hasattr(self, "feature_columns"):
            self.feature_columns: dict[str, list[str]] = {
                prefix: [] for prefix in PREFIXES
            }

        if reset_fitted_state or not hasattr(self, "column_groups"):
            self.column_groups: dict[str, dict[str, list[str]]] = {
                prefix: self._empty_column_groups() for prefix in PREFIXES
            }

        if reset_fitted_state or not hasattr(self, "feature_metadata"):
            self.feature_metadata: dict[str, FeatureMetadata] = {
                prefix: FeatureMetadata() for prefix in PREFIXES
            }

        for prefix in PREFIXES:
            groups = self.column_groups.get(prefix, self._empty_column_groups())
            self.column_groups[prefix] = groups
            self.feature_metadata[prefix] = self._build_feature_metadata(
                prefix=prefix,
                groups=groups,
                categorical_cardinalities=self.feature_metadata[
                    prefix
                ].categorical_cardinalities,
            )

    def _normalize_schema(self, prefix: str) -> SchemaColumns:
        schema_group = self.schema.get(prefix, {})
        return SchemaColumns(
            binary_cols=list(schema_group.get("bin", [])),
            numeric_cols=list(schema_group.get("num", [])),
            categorical_cols=list(schema_group.get("cat", [])),
            text_cols=list(schema_group.get("text", [])),
            list_cols=list(schema_group.get("list", [])),
        )

    def _empty_column_groups(self) -> dict[str, list[str]]:
        return {
            "binary": [],
            "numeric": [],
            "categorical": [],
            "text": [],
            "list": [],
            "time": [],
            "active": [],
            "input": [],
            "passthrough": [],
        }

    def _fit_ct_feats(self, df: pd.DataFrame, prefix: str) -> None:
        """Build and fit feature preprocessors for one feature group."""
        groups = self._resolve_column_groups(
            prefix=prefix, available_columns=df.columns
        )
        df_features = (
            df[groups["input"]].copy()
            if groups["input"]
            else pd.DataFrame(index=df.index)
        )

        preprocessor = self._build_ct(
            prefix=prefix,
            num_cols=groups["numeric"],
            cat_cols=groups["categorical"],
            text_cols=groups["text"],
            list_cols=groups["list"],
            time_col=groups["time"][0] if groups["time"] else None,
        )
        preprocessor.fit(df_features)

        self.column_groups[prefix] = groups
        self.preprocessors[prefix] = preprocessor
        self.feature_columns[prefix] = list(groups["active"])
        self.feature_metadata[prefix] = self._build_feature_metadata(
            prefix=prefix,
            groups=groups,
        )

    def _resolve_column_groups(
        self,
        prefix: str,
        available_columns: pd.Index | list[str],
    ) -> dict[str, list[str]]:
        declared = self.schema_columns[prefix]
        available = set(available_columns)
        reserved = {
            config.USER_COL,
            config.ITEM_COL,
            config.RATING_COL,
            config.RELEVANT_COL,
        }

        binary_cols = [
            col
            for col in declared.binary_cols
            if col in available and col not in reserved
        ]
        numeric_cols = [
            col
            for col in declared.numeric_cols
            if col in available and col not in reserved and col != config.TIME_COL
        ]
        categorical_cols = [
            col
            for col in declared.categorical_cols
            if col in available and col not in reserved
        ]
        text_cols = [
            col
            for col in declared.text_cols
            if col in available and col not in reserved
        ]
        list_cols = [
            col
            for col in declared.list_cols
            if col in available and col not in reserved
        ]
        time_cols = (
            [config.TIME_COL]
            if config.TIME_COL in available and config.TIME_COL not in reserved
            else []
        )

        encoded_categorical_cols = [*binary_cols, *categorical_cols]
        active_cols: list[str] = []
        if self._uses_feature_type(prefix, "numeric"):
            active_cols.extend(numeric_cols)
        if self._uses_feature_type(prefix, "categorical"):
            active_cols.extend(encoded_categorical_cols)

        input_cols = list(active_cols)
        if self._uses_feature_type(prefix, "text"):
            input_cols.extend(text_cols)
        if self._uses_feature_type(prefix, "list"):
            input_cols.extend(list_cols)
        if self._uses_feature_type(prefix, "time"):
            input_cols.extend(time_cols)

        return {
            "binary": binary_cols,
            "numeric": numeric_cols,
            "categorical": encoded_categorical_cols,
            "text": text_cols,
            "list": list_cols,
            "time": time_cols,
            "active": active_cols,
            "input": input_cols,
            "passthrough": self._get_passthrough_cols(prefix),
        }

    def _build_ct(
        self,
        prefix: str,
        num_cols: list[str],
        cat_cols: list[str],
        text_cols: list[str],
        list_cols: list[str],
        time_col: str | None = None,
    ) -> ColumnTransformer:
        """
        Constructs a Scikit-Learn ColumnTransformer with pipelines for all supported
        feature types. Only the active feature types are enabled by default.
        """
        transformers = []

        if time_col and time_col in num_cols:
            num_cols = [c for c in num_cols if c != time_col]

        if num_cols and self._uses_feature_type(prefix, "numeric"):
            transformers.append(("num", self._build_numeric_pipeline(), num_cols))

        if cat_cols and self._uses_feature_type(prefix, "categorical"):
            transformers.append(("cat", self._build_categorical_pipeline(), cat_cols))

        if text_cols and self._uses_feature_type(prefix, "text"):
            transformers.append(
                ("text", self._build_text_pipeline(text_cols), text_cols)
            )

        if list_cols and self._uses_feature_type(prefix, "list"):
            transformers.append(
                ("list", self._build_list_pipeline(list_cols), list_cols)
            )

        if time_col and self._uses_feature_type(prefix, "time"):
            transformers.append(("time", self._build_time_pipeline(), [time_col]))

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=self.ct_sparse_threshold,
            verbose_feature_names_out=False,
        )

    def _uses_feature_type(self, prefix: str, feature_type: str) -> bool:
        active_types = (
            INTERACTION_FEATURE_TYPES if prefix == "inter" else ACTIVE_FEATURE_TYPES
        )
        return feature_type in active_types

    def _build_numeric_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", MinMaxScaler()),
            ]
        )

    def _build_categorical_pipeline(self) -> Pipeline:
        return Pipeline(
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

    def _build_text_pipeline(self, text_cols: list[str]) -> Pipeline:
        return Pipeline(
            [
                ("concat", TextConcatenator(text_cols)),
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.tfidf_max_features,
                        ngram_range=(1, 2),
                    ),
                ),
            ]
        )

    def _build_list_pipeline(self, list_cols: list[str]) -> Pipeline:
        return Pipeline(
            [
                ("concat", ListConcatenator(list_cols)),
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.tfidf_max_features,
                        token_pattern=r"(?u)\b\w+\b",
                    ),
                ),
            ]
        )

    def _build_time_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("time_feats", TimeFeaturesTransformer()),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler()),
            ]
        )

    def transform(
        self,
        users: pd.DataFrame | None = None,
        items: pd.DataFrame | None = None,
        interactions: pd.DataFrame | None = None,
    ) -> ProcessedFeatures:
        """
        Applies fitted transformations to new or existing data.

        Args:
            users: User DataFrame to transform.
            items: Item DataFrame to transform.
            interactions: Interaction DataFrame to transform.

        Returns:
            ProcessedFeatures: Dataclass containing the transformed DataFrames.
        """
        self._initialize_runtime_state()

        if self.preprocessors["users"] is None or self.preprocessors["items"] is None:
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
        groups = self.column_groups[key]

        if ct is None:
            raise RuntimeError(f"DataProcessor not fitted for {key}")

        if df is None:
            raise RuntimeError(f"Input dataframe for {key} cannot be None")

        df_features = (
            df[groups["input"]].copy()
            if groups["input"]
            else pd.DataFrame(index=df.index)
        )
        processed_df = ct.transform(df_features)

        assert isinstance(processed_df, pd.DataFrame)

        passthrough_cols = [col for col in groups["passthrough"] if col in df.columns]
        output_df = pd.concat([df[passthrough_cols].copy(), processed_df], axis=1)

        if config.USER_COL in output_df.columns:
            output_df[config.USER_COL] = self.user_encoder.transform(
                output_df[[config.USER_COL]]
            ).astype("int64")

        if config.ITEM_COL in output_df.columns:
            output_df[config.ITEM_COL] = self.item_encoder.transform(
                output_df[[config.ITEM_COL]]
            ).astype("int64")

        return output_df

    def _build_feature_metadata(
        self,
        prefix: str,
        groups: dict[str, list[str]],
        categorical_cardinalities: dict[str, int] | None = None,
    ) -> FeatureMetadata:
        metadata = FeatureMetadata(
            binary_cols=list(groups["binary"]),
            numeric_cols=list(groups["numeric"]),
            categorical_cols=list(groups["categorical"]),
            categorical_cardinalities=dict(categorical_cardinalities or {}),
            text_cols=list(groups["text"]),
            list_cols=list(groups["list"]),
            time_cols=list(groups["time"]),
            pending_cols=[
                *([] if self._uses_feature_type(prefix, "text") else groups["text"]),
                *([] if self._uses_feature_type(prefix, "list") else groups["list"]),
            ],
        )

        preprocessor = self.preprocessors[prefix]
        if preprocessor is None or not groups["categorical"]:
            return metadata

        ct_named = dict(preprocessor.named_transformers_)
        cat_pipe = ct_named.get("cat")
        if cat_pipe is None:
            return metadata

        encoder = cat_pipe.named_steps["encoder"]
        metadata.categorical_cardinalities = {
            col: len(categories) + 1
            for col, categories in zip(
                groups["categorical"],
                encoder.categories_,
                strict=True,
            )
        }
        return metadata

    def _get_passthrough_cols(self, key: str) -> list[str]:
        if key == "users":
            return [config.USER_COL]

        if key == "items":
            return [config.ITEM_COL]

        return [
            config.USER_COL,
            config.ITEM_COL,
            config.RATING_COL,
            config.RELEVANT_COL,
        ]

    def save(self, path: str | Path) -> None:
        """
        Save the fitted processor state to disk.

        Args:
            path: Target file path (usually with .joblib extension).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Restores a fitted processor from a saved file.

        Args:
            path: Path to the saved .joblib file.

        Returns:
            Self: The restored DataProcessor instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No processor found at {path}")

        processor = joblib.load(path)

        if not isinstance(processor, cls):
            raise TypeError(f"File at {path} is not a {cls.__name__} object")

        processor._initialize_runtime_state()
        return processor


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
                # config.TIME_COL: ts,
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


class ListConcatenator(BaseEstimator, TransformerMixin):
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
        rows = []
        for _, row in Xdf.iterrows():
            tokens: list[str] = []
            for value in row.tolist():
                tokens.extend(_coerce_list_tokens(value))
            rows.append(" ".join(tokens))
        return np.array(rows)


def _coerce_list_tokens(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        separators = [",", ";", "|"]
        for separator in separators:
            if separator in stripped:
                return [
                    token.strip()
                    for token in stripped.split(separator)
                    if token.strip()
                ]
        return [stripped]

    return [str(value).strip()]
