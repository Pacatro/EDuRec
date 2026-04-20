import ast
import re
from dataclasses import dataclass, field
from functools import lru_cache
from html import unescape
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
SUPPORTED_FEATURE_TYPES = ("numeric", "categorical", "text", "list", "time")


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
    dense_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    categorical_cardinalities: dict[str, int] = field(default_factory=dict)
    text_cols: list[str] = field(default_factory=list)
    list_cols: list[str] = field(default_factory=list)
    time_cols: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SchemaColumns:
    binary_cols: list[str] = field(default_factory=list)
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    text_cols: list[str] = field(default_factory=list)
    list_cols: list[str] = field(default_factory=list)


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
        text_embedding_model: str = config.TEXT_EMBEDDING_MODEL,
        text_embedding_dim: int = config.TEXT_EMBEDDING_DIM,
        text_embedding_batch_size: int = config.TEXT_EMBEDDING_BATCH_SIZE,
        text_max_tokens: int = config.TEXT_MAX_TOKENS,
    ):
        self.schema = schema
        self.ct_sparse_threshold = ct_sparse_threshold
        self.tfidf_max_features = tfidf_max_features
        self.text_embedding_model = text_embedding_model
        self.text_embedding_dim = text_embedding_dim
        self.text_embedding_batch_size = text_embedding_batch_size
        self.text_max_tokens = text_max_tokens
        self.active_feature_types = self._normalize_feature_types(
            config.PREPROCESS_FEATURE_TYPES
        )

        self.user_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self.item_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )

        self.preprocessors: dict[str, ColumnTransformer | None] = {}

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

        train_dfs = {
            "users": users_train,
            "items": items_train,
            "inter": interactions_train,
        }
        for prefix, df in train_dfs.items():
            (
                self.column_groups[prefix],
                self.preprocessors[prefix],
                self.feature_metadata[prefix],
            ) = self._fit_prefix(prefix, df)

        return self

    def _initialize_runtime_state(self, reset_fitted_state: bool = False):
        self.schema_columns = {
            prefix: self._normalize_schema(prefix) for prefix in PREFIXES
        }

        legacy_states = getattr(self, "prefix_states", {})
        if reset_fitted_state:
            self.preprocessors = {prefix: None for prefix in PREFIXES}
            self.column_groups = {
                prefix: self._empty_column_groups(prefix) for prefix in PREFIXES
            }
            self.feature_metadata = {prefix: FeatureMetadata() for prefix in PREFIXES}
            return

        if not hasattr(self, "preprocessors"):
            self.preprocessors = {
                prefix: getattr(legacy_states.get(prefix), "preprocessor", None)
                for prefix in PREFIXES
            }

        if not hasattr(self, "column_groups"):
            self.column_groups = {
                prefix: self._coerce_column_groups(
                    prefix,
                    getattr(legacy_states.get(prefix), "column_groups", None),
                )
                for prefix in PREFIXES
            }

        if not hasattr(self, "feature_metadata"):
            self.feature_metadata = {
                prefix: getattr(
                    legacy_states.get(prefix),
                    "feature_metadata",
                    FeatureMetadata(),
                )
                for prefix in PREFIXES
            }

        for prefix in PREFIXES:
            self.preprocessors.setdefault(prefix, None)
            self.column_groups[prefix] = self._coerce_column_groups(
                prefix,
                self.column_groups.get(prefix),
            )
            self.feature_metadata.setdefault(prefix, FeatureMetadata())

    def _normalize_schema(self, prefix: str) -> SchemaColumns:
        schema_group = self.schema.get(prefix, {})
        return SchemaColumns(
            binary_cols=list(schema_group.get("bin", [])),
            numeric_cols=list(schema_group.get("num", [])),
            categorical_cols=list(schema_group.get("cat", [])),
            text_cols=list(schema_group.get("text", [])),
            list_cols=list(schema_group.get("list", [])),
        )

    def _empty_column_groups(self, prefix: str) -> dict[str, list[str]]:
        return {
            "binary": [],
            "numeric": [],
            "categorical": [],
            "text": [],
            "list": [],
            "time": [],
            "input": [],
            "passthrough": self._passthrough_cols(prefix),
        }

    def _coerce_column_groups(
        self,
        prefix: str,
        groups: dict[str, list[str]] | None,
    ) -> dict[str, list[str]]:
        merged = self._empty_column_groups(prefix)
        if groups is None:
            return merged
        for key, values in groups.items():
            merged[key] = list(values)
        return merged

    def _fit_prefix(
        self,
        prefix: str,
        df: pd.DataFrame,
    ) -> tuple[dict[str, list[str]], ColumnTransformer, FeatureMetadata]:
        groups = self._resolve_column_groups(
            prefix=prefix, available_columns=df.columns
        )
        df_features = self._prepare_feature_frame(df, groups)
        preprocessor = self._build_ct(
            num_cols=groups["numeric"],
            cat_cols=groups["categorical"],
            text_cols=groups["text"],
            list_cols=groups["list"],
            time_col=groups["time"][0] if groups["time"] else None,
        )
        preprocessor.fit(df_features)
        transformed_df = preprocessor.transform(df_features)

        assert isinstance(transformed_df, pd.DataFrame)

        output_cols = transformed_df.columns.tolist()
        dense_cols: list[str] = []
        categorical_cols: list[str] = []
        for transformer_name, target_cols in (
            ("num", dense_cols),
            ("text", dense_cols),
            ("list", dense_cols),
            ("time", dense_cols),
            ("cat", categorical_cols),
        ):
            indices = preprocessor.output_indices_.get(transformer_name)
            if indices is None:
                continue
            if isinstance(indices, slice):
                target_cols.extend(output_cols[indices])
                continue
            target_cols.extend(output_cols[index] for index in indices)

        return (
            groups,
            preprocessor,
            self._build_feature_metadata(
                groups=groups,
                preprocessor=preprocessor,
                dense_cols=dense_cols,
                categorical_cols=categorical_cols,
            ),
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
            for col in [*declared.binary_cols, *declared.categorical_cols]
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

        input_cols: list[str] = []
        if self._uses_feature_type("numeric"):
            input_cols.extend(numeric_cols)
        if self._uses_feature_type("categorical"):
            input_cols.extend(categorical_cols)
        if self._uses_feature_type("text"):
            input_cols.extend(text_cols)
        if self._uses_feature_type("list"):
            input_cols.extend(list_cols)
        if self._uses_feature_type("time"):
            input_cols.extend(time_cols)

        return {
            "binary": binary_cols,
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "text": text_cols,
            "list": list_cols,
            "time": time_cols,
            "input": input_cols,
            "passthrough": self._passthrough_cols(prefix),
        }

    def _passthrough_cols(self, prefix: str) -> list[str]:
        if prefix == "users":
            return [config.USER_COL]
        if prefix == "items":
            return [config.ITEM_COL]
        return [
            config.USER_COL,
            config.ITEM_COL,
            config.RATING_COL,
            config.RELEVANT_COL,
            config.TIME_COL,
            config.INTERACTION_ORDER_COL,
        ]

    def _prepare_feature_frame(
        self,
        df: pd.DataFrame,
        groups: dict[str, list[str]],
    ) -> pd.DataFrame:
        df_features = (
            df[groups["input"]].copy()
            if groups["input"]
            else pd.DataFrame(index=df.index)
        )
        return self._normalize_categorical_inputs(
            df_features,
            groups["categorical"],
        )

    def _build_ct(
        self,
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

        if num_cols and self._uses_feature_type("numeric"):
            transformers.append(("num", self._build_numeric_pipeline(), num_cols))

        if cat_cols and self._uses_feature_type("categorical"):
            transformers.append(("cat", self._build_categorical_pipeline(), cat_cols))

        if text_cols and self._uses_feature_type("text"):
            transformers.append(
                ("text", self._build_text_pipeline(text_cols), text_cols)
            )

        if list_cols and self._uses_feature_type("list"):
            transformers.append(
                ("list", self._build_list_pipeline(list_cols), list_cols)
            )

        if time_col and self._uses_feature_type("time"):
            transformers.append(("time", self._build_time_pipeline(), [time_col]))

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=self.ct_sparse_threshold,
            verbose_feature_names_out=True,
        )

    def _uses_feature_type(self, feature_type: str) -> bool:
        return feature_type in self.active_feature_types

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

    def _build_text_pipeline(self, text_cols: list[str]) -> BaseEstimator:
        return SentenceEmbeddingTransformer(
            text_cols,
            model_name=self.text_embedding_model,
            embedding_dim=self.text_embedding_dim,
            batch_size=self.text_embedding_batch_size,
            max_tokens=self.text_max_tokens,
        )

    def _build_list_pipeline(self, list_cols: list[str]) -> BaseEstimator:
        return ListConcatenator(
            list_cols,
            max_features=self.tfidf_max_features,
            token_pattern=r"(?u)\b\w+\b",
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

        return ProcessedFeatures(
            users=self._transform("users", users),
            items=self._transform("items", items),
            interactions=self._transform("inter", interactions),
            preprocessors=self.preprocessors,
        )

    def _transform(
        self,
        prefix: str,
        df: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        if df is None:
            return None

        preprocessor = self.preprocessors[prefix]
        if preprocessor is None:
            raise RuntimeError(f"DataProcessor not fitted for {prefix}")

        groups = self.column_groups[prefix]
        processed_df = preprocessor.transform(self._prepare_feature_frame(df, groups))

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

    def _normalize_categorical_inputs(
        self,
        df: pd.DataFrame,
        categorical_cols: list[str],
    ) -> pd.DataFrame:
        if not categorical_cols or df.empty:
            return df

        normalized_df = df.copy()
        for col in categorical_cols:
            if col not in normalized_df.columns:
                continue

            values = normalized_df[col].map(
                lambda value: np.nan if pd.isna(value) else str(value).strip()
            )
            normalized_df[col] = values.mask(values == "", np.nan)

        return normalized_df

    def _build_feature_metadata(
        self,
        groups: dict[str, list[str]],
        preprocessor: ColumnTransformer | None = None,
        dense_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
    ) -> FeatureMetadata:
        resolved_categorical_cols = list(categorical_cols or groups["categorical"])
        metadata = FeatureMetadata(
            binary_cols=list(groups["binary"]),
            numeric_cols=list(groups["numeric"]),
            dense_cols=list(dense_cols or groups["numeric"]),
            categorical_cols=resolved_categorical_cols,
            text_cols=list(groups["text"]),
            list_cols=list(groups["list"]),
            time_cols=list(groups["time"]),
        )

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
                resolved_categorical_cols,
                encoder.categories_,
                strict=True,
            )
        }
        return metadata

    def _normalize_feature_types(
        self, feature_types: tuple[str, ...]
    ) -> tuple[str, ...]:
        invalid_types = [
            feature_type
            for feature_type in feature_types
            if feature_type not in SUPPORTED_FEATURE_TYPES
        ]
        if invalid_types:
            raise ValueError(
                f"Unsupported preprocess feature types: {', '.join(invalid_types)}"
            )
        return tuple(dict.fromkeys(feature_types))

    def save(self, path: str | Path):
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

        if not hasattr(processor, "active_feature_types"):
            processor.active_feature_types = processor._normalize_feature_types(
                config.PREPROCESS_FEATURE_TYPES
            )
        if not hasattr(processor, "text_embedding_model"):
            processor.text_embedding_model = config.TEXT_EMBEDDING_MODEL
        if not hasattr(processor, "text_embedding_dim"):
            processor.text_embedding_dim = config.TEXT_EMBEDDING_DIM
        if not hasattr(processor, "text_embedding_batch_size"):
            processor.text_embedding_batch_size = config.TEXT_EMBEDDING_BATCH_SIZE
        if not hasattr(processor, "text_max_tokens"):
            processor.text_max_tokens = config.TEXT_MAX_TOKENS

        processor._initialize_runtime_state()
        return processor


class TimeFeaturesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None) -> Self:
        _ = X, y
        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        _ = input_features
        return np.array(
            [
                "time_hour",
                "time_dow",
                "time_month",
                "time_hour_sin",
                "time_hour_cos",
                "time_dow_sin",
                "time_dow_cos",
            ],
            dtype=object,
        )

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        X_df = pd.DataFrame(X)
        s = X_df.iloc[:, 0]

        dt = pd.to_datetime(s, errors="coerce", utc=True)

        # timestamp in secods
        # ts = (dt.astype(np.int64) // 10**9).astype("float64")

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


class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cols: list[str],
        model_name: str,
        embedding_dim: int,
        batch_size: int,
        max_tokens: int,
    ):
        self.cols = cols
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.max_tokens = max_tokens

    def fit(self, X, y=None) -> Self:
        _ = X, y
        model = _get_sentence_embedding_model(self.model_name)
        if hasattr(model, "max_seq_length"):
            model.max_seq_length = self.max_tokens
        if hasattr(model, "get_sentence_embedding_dimension"):
            dim = model.get_sentence_embedding_dimension()
            if dim != self.embedding_dim:
                raise RuntimeError(
                    f"Expected text embedding dim {self.embedding_dim}, got {dim}"
                )
        self.is_fitted_ = True
        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        _ = input_features
        return np.array(
            [f"embedding_{idx:03d}" for idx in range(self.embedding_dim)],
            dtype=object,
        )

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError("SentenceEmbeddingTransformer must be fitted first.")

        texts = self._build_documents(X)
        model = _get_sentence_embedding_model(self.model_name)
        if hasattr(model, "max_seq_length"):
            model.max_seq_length = self.max_tokens
        embeddings = model.encode(
            texts.tolist(),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise RuntimeError(
                "Sentence embedding output shape does not match configured dim."
            )

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return embeddings / norms

    def _build_documents(self, X: pd.DataFrame) -> np.ndarray:
        Xdf = (
            pd.DataFrame(X, columns=self.cols)
            if not isinstance(X, pd.DataFrame)
            else X[self.cols]
        )
        rows: list[str] = []
        for _, row in Xdf.iterrows():
            parts = []
            for col, value in zip(self.cols, row.tolist(), strict=True):
                cleaned = _clean_text(value)
                if cleaned:
                    parts.append(f"{col}: {cleaned}")
            text = " [SEP] ".join(parts)
            rows.append(_truncate_text(text, self.max_tokens))
        return np.array(rows, dtype=object)


class ListConcatenator(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cols: list[str],
        max_features: int,
        token_pattern: str,
    ):
        self.cols = cols
        self.max_features = max_features
        self.token_pattern = token_pattern

    def fit(self, X, y=None) -> Self:
        _ = y
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            token_pattern=self.token_pattern,
        )
        self.vectorizer_.fit(self._build_documents(X))
        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        _ = input_features
        return self.vectorizer_.get_feature_names_out()

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.vectorizer_.transform(self._build_documents(X)).toarray()  # type: ignore

    def _build_documents(self, X: pd.DataFrame) -> np.ndarray:
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


@lru_cache(maxsize=None)
def _get_sentence_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for text preprocessing."
        ) from exc

    model = SentenceTransformer(model_name)
    return model


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""

    text = unescape(str(value)).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _truncate_text(text: str, max_tokens: int) -> str:
    if not text:
        return ""

    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])
