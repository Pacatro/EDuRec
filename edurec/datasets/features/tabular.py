from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OrdinalEncoder

from edurec import settings
from edurec.datasets.features import text
from edurec.datasets.features.parsing import (
    as_float_array,
    clean_text,
    coerce_list_tokens,
    is_missing,
)
from edurec.datasets.registry import Schema

PREFIXES = ("users", "items", "inter")
SUPPORTED_FEATURE_TYPES = {"numeric", "categorical", "text", "list"}
IGNORED_FEATURE_TYPES = {"time"}


@dataclass
class ProcessedFeatures:
    users: pd.DataFrame | None
    items: pd.DataFrame | None
    interactions: pd.DataFrame | None
    text_embeddings: dict[str, pd.DataFrame | None]


@dataclass
class FeatureMetadata:
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    categorical_cardinalities: dict[str, int] = field(default_factory=dict)
    text_embedding_cols: list[str] = field(default_factory=list)


class DataProcessor:
    """
    Preprocesses users, items and interactions.

    Inputs and outputs are pandas DataFrames. Text embeddings are returned
    separately from the remaining features. Timestamp columns are ignored.
    """

    def __init__(
        self,
        schema: Schema,
        ct_sparse_threshold: float = 0.0,
        text_embedding_model: str = settings.TEXT_EMBEDDING_MODEL,
        text_embedding_dim: int = settings.TEXT_EMBEDDING_DIM,
        text_embedding_batch_size: int = settings.TEXT_EMBEDDING_BATCH_SIZE,
        text_max_tokens: int = settings.TEXT_MAX_TOKENS,
    ):
        self.schema = schema
        self.ct_sparse_threshold = ct_sparse_threshold
        self.text_embedding_model = text_embedding_model
        self.text_embedding_dim = text_embedding_dim
        self.text_embedding_batch_size = text_embedding_batch_size
        self.text_max_tokens = text_max_tokens
        self.active_feature_types = self._normalize_feature_types(
            settings.PREPROCESS_FEATURE_TYPES
        )

        self.user_id_map: dict[object, int] = {}
        self.item_id_map: dict[object, int] = {}
        self.preprocessors: dict[str, ColumnTransformer | None] = {}
        self.list_binarizers: dict[str, dict[str, MultiLabelBinarizer]] = {}
        self.column_groups: dict[str, dict[str, list[str]]] = {}
        self.feature_metadata: dict[str, FeatureMetadata] = {}
        self.feature_columns: dict[str, list[str]] = {}
        self.text_embedding_columns: dict[str, list[str]] = {}
        # Indexed by original entity ID; persisted with the processor in joblib.
        self.graph_node_features: dict[str, pd.DataFrame] = {}
        self.is_fitted_ = False

    def fit(
        self,
        users_train: pd.DataFrame,
        items_train: pd.DataFrame,
        interactions_train: pd.DataFrame,
    ) -> Self:
        self.graph_node_features = {}
        self.user_id_map = _build_id_map(
            users_train[settings.USER_COL],
            interactions_train[settings.USER_COL],
        )
        self.item_id_map = _build_id_map(
            items_train[settings.ITEM_COL],
            interactions_train[settings.ITEM_COL],
        )

        for prefix, df in {
            "users": users_train,
            "items": items_train,
            "inter": interactions_train,
        }.items():
            self._fit_prefix(prefix, df)

        self.is_fitted_ = True
        return self

    def _fit_prefix(self, prefix: str, df: pd.DataFrame) -> None:
        groups = self._resolve_columns(prefix, df.columns)
        self.column_groups[prefix] = groups
        self.list_binarizers[prefix] = {}

        preprocessor, preprocessor_cols, dense_cols, categorical_cols = (
            self._fit_preprocessor(groups, df)
        )
        self.preprocessors[prefix] = preprocessor

        list_cols = self._fit_list_binarizers(prefix, groups, df)

        text_cols = (
            [f"text__embedding_{idx:03d}" for idx in range(self.text_embedding_dim)]
            if groups["text"]
            else []
        )

        self.feature_columns[prefix] = [
            *groups["passthrough"],
            *preprocessor_cols,
            *list_cols,
        ]
        self.text_embedding_columns[prefix] = text_cols

        self.feature_metadata[prefix] = self._build_metadata(
            groups,
            preprocessor,
            dense_cols,
            categorical_cols,
            text_cols,
        )

    def _fit_preprocessor(
        self,
        groups: dict[str, list[str]],
        df: pd.DataFrame,
    ) -> tuple[ColumnTransformer | None, list[str], list[str], list[str]]:
        transformers = []
        if groups["numeric"]:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("scaler", MinMaxScaler()),
                        ]
                    ),
                    groups["numeric"],
                )
            )
        if groups["categorical"]:
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
                        ]
                    ),
                    groups["categorical"],
                )
            )

        preprocessor = (
            ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                sparse_threshold=self.ct_sparse_threshold,
                verbose_feature_names_out=True,
            )
            if transformers
            else None
        )
        preprocessor_cols: list[str] = []
        dense_cols: list[str] = []
        categorical_cols: list[str] = []
        if preprocessor is not None:
            preprocessor.fit(self._preprocessor_frame(df, groups))
            preprocessor_cols = preprocessor.get_feature_names_out().tolist()
            for name, target in (("num", dense_cols), ("cat", categorical_cols)):
                indices = preprocessor.output_indices_.get(name)
                if isinstance(indices, slice):
                    target.extend(preprocessor_cols[indices])
        return preprocessor, preprocessor_cols, dense_cols, categorical_cols

    def _fit_list_binarizers(
        self,
        prefix: str,
        groups: dict[str, list[str]],
        df: pd.DataFrame,
    ) -> list[str]:
        list_cols: list[str] = []
        for col in groups["list"]:
            binarizer = MultiLabelBinarizer().fit(df[col].map(coerce_list_tokens))
            self.list_binarizers[prefix][col] = binarizer
            list_cols.extend(f"list__{col}__{value}" for value in binarizer.classes_)
        return list_cols

    def _build_metadata(
        self,
        groups: dict[str, list[str]],
        preprocessor: ColumnTransformer | None,
        numeric_cols: list[str],
        categorical_cols: list[str],
        text_cols: list[str],
    ) -> FeatureMetadata:
        metadata = FeatureMetadata(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            text_embedding_cols=text_cols,
        )
        if preprocessor is not None and groups["categorical"]:
            encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            metadata.categorical_cardinalities = {
                col: len(categories) + 1
                for col, categories in zip(
                    categorical_cols,
                    encoder.categories_,
                    strict=True,
                )
            }
        return metadata

    def _resolve_columns(
        self,
        prefix: str,
        available_columns: pd.Index | list[str],
    ) -> dict[str, list[str]]:
        schema_group = self.schema.get(prefix, {})
        available = set(available_columns)
        reserved = {
            settings.USER_COL,
            settings.ITEM_COL,
            settings.RATING_COL,
            settings.RELEVANT_COL,
            settings.TIME_COL,
        }

        def present(cols: list[str]) -> list[str]:
            return [col for col in cols if col in available and col not in reserved]

        numeric_cols = (
            present(list(schema_group.get("num", [])))
            if "numeric" in self.active_feature_types
            else []
        )
        categorical_cols = (
            present([*schema_group.get("bin", []), *schema_group.get("cat", [])])
            if "categorical" in self.active_feature_types
            else []
        )

        return {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "text": (
                present(list(schema_group.get("text", [])))
                if "text" in self.active_feature_types
                else []
            ),
            "list": (
                present(list(schema_group.get("list", [])))
                if "list" in self.active_feature_types
                else []
            ),
            "passthrough": [
                col for col in self._passthrough_cols(prefix) if col in available
            ],
        }

    def _passthrough_cols(self, prefix: str) -> list[str]:
        if prefix == "users":
            return [settings.USER_COL]
        if prefix == "items":
            return [settings.ITEM_COL]
        return [
            settings.USER_COL,
            settings.ITEM_COL,
            settings.RATING_COL,
            settings.RELEVANT_COL,
            settings.TIME_COL,
        ]

    def _preprocessor_frame(
        self,
        df: pd.DataFrame,
        groups: dict[str, list[str]],
    ) -> pd.DataFrame:
        cols = [*groups["numeric"], *groups["categorical"]]
        frame = df[cols].copy() if cols else pd.DataFrame(index=df.index)
        for col in groups["categorical"]:
            values = frame[col].map(
                lambda value: np.nan if is_missing(value) else str(value).strip()
            )
            frame[col] = values.mask(values == "", np.nan)
        return frame

    def transform(
        self,
        users: pd.DataFrame | None = None,
        items: pd.DataFrame | None = None,
        interactions: pd.DataFrame | None = None,
    ) -> ProcessedFeatures:
        if not self.is_fitted_:
            raise RuntimeError("DataProcessor not fitted")

        return ProcessedFeatures(
            users=self._transform("users", users),
            items=self._transform("items", items),
            interactions=self._transform("inter", interactions),
            text_embeddings={
                "users": self._text_embeddings("users", users),
                "items": self._text_embeddings("items", items),
                "inter": self._text_embeddings("inter", interactions),
            },
        )

    def _transform(
        self,
        prefix: str,
        df: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        if df is None:
            return None

        groups = self.column_groups[prefix]
        parts: list[pd.DataFrame] = []

        for col in groups["passthrough"]:
            # Epoch-second timestamps must keep full precision: casting them to
            # float32 quantises gaps of a few seconds to zero near 1.7e9.
            dtype = np.float64 if col == settings.TIME_COL else np.float32
            if col == settings.USER_COL:
                values = df[col].map(self.user_id_map).fillna(-1)
            elif col == settings.ITEM_COL:
                values = df[col].map(self.item_id_map).fillna(-1)
            elif col == settings.TIME_COL:
                values = pd.to_numeric(df[col], errors="coerce")
                if values.isna().any():
                    raise ValueError(
                        f"Temporal column {col!r} contains invalid timestamps."
                    )
            else:
                values = pd.to_numeric(df[col], errors="coerce").fillna(0)
            parts.append(
                pd.DataFrame({col: values.to_numpy(dtype=dtype)}, index=df.index)
            )

        preprocessor = self.preprocessors[prefix]
        if preprocessor is not None:
            parts.append(
                pd.DataFrame(
                    as_float_array(
                        preprocessor.transform(self._preprocessor_frame(df, groups)),
                        n_rows=len(df),
                    ),
                    columns=preprocessor.get_feature_names_out(),
                    index=df.index,
                )
            )

        for col, binarizer in self.list_binarizers[prefix].items():
            tokens = (
                df[col].map(coerce_list_tokens)
                if col in df
                else [[] for _ in range(len(df))]
            )

            transformed_tokens = binarizer.transform(tokens)
            transformed_tokens = np.asarray(transformed_tokens, dtype=np.float32)

            parts.append(
                pd.DataFrame(
                    transformed_tokens,
                    columns=[f"list__{col}__{value}" for value in binarizer.classes_],
                    index=df.index,
                )
            )

        if not parts:
            return pd.DataFrame(index=df.index)

        return pd.concat(parts, axis=1)[self.feature_columns[prefix]]

    def _text_embeddings(
        self,
        prefix: str,
        df: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        if df is None:
            return None

        text_cols = self.column_groups[prefix]["text"]
        embedding_cols = self.text_embedding_columns[prefix]
        if not text_cols:
            return pd.DataFrame(
                index=df.index, columns=embedding_cols, dtype=np.float32
            )

        docs: list[str] = []
        for row in df[text_cols].itertuples(index=False, name=None):
            parts = []
            for col, value in zip(text_cols, row, strict=True):
                cleaned = clean_text(value)
                if cleaned:
                    parts.append(f"{col}: {cleaned}")
            docs.append(" [SEP] ".join(parts))

        return pd.DataFrame(
            self.encode_text_documents(docs),
            columns=embedding_cols,
            index=df.index,
            dtype=np.float32,
        )

    def fit_graph_text_features(
        self,
        items: pd.DataFrame,
        entities: dict[str, pd.DataFrame] | None,
    ) -> None:
        self.graph_node_features = text.graph_node_features(self, items, entities)

    def encode_text_documents(self, docs: list[str]) -> np.ndarray:
        """Return normalized float32 embeddings [num_documents, text_embedding_dim]."""
        return text.encode_documents(
            docs,
            self.text_embedding_model,
            self.text_embedding_dim,
            self.text_embedding_batch_size,
            self.text_max_tokens,
        )

    def _normalize_feature_types(
        self, feature_types: tuple[str, ...]
    ) -> tuple[str, ...]:
        invalid = set(feature_types) - SUPPORTED_FEATURE_TYPES - IGNORED_FEATURE_TYPES
        if invalid:
            raise ValueError(
                f"Unsupported preprocess feature types: {', '.join(sorted(invalid))}"
            )
        return tuple(
            dict.fromkeys(
                feature_type
                for feature_type in feature_types
                if feature_type in SUPPORTED_FEATURE_TYPES
            )
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        processor = joblib.load(path)
        if not isinstance(processor, cls):
            raise TypeError(f"File at {path} is not a {cls.__name__} object")
        return processor


def _build_id_map(*series_list: pd.Series) -> dict[object, int]:
    values = pd.concat(series_list, ignore_index=True).dropna().drop_duplicates()
    return {value: idx for idx, value in enumerate(values.tolist())}
