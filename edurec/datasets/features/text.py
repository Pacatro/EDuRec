from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from edurec.datasets.features.parsing import (
    as_float_array,
    clean_text,
    coerce_list_tokens,
    single_token,
)

if TYPE_CHECKING:
    from edurec.datasets.features.tabular import DataProcessor


def encode_documents(
    docs: list[str],
    model_name: str,
    dim: int,
    batch_size: int,
    max_tokens: int,
) -> np.ndarray:
    """Return normalized float32 embeddings [num_documents, dim]."""
    if not docs:
        return np.empty((0, dim), dtype=np.float32)
    model = _get_sentence_embedding_model(model_name)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_tokens

    embeddings = as_float_array(
        model.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ),
        n_rows=len(docs),
    )
    if embeddings.shape[1] != dim:
        raise RuntimeError(
            "Sentence embedding output shape does not match configured dim."
        )

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms


def graph_node_features(
    processor: "DataProcessor",
    items: pd.DataFrame,
    entities: dict[str, pd.DataFrame] | None,
) -> dict[str, pd.DataFrame]:
    """Encode each retained graph entity once, independent of interactions."""
    node_features: dict[str, pd.DataFrame] = {}
    if "text" not in processor.active_feature_types:
        return node_features

    entities = entities or {}
    pipeline = processor.pipelines.get("items")
    groups = {} if pipeline is None else pipeline.groups
    item_schema = processor.schema.get("items", {})
    list_fields = set(groups.get("list", [])) | set(item_schema.get("graph_list", []))
    enabled_fields = list_fields | set(groups.get("categorical", []))
    for graph_field, columns in item_schema.get("graph_text", {}).items():
        if graph_field not in enabled_fields or graph_field not in items:
            continue
        # Scalar categories may contain commas; only split declared lists.
        values = items[graph_field].map(
            coerce_list_tokens if graph_field in list_fields else single_token
        )
        ids = sorted({token for tokens in values for token in tokens if token})
        if not ids:
            continue
        if graph_field in entities:
            frame = entities[graph_field].set_index("id").reindex(ids)
        elif columns == ["name"]:
            # Human-readable labels already identify these attribute nodes.
            frame = pd.DataFrame({"name": ids}, index=ids)
        else:
            raise ValueError(f"Missing graph entity metadata for {graph_field}")
        docs = [
            " [SEP] ".join(
                f"{col}: {clean_text(value)}"
                for col, value in zip(columns, row, strict=True)
                if clean_text(value)
            )
            for row in frame[columns].itertuples(index=False, name=None)
        ]
        # E5 recommends the query prefix when embeddings are used as features.
        if "e5" in processor.text_embedding_model.lower():
            docs = [f"query: {doc}" if doc else "" for doc in docs]
        vectors = np.zeros((len(ids), processor.text_embedding_dim), dtype=np.float32)
        present = [index for index, doc in enumerate(docs) if doc]
        if present:
            vectors[present] = processor.encode_text_documents(
                [docs[index] for index in present]
            )
        node_features[graph_field] = pd.DataFrame(vectors, index=ids)
    return node_features


@cache
def _get_sentence_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for text preprocessing."
        ) from exc
    return SentenceTransformer(model_name)
