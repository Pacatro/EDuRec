from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from edurec import settings
from edurec.datasets.features.lists import graph_field_kinds
from edurec.datasets.features.parsing import (
    as_float_array,
    clean_text,
    coerce_list_tokens,
    single_token,
)


def encode_documents(
    docs: list[str],
    model_name: str = settings.TEXT_EMBEDDING_MODEL,
    dim: int = settings.TEXT_EMBEDDING_DIM,
    batch_size: int = settings.TEXT_EMBEDDING_BATCH_SIZE,
    max_tokens: int = settings.TEXT_MAX_TOKENS,
) -> np.ndarray:
    """Return normalized float32 embeddings [num_documents, dim]."""
    vectors = np.zeros((len(docs), dim), dtype=np.float32)
    present = [index for index, doc in enumerate(docs) if doc]
    if not present:
        return vectors
    documents = [docs[index] for index in present]
    if "e5" in model_name.lower():
        documents = [f"query: {doc}" for doc in documents]
    model = _get_sentence_embedding_model(model_name)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_tokens

    embeddings = as_float_array(
        model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ),
        n_rows=len(documents),
    )
    if embeddings.shape != (len(documents), dim):
        raise RuntimeError(
            "Sentence embedding output shape does not match configured dim."
        )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    vectors[present] = embeddings / norms
    return vectors


def build_documents(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Clean and join fields in schema order, preserving empty documents."""
    docs = []
    for row in frame[columns].itertuples(index=False, name=None):
        parts = []
        for col, value in zip(columns, row, strict=True):
            cleaned = clean_text(value)
            if cleaned:
                parts.append(f"{col}: {cleaned}")
        docs.append(" [SEP] ".join(parts))
    return docs


def encode_graph_text(
    items: pd.DataFrame,
    entities: dict[str, pd.DataFrame] | None,
    schema: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Encode each retained graph entity once, independent of interactions."""
    node_features: dict[str, pd.DataFrame] = {}
    entities = entities or {}
    kinds = graph_field_kinds(schema)
    for graph_field, columns in schema.get("graph_text", {}).items():
        if graph_field not in kinds or graph_field not in items:
            continue
        # Scalar categories may contain commas; only split declared lists.
        values = items[graph_field].map(
            coerce_list_tokens if kinds[graph_field] else single_token
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
        vectors = encode_documents(build_documents(frame, columns))
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
