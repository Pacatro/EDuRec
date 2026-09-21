from dataclasses import dataclass, field

import torch

from .. import settings
from .cache import ProcessedData
from .dataprocessor import DataProcessor

EdgeType = tuple[str, str, str]


@dataclass
class KnowledgeGraph:
    """Heterogeneous knowledge graph derived from a dataset schema.

    Nodes are users, items and one attribute node type per categorical or
    list-valued metadata field. Edges are typed (source, relation,
    destination) triples. Attribute node indices are local to their node type.
    """

    num_users: int
    num_items: int
    node_counts: dict[str, int] = field(default_factory=dict)
    edge_index: dict[EdgeType, torch.Tensor] = field(default_factory=dict)
    node_types: tuple[str, ...] = ()

    @property
    def edge_types(self) -> list[EdgeType]:
        return list(self.edge_index)


def knowledge_graph_metadata(
    processor: DataProcessor,
) -> tuple[dict[str, int], list[EdgeType]]:
    """Return attribute node counts and typed edges for a fitted processor.

    This only inspects the preprocessing metadata, so it can be used to build a
    model configuration without materializing the graph tensors.
    """
    node_counts: dict[str, int] = {}
    edge_types: list[EdgeType] = [
        ("user", "interacts", "item"),
        ("item", "rev_interacts", "user"),
    ]

    for prefix in ("users", "items"):
        node_prefix = "user" if prefix == "users" else "item"
        metadata = processor.feature_metadata.get(prefix)
        if metadata is None:
            continue

        for col in metadata.categorical_cols:
            cardinality = metadata.categorical_cardinalities.get(col, 0) - 1
            if cardinality <= 0:
                continue
            node_type = f"{node_prefix}::{col}"
            node_counts[node_type] = cardinality
            edge_types.append((node_prefix, f"has::{col}", node_type))
            edge_types.append((node_type, f"rev_has::{col}", node_prefix))

        for col in processor.column_groups.get(prefix, {}).get("list", []):
            value_cols = _list_value_cols(metadata.dense_cols, col)
            if not value_cols:
                continue
            node_type = f"{node_prefix}::list::{col}"
            node_counts[node_type] = len(value_cols)
            edge_types.append((node_prefix, f"has::{col}", node_type))
            edge_types.append((node_type, f"rev_has::{col}", node_prefix))

    return node_counts, edge_types


def build_knowledge_graph(
    artifacts: ProcessedData,
    processor: DataProcessor,
) -> KnowledgeGraph:
    """Materialize the schema-driven knowledge graph from processed data."""
    if (
        artifacts.train is None
        or artifacts.u_static_feats is None
        or artifacts.i_static_feats is None
    ):
        raise RuntimeError("Data must be processed before building the graph.")

    node_counts, _ = knowledge_graph_metadata(processor)
    num_users = artifacts.u_static_feats.shape[0]
    num_items = artifacts.i_static_feats.shape[0]

    edge_index: dict[EdgeType, torch.Tensor] = {}

    interactions = artifacts.train
    if settings.RELEVANT_COL in interactions.columns:
        interactions = interactions.loc[interactions[settings.RELEVANT_COL] > 0]

    user_ids = torch.as_tensor(
        interactions[settings.USER_COL].to_numpy(dtype="int64"), dtype=torch.long
    )
    item_ids = torch.as_tensor(
        interactions[settings.ITEM_COL].to_numpy(dtype="int64"), dtype=torch.long
    )
    valid = (user_ids >= 0) & (item_ids >= 0)
    user_ids = user_ids[valid]
    item_ids = item_ids[valid]

    edge_index[("user", "interacts", "item")] = torch.stack([user_ids, item_ids])
    edge_index[("item", "rev_interacts", "user")] = torch.stack([item_ids, user_ids])

    _add_attribute_edges(
        edge_index,
        processor=processor,
        prefix="users",
        node_prefix="user",
        feats=artifacts.u_static_feats,
    )
    _add_attribute_edges(
        edge_index,
        processor=processor,
        prefix="items",
        node_prefix="item",
        feats=artifacts.i_static_feats,
    )

    return KnowledgeGraph(
        num_users=num_users,
        num_items=num_items,
        node_counts=node_counts,
        edge_index=edge_index,
        node_types=("user", "item", *node_counts),
    )


def _add_attribute_edges(
    edge_index: dict[EdgeType, torch.Tensor],
    processor: DataProcessor,
    prefix: str,
    node_prefix: str,
    feats: torch.Tensor,
) -> None:
    metadata = processor.feature_metadata.get(prefix)
    if metadata is None:
        return

    cat_offset = len(metadata.dense_cols) + len(metadata.text_embedding_cols)
    num_rows = feats.shape[0]
    rows = torch.arange(num_rows, device=feats.device)

    for local_idx, col in enumerate(metadata.categorical_cols):
        if metadata.categorical_cardinalities.get(col, 0) <= 1:
            continue
        node_type = f"{node_prefix}::{col}"
        codes = feats[:, cat_offset + local_idx].long()
        valid = codes >= 0
        sources = rows[valid]
        targets = codes[valid]
        edge_index[(node_prefix, f"has::{col}", node_type)] = torch.stack(
            [sources, targets]
        )
        edge_index[(node_type, f"rev_has::{col}", node_prefix)] = torch.stack(
            [targets, sources]
        )

    for col in processor.column_groups.get(prefix, {}).get("list", []):
        value_cols = _list_value_cols(metadata.dense_cols, col)
        if not value_cols:
            continue

        positions = [metadata.dense_cols.index(name) for name in value_cols]
        present = feats[:, positions] > 0.5
        sources, targets = torch.nonzero(present, as_tuple=True)
        node_type = f"{node_prefix}::list::{col}"
        edge_index[(node_prefix, f"has::{col}", node_type)] = torch.stack(
            [sources, targets]
        )
        edge_index[(node_type, f"rev_has::{col}", node_prefix)] = torch.stack(
            [targets, sources]
        )


def _list_value_cols(dense_cols: list[str], col: str) -> list[str]:
    prefix = f"list__{col}__"
    return [name for name in dense_cols if name.startswith(prefix)]
