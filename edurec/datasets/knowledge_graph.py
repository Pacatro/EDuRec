import re
from collections.abc import Iterator

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .. import settings
from .dataprocessor import DataProcessor

EdgeType = tuple[str, str, str]

_INTERACTION_EDGE: EdgeType = ("user", "interacts", "item")
_NODE_PREFIXES = (("users", "user"), ("items", "item"))

# Characters allowed in MLflow parameter keys. Runs of any other character
# (including ``%`` and ``_``) collapse into a single underscore, which also
# keeps node types free of the double underscores PyG warns about.
_INVALID_NAME_CHARS = re.compile(r"[^0-9A-Za-z.:/-]+")


def _escape_attribute_name(col: str) -> str:
    """Sanitize an attribute column name for use as a PyG/MLflow type name."""
    return _INVALID_NAME_CHARS.sub("_", col)


def _edge_pair(edge: EdgeType) -> tuple[EdgeType, EdgeType]:
    source, relation, target = edge
    return edge, (target, f"rev_{relation}", source)


def _add_edge_pair(
    graph: HeteroData,
    edge: EdgeType,
    sources: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    forward, reverse = _edge_pair(edge)
    graph[forward].edge_index = torch.stack([sources, targets])
    graph[reverse].edge_index = torch.stack([targets, sources])


def knowledge_graph_metadata(
    processor: DataProcessor,
) -> tuple[dict[str, int], list[EdgeType]]:
    """Return attribute node counts and typed edges for a fitted processor.

    This only inspects the preprocessing metadata, so it can be used to build a
    model configuration without materializing the graph tensors.
    """
    node_counts: dict[str, int] = {}
    edge_types: list[EdgeType] = list(_edge_pair(_INTERACTION_EDGE))

    for prefix, node_prefix in _NODE_PREFIXES:
        for node_type, relation, count, _, _ in _attribute_nodes(
            processor, prefix, node_prefix
        ):
            node_counts[node_type] = count
            edge_types.extend(_edge_pair((node_prefix, relation, node_type)))

    return node_counts, edge_types


def _attribute_nodes(
    processor: DataProcessor,
    prefix: str,
    node_prefix: str,
) -> Iterator[tuple[str, str, int, tuple[int, ...], bool]]:
    """Yield ``(node_type, relation, count, columns, is_list)`` per attribute.

    ``columns`` indexes the static feature tensor: a single ordinal-code column
    for categorical attributes and the one-hot columns for list attributes.
    """
    metadata = processor.feature_metadata.get(prefix)
    if metadata is None:
        return

    cat_offset = len(metadata.dense_cols) + len(metadata.text_embedding_cols)
    for local_idx, col in enumerate(metadata.categorical_cols):
        count = metadata.categorical_cardinalities.get(col, 0) - 1
        if count <= 0:
            continue
        name = _escape_attribute_name(col)
        yield (
            f"{node_prefix}::{name}",
            f"has::{name}",
            count,
            (cat_offset + local_idx,),
            False,
        )

    for col in processor.column_groups.get(prefix, {}).get("list", []):
        columns = tuple(
            idx
            for idx, name in enumerate(metadata.dense_cols)
            if name.startswith(f"list__{col}__")
        )
        if not columns:
            continue
        name = _escape_attribute_name(col)
        yield (
            f"{node_prefix}::list::{name}",
            f"has::{name}",
            len(columns),
            columns,
            True,
        )


def add_interaction_edges(
    graph: HeteroData,
    interactions: pd.DataFrame,
) -> None:
    if settings.RELEVANT_COL in interactions.columns:
        interactions = interactions.loc[interactions[settings.RELEVANT_COL] > 0]

    user_ids = torch.as_tensor(
        interactions[settings.USER_COL].to_numpy(dtype="int64"), dtype=torch.long
    )
    item_ids = torch.as_tensor(
        interactions[settings.ITEM_COL].to_numpy(dtype="int64"), dtype=torch.long
    )
    valid = (user_ids >= 0) & (item_ids >= 0)
    user_ids, item_ids = user_ids[valid], item_ids[valid]

    _add_edge_pair(graph, _INTERACTION_EDGE, user_ids, item_ids)


def add_attribute_edges(
    graph: HeteroData,
    processor: DataProcessor,
    prefix: str,
    node_prefix: str,
    feats: torch.Tensor,
) -> None:
    rows = torch.arange(feats.shape[0], device=feats.device)

    for node_type, relation, count, columns, is_list in _attribute_nodes(
        processor, prefix, node_prefix
    ):
        graph[node_type].num_nodes = count
        if is_list:
            present = feats[:, list(columns)] > 0.5
            sources, targets = torch.nonzero(present, as_tuple=True)
        else:
            (column,) = columns
            targets = feats[:, column].long()
            valid = targets >= 0
            sources, targets = rows[valid], targets[valid]

        _add_edge_pair(graph, (node_prefix, relation, node_type), sources, targets)
