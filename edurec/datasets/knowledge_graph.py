from collections.abc import Iterator

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .. import settings
from .dataprocessor import DataProcessor

EdgeType = tuple[str, str, str]

_INTERACTION_EDGE: EdgeType = ("user", "interacts", "item")
_REVERSE_INTERACTION_EDGE: EdgeType = ("item", "rev_interacts", "user")
_NODE_PREFIXES = (("users", "user"), ("items", "item"))


def knowledge_graph_metadata(
    processor: DataProcessor,
) -> tuple[dict[str, int], list[EdgeType]]:
    """Return attribute node counts and typed edges for a fitted processor.

    This only inspects the preprocessing metadata, so it can be used to build a
    model configuration without materializing the graph tensors.
    """
    node_counts: dict[str, int] = {}
    edge_types: list[EdgeType] = [_INTERACTION_EDGE, _REVERSE_INTERACTION_EDGE]

    for prefix, node_prefix in _NODE_PREFIXES:
        for node_type, relation, count, _, _ in _attribute_nodes(
            processor, prefix, node_prefix
        ):
            node_counts[node_type] = count
            edge_types.append((node_prefix, relation, node_type))
            edge_types.append((node_type, f"rev_{relation}", node_prefix))

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
        yield (
            f"{node_prefix}::{col}",
            f"has::{col}",
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
        yield f"{node_prefix}::list::{col}", f"has::{col}", len(columns), columns, True


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

    graph[_INTERACTION_EDGE].edge_index = torch.stack([user_ids, item_ids])
    graph[_REVERSE_INTERACTION_EDGE].edge_index = torch.stack([item_ids, user_ids])


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

        graph[node_prefix, relation, node_type].edge_index = torch.stack(
            [sources, targets]
        )
        graph[node_type, f"rev_{relation}", node_prefix].edge_index = torch.stack(
            [targets, sources]
        )
