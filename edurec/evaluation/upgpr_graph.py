from dataclasses import dataclass

import torch

from .. import settings
from ..datasets import ElearningDataModule
from ..datasets.dataprocessor import FeatureMetadata


@dataclass(frozen=True)
class KnowledgeGraphData:
    """Compact, typed knowledge graph consumed by UPGPR.

    Users occupy ``[0, num_users)`` and items the following ``num_items``
    nodes. Attribute values are appended afterwards. Every edge is stored in
    both directions with a distinct relation, as required by path reasoning.
    """

    edge_index: torch.Tensor
    edge_type: torch.Tensor
    node_type: torch.Tensor
    num_users: int
    num_items: int
    num_relations: int
    num_node_types: int

    @property
    def num_nodes(self) -> int:
        return int(self.node_type.numel())

    @property
    def item_offset(self) -> int:
        return self.num_users


def build_knowledge_graph(dm: ElearningDataModule) -> KnowledgeGraphData:
    """Build a dataset-independent KG from positive interactions and metadata."""

    train = dm.artifacts.train
    assert train is not None
    num_users = dm.num_users
    num_items = dm.num_items
    node_types = [0] * num_users + [1] * num_items
    num_node_types = 2
    num_relations = 0
    sources: list[int] = []
    targets: list[int] = []
    relations: list[int] = []

    def add_edges(heads: list[int], tails: list[int]) -> None:
        nonlocal num_relations
        if not heads:
            return
        forward = num_relations
        num_relations += 2
        sources.extend(heads)
        targets.extend(tails)
        relations.extend([forward] * len(heads))
        sources.extend(tails)
        targets.extend(heads)
        relations.extend([forward + 1] * len(heads))

    positive = train[train[settings.RELEVANT_COL] > 0]
    interaction_heads = positive[settings.USER_COL].astype(int).tolist()
    interaction_tails = (positive[settings.ITEM_COL].astype(int) + num_users).tolist()
    add_edges(interaction_heads, interaction_tails)

    def add_feature_edges(
        features: torch.Tensor,
        metadata: FeatureMetadata | None,
        owner_offset: int,
    ) -> None:
        nonlocal num_node_types
        if metadata is None:
            return

        dense_cols = metadata.dense_cols
        cat_cols = metadata.categorical_cols
        feature_cols = dense_cols + metadata.text_embedding_cols + cat_cols
        col_to_idx = {name: idx for idx, name in enumerate(feature_cols)}

        # Categorical values become entities scoped by their column. Scoping
        # avoids conflating values such as an unknown language and job.
        for col in cat_cols:
            col_idx = col_to_idx[col]
            values = features[:, col_idx].round().long()
            valid_values = sorted(int(v) for v in values.unique().tolist() if v >= 0)
            value_nodes: dict[int, int] = {}
            type_id = num_node_types
            num_node_types += 1
            for value in valid_values:
                value_nodes[value] = len(node_types)
                node_types.append(type_id)
            heads, tails = [], []
            for owner_id, value in enumerate(values.tolist()):
                if value in value_nodes:
                    heads.append(owner_offset + owner_id)
                    tails.append(value_nodes[value])
            add_edges(heads, tails)

        # Multi-label columns are already one-hot encoded by DataProcessor.
        list_groups: dict[str, list[tuple[int, str]]] = {}
        for idx, col in enumerate(dense_cols):
            if not col.startswith("list__"):
                continue
            _, raw_col, value = col.split("__", maxsplit=2)
            list_groups.setdefault(raw_col, []).append((idx, value))

        for columns in list_groups.values():
            type_id = num_node_types
            num_node_types += 1
            value_nodes = {}
            for _, value in columns:
                value_nodes[value] = len(node_types)
                node_types.append(type_id)
            heads, tails = [], []
            for col_idx, value in columns:
                present = torch.nonzero(features[:, col_idx] > 0.5).flatten().tolist()
                heads.extend(owner_offset + int(idx) for idx in present)
                tails.extend([value_nodes[value]] * len(present))
            add_edges(heads, tails)

    add_feature_edges(
        dm.u_static_feats,
        dm.data_processor.feature_metadata.get("users"),
        0,
    )
    add_feature_edges(
        dm.i_static_feats,
        dm.data_processor.feature_metadata.get("items"),
        num_users,
    )

    return KnowledgeGraphData(
        edge_index=torch.tensor([sources, targets], dtype=torch.long),
        edge_type=torch.tensor(relations, dtype=torch.long),
        node_type=torch.tensor(node_types, dtype=torch.long),
        num_users=num_users,
        num_items=num_items,
        num_relations=num_relations,
        num_node_types=num_node_types,
    )
