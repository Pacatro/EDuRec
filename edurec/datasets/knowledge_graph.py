import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import coalesce, remove_self_loops

from .. import settings
from .dataprocessor import DataProcessor, _coerce_list_tokens, _is_missing

EdgeType = tuple[str, str, str]


def build_knowledge_graph(
    item_frame: pd.DataFrame,
    item_feats: torch.Tensor,
    processor: DataProcessor,
) -> HeteroData:
    """Build the item-item knowledge graph from already-processed data.

    Nodes are items and one attribute node type per categorical or list-valued
    item field (``attr::<field>``). Each item connects to its attribute values
    through an edge named after the field, so items that share an attribute
    value become neighbours through that shared attribute node. User-item
    interactions are not modeled: the graph only captures item-item relations.
    Extra relations are declared in the dataset schema (``refs`` and ``cooc``)
    and resolved by the same row-wise primitive. PyG handles reverse edges and
    edge cleanup.
    """
    data = HeteroData()
    data["item"].x = item_feats

    node_ids = item_frame[settings.ITEM_COL].map(processor.item_id_map)
    vocab = _add_attribute_edges(data, item_frame, node_ids, processor)
    _add_reference_edges(data, item_frame, node_ids, processor)
    _add_cooccurrence_edges(data, item_frame, node_ids, processor, vocab)

    _clean_edges(data)
    return ToUndirected()(data)


def _field_kinds(processor: DataProcessor) -> dict[str, bool]:
    """Map every item feature field to whether it is list-valued."""
    groups = processor.column_groups.get("items", {})
    kinds: dict[str, bool] = {col: False for col in groups.get("categorical", [])}
    for col in groups.get("list", []):
        kinds[col] = True
    return kinds


def _add_attribute_edges(
    data: HeteroData,
    item_frame: pd.DataFrame,
    node_ids: pd.Series,
    processor: DataProcessor,
) -> dict[str, dict[str, int]]:
    """Create one attribute node type per item field and return its vocabulary."""
    vocab: dict[str, dict[str, int]] = {}

    for field, is_list in _field_kinds(processor).items():
        tokens = item_frame[field].map(
            _coerce_list_tokens if is_list else _single_token
        )
        long = pd.DataFrame({"node": node_ids.to_numpy(), "token": tokens.to_numpy()})
        long = long.explode("token").dropna(subset=["token", "node"])
        long = long.loc[long["node"] >= 0].reset_index(drop=True)
        long["token"] = long["token"].astype(str).str.strip()
        long = long.loc[long["token"] != ""].reset_index(drop=True)
        if long.empty:
            continue

        codes, uniques = pd.factorize(long["token"].to_numpy(), sort=True)
        vocab[field] = {str(token): idx for idx, token in enumerate(uniques)}

        attr_node = f"attr::{field}"
        data[attr_node].num_nodes = len(uniques)
        data["item", field, attr_node].edge_index = torch.as_tensor(
            np.stack([long["node"].to_numpy(), codes]),
            dtype=torch.long,
        )

    return vocab


def _add_reference_edges(
    data: HeteroData,
    item_frame: pd.DataFrame,
    node_ids: pd.Series,
    processor: DataProcessor,
) -> None:
    for field, target_field in processor.schema.get("items", {}).get("refs", {}).items():
        if field not in item_frame.columns or target_field not in item_frame.columns:
            continue

        lookup: dict[str, int] = {}
        for value, node in zip(item_frame[target_field], node_ids):
            key = _normalize_label(value)
            if key and not pd.isna(node):
                lookup.setdefault(key, int(node))

        sources: list[int] = []
        targets: list[int] = []
        for node, value in zip(node_ids, item_frame[field]):
            if pd.isna(node) or node < 0:
                continue
            for token in _coerce_list_tokens(value):
                target = lookup.get(_normalize_label(token))
                if target is not None:
                    sources.append(int(node))
                    targets.append(target)

        if sources:
            data["item", f"ref::{field}", "item"].edge_index = torch.as_tensor(
                np.stack([sources, targets]), dtype=torch.long
            )


def _add_cooccurrence_edges(
    data: HeteroData,
    item_frame: pd.DataFrame,
    node_ids: pd.Series,
    processor: DataProcessor,
    vocab: dict[str, dict[str, int]],
) -> None:
    for left, right in processor.schema.get("items", {}).get("cooc", []):
        if left not in vocab or right not in vocab:
            continue

        left_codes = _row_codes(item_frame[left], vocab[left])
        right_codes = _row_codes(item_frame[right], vocab[right])
        rows = node_ids.to_numpy()
        valid = (left_codes >= 0) & (right_codes >= 0) & (rows >= 0)
        if not valid.any():
            continue

        data[f"attr::{left}", f"cooc::{left}::{right}", f"attr::{right}"].edge_index = (
            torch.as_tensor(
                np.stack([left_codes[valid], right_codes[valid]]),
                dtype=torch.long,
            )
        )


def _row_codes(values: pd.Series, vocab: dict[str, int]) -> np.ndarray:
    return np.array(
        [vocab.get(_normalize_label(value), -1) for value in values], dtype=np.int64
    )


def _clean_edges(data: HeteroData) -> None:
    for edge_type in list(data.edge_types):
        source, _, target = edge_type
        edge_index = data[edge_type].edge_index
        if source == target:
            edge_index, _ = remove_self_loops(edge_index)
        data[edge_type].edge_index = coalesce(edge_index)


def _single_token(value: object) -> list[str]:
    if _is_missing(value):
        return []
    return [str(value).strip()]


def _normalize_label(value: object) -> str:
    if _is_missing(value):
        return ""
    text = str(value).strip().strip("\"'").strip().lower()
    return " ".join(text.split())
