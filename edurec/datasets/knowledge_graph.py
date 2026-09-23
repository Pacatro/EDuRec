import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import coalesce, remove_self_loops

from .. import settings
from .dataprocessor import DataProcessor, _coerce_list_tokens, _is_missing

EdgeType = tuple[str, str, str]

_ENTITIES = (
    ("users", "user", settings.USER_COL),
    ("items", "item", settings.ITEM_COL),
)


def build_knowledge_graph(
    interactions: pd.DataFrame,
    user_frame: pd.DataFrame,
    item_frame: pd.DataFrame,
    user_feats: torch.Tensor,
    item_feats: torch.Tensor,
    processor: DataProcessor,
) -> HeteroData:
    """Build the heterogeneous knowledge graph from already-processed data.

    Nodes are users, items and one attribute node type per categorical or
    list-valued field (``attr::<field>``). Attribute node types are shared
    across users and items when the field name matches. Extra relations are
    declared in the dataset schema (``refs`` and ``cooc``) and resolved by the
    same row-wise primitive. PyG handles reverse edges and edge cleanup.
    """
    data = HeteroData()
    data["user"].x = user_feats
    data["item"].x = item_feats

    frames = {"users": user_frame, "items": item_frame}
    node_ids = _node_ids(frames, processor)

    _add_interaction_edges(data, interactions)
    vocab = _add_attribute_edges(data, frames, node_ids, processor)
    _add_reference_edges(data, frames, node_ids, processor)
    _add_cooccurrence_edges(data, frames, node_ids, processor, vocab)

    _clean_edges(data)
    return ToUndirected()(data)


def _node_ids(
    frames: dict[str, pd.DataFrame],
    processor: DataProcessor,
) -> dict[str, pd.Series]:
    id_cols = {"users": settings.USER_COL, "items": settings.ITEM_COL}
    id_maps = {"users": processor.user_id_map, "items": processor.item_id_map}
    return {
        prefix: frames[prefix][id_cols[prefix]].map(id_maps[prefix])
        for prefix in frames
    }


def _add_interaction_edges(data: HeteroData, interactions: pd.DataFrame) -> None:
    if settings.RELEVANT_COL in interactions.columns:
        interactions = interactions.loc[interactions[settings.RELEVANT_COL] > 0]

    users = interactions[settings.USER_COL].to_numpy(dtype=np.int64)
    items = interactions[settings.ITEM_COL].to_numpy(dtype=np.int64)
    valid = (users >= 0) & (items >= 0)
    if not valid.any():
        return

    data["user", "interacts", "item"].edge_index = torch.as_tensor(
        np.stack([users[valid], items[valid]]), dtype=torch.long
    )


def _field_kinds(processor: DataProcessor) -> dict[str, bool]:
    """Map every user/item feature field to whether it is list-valued."""
    kinds: dict[str, bool] = {}
    for prefix, _, _ in _ENTITIES:
        groups = processor.column_groups.get(prefix, {})
        for col in groups.get("categorical", []):
            kinds.setdefault(col, False)
        for col in groups.get("list", []):
            kinds[col] = True
    return kinds


def _add_attribute_edges(
    data: HeteroData,
    frames: dict[str, pd.DataFrame],
    node_ids: dict[str, pd.Series],
    processor: DataProcessor,
) -> dict[str, dict[str, int]]:
    """Create one attribute node type per field and return its vocabulary."""
    vocab: dict[str, dict[str, int]] = {}

    for field, is_list in _field_kinds(processor).items():
        rows: list[pd.DataFrame] = []
        for prefix, node_type, _ in _ENTITIES:
            groups = processor.column_groups.get(prefix, {})
            if field not in groups.get("categorical", []) and field not in groups.get(
                "list", []
            ):
                continue

            tokens = frames[prefix][field].map(
                _coerce_list_tokens if is_list else _single_token
            )
            long = pd.DataFrame(
                {
                    "entity": node_type,
                    "node": node_ids[prefix].to_numpy(),
                    "token": tokens.to_numpy(),
                }
            )
            rows.append(long)

        if not rows:
            continue

        long = pd.concat(rows, ignore_index=True)
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
        for entity in ("user", "item"):
            mask = (long["entity"] == entity).to_numpy()
            if not mask.any():
                continue
            data[entity, f"has::{field}", attr_node].edge_index = torch.as_tensor(
                np.stack([long["node"].to_numpy()[mask], codes[mask]]),
                dtype=torch.long,
            )

    return vocab


def _add_reference_edges(
    data: HeteroData,
    frames: dict[str, pd.DataFrame],
    node_ids: dict[str, pd.Series],
    processor: DataProcessor,
) -> None:
    for prefix, node_type, _ in _ENTITIES:
        for field, target_field in processor.schema.get(prefix, {}).get("refs", {}).items():
            frame = frames[prefix]
            if field not in frame.columns or target_field not in frame.columns:
                continue

            lookup: dict[str, int] = {}
            for value, node in zip(frame[target_field], node_ids[prefix]):
                key = _normalize_label(value)
                if key and not pd.isna(node):
                    lookup.setdefault(key, int(node))

            sources: list[int] = []
            targets: list[int] = []
            for node, value in zip(node_ids[prefix], frame[field]):
                if pd.isna(node) or node < 0:
                    continue
                for token in _coerce_list_tokens(value):
                    target = lookup.get(_normalize_label(token))
                    if target is not None:
                        sources.append(int(node))
                        targets.append(target)

            if sources:
                data[node_type, f"ref::{field}", node_type].edge_index = torch.as_tensor(
                    np.stack([sources, targets]), dtype=torch.long
                )


def _add_cooccurrence_edges(
    data: HeteroData,
    frames: dict[str, pd.DataFrame],
    node_ids: dict[str, pd.Series],
    processor: DataProcessor,
    vocab: dict[str, dict[str, int]],
) -> None:
    for prefix, _, _ in _ENTITIES:
        for left, right in processor.schema.get(prefix, {}).get("cooc", []):
            if left not in vocab or right not in vocab:
                continue

            frame = frames[prefix]
            left_codes = _row_codes(frame[left], vocab[left])
            right_codes = _row_codes(frame[right], vocab[right])
            rows = node_ids[prefix].to_numpy()
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
        edge_index, _ = remove_self_loops(data[edge_type].edge_index)
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
