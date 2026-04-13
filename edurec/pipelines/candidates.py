from __future__ import annotations

import pandas as pd
import torch

from .. import config
from ..datasets import ElearningDataModule, History, Phase
from ..recsys.retrieval_engine import Retrieval

_CANDIDATE_IDS_COL = "candidate_ids"
_CANDIDATE_LABELS_COL = "candidate_labels"
_POSITIVE_POSITION_COL = "positive_position"


@torch.no_grad()
def generate_candidates(
    retrieval: Retrieval,
    dm: ElearningDataModule,
    top_n: int,
    batch_size: int | None = None,
) -> dict[str, pd.DataFrame | None]:
    if top_n <= 0:
        raise RuntimeError("top_n must be greater than 0.")

    dm.setup(phase=Phase.RETRIEVAL)

    if dm.u_static_feats is None or dm.i_static_feats is None:
        raise RuntimeError("Static features must be available to generate candidates.")

    retrieval.eval()

    effective_batch_size = batch_size or dm.batch_size
    split_candidates = {
        split: _generate_split_topn_candidates(
            retrieval=retrieval,
            interactions=getattr(dm.artifacts, split),
            history=dm.history_prefixes_by_split[split],
            top_n=top_n,
            batch_size=effective_batch_size,
            u_static_feats=dm.u_static_feats,
            i_static_feats=dm.i_static_feats,
        )
        for split in ("train", "val", "test")
    }

    dm.artifacts.train = split_candidates["train"]
    dm.artifacts.val = split_candidates["val"]
    dm.artifacts.test = split_candidates["test"]

    return split_candidates


@torch.no_grad()
def _generate_split_topn_candidates(
    retrieval: Retrieval,
    interactions: pd.DataFrame | None,
    history: History,
    top_n: int,
    batch_size: int,
    u_static_feats: torch.Tensor,
    i_static_feats: torch.Tensor,
) -> pd.DataFrame | None:
    if interactions is None:
        return None

    df = interactions.reset_index(drop=True).copy()
    if df.empty:
        df[_CANDIDATE_IDS_COL] = [[] for _ in range(len(df))]
        df[_CANDIDATE_LABELS_COL] = [[] for _ in range(len(df))]
        df[_POSITIVE_POSITION_COL] = []
        return df

    device = next(retrieval.parameters()).device
    u_static_feats = u_static_feats.to(device)
    i_static_feats = i_static_feats.to(device)
    num_items = i_static_feats.size(0)
    effective_k = min(top_n, num_items)

    if effective_k <= 0:
        raise RuntimeError("The item catalog must contain at least one item.")

    all_item_embs = retrieval.model.encode_all_items(i_static_feats)

    candidate_ids: list[list[int]] = []
    candidate_labels: list[list[float]] = []
    positive_positions: list[int] = []

    user_ids = torch.as_tensor(
        df[config.USER_COL].to_numpy(copy=True), dtype=torch.long
    )
    item_ids = torch.as_tensor(
        df[config.ITEM_COL].to_numpy(copy=True), dtype=torch.long
    )
    targets = torch.as_tensor(
        df[config.RELEVANT_COL].astype("float32").to_numpy(copy=True),
        dtype=torch.float32,
    )

    for start in range(0, len(df), batch_size):
        stop = min(start + batch_size, len(df))
        batch_slice = slice(start, stop)

        query_emb = retrieval.model.encode_query(
            user_ids=user_ids[batch_slice].to(device),
            history_items=history.items[batch_slice].to(device),
            history_ctx=history.ctx[batch_slice].to(device),
            history_valid_mask=history.valid_mask[batch_slice].to(device),
            u_static_feats=u_static_feats,
            i_static_feats=i_static_feats,
        )
        ranked_items = (
            (query_emb @ all_item_embs.T).argsort(dim=1, descending=True).cpu()
        )

        for offset, ranked_row in enumerate(ranked_items):
            row_idx = start + offset
            row_candidates = _select_candidate_ids(
                ranked_items=ranked_row.tolist(),
                history_items=history.items[row_idx],
                history_valid_mask=history.valid_mask[row_idx],
                target_item_id=int(item_ids[row_idx]),
                target=float(targets[row_idx]),
                top_n=effective_k,
            )
            row_labels, positive_position = _build_candidate_labels(
                candidate_ids=row_candidates,
                target_item_id=int(item_ids[row_idx]),
                target=float(targets[row_idx]),
            )

            candidate_ids.append(row_candidates)
            candidate_labels.append(row_labels)
            positive_positions.append(positive_position)

    df[_CANDIDATE_IDS_COL] = candidate_ids
    df[_CANDIDATE_LABELS_COL] = candidate_labels
    df[_POSITIVE_POSITION_COL] = positive_positions
    return df


def _select_candidate_ids(
    ranked_items: list[int],
    history_items: torch.Tensor,
    history_valid_mask: torch.Tensor,
    target_item_id: int,
    target: float,
    top_n: int,
) -> list[int]:
    selected: list[int] = []
    selected_set: set[int] = set()
    seen_items = _history_to_seen_items(history_items, history_valid_mask)

    for item_id in ranked_items:
        if item_id in selected_set or item_id in seen_items:
            continue
        selected.append(item_id)
        selected_set.add(item_id)
        if len(selected) == top_n:
            break

    if len(selected) < top_n:
        for item_id in ranked_items:
            if item_id in selected_set:
                continue
            selected.append(item_id)
            selected_set.add(item_id)
            if len(selected) == top_n:
                break

    if target > 0 and target_item_id not in selected_set:
        selected[-1] = target_item_id

    return selected


def _history_to_seen_items(
    history_items: torch.Tensor, history_valid_mask: torch.Tensor
) -> set[int]:
    valid_items = history_items[history_valid_mask]
    return {int(item_id) - 1 for item_id in valid_items.tolist() if int(item_id) > 0}


def _build_candidate_labels(
    candidate_ids: list[int],
    target_item_id: int,
    target: float,
) -> tuple[list[float], int]:
    labels = [0.0] * len(candidate_ids)

    if target <= 0:
        return labels, 0

    positive_position = candidate_ids.index(target_item_id)
    labels[positive_position] = float(target)
    return labels, positive_position
