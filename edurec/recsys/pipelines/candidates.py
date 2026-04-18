import pandas as pd
import torch

from ... import config
from ...datasets import ElearningDataModule, History, Phase
from ..retrieval import Retrieval


@torch.no_grad()
def generate_candidates(
    retrieval: Retrieval,
    dm: ElearningDataModule,
    top_n: int,
    i_static_feats: torch.Tensor,
    batch_size: int | None = None,
) -> dict[str, pd.DataFrame | None]:
    if top_n <= 0:
        raise RuntimeError("top_n must be greater than 0.")

    dm.setup(phase=Phase.RETRIEVAL)
    retrieval.eval()

    effective_batch_size = batch_size or dm.batch_size
    split_candidates = {
        split: _generate_split_candidates(
            retrieval=retrieval,
            interactions=getattr(dm.artifacts, split),
            history=dm.history_prefixes_by_split[split],
            top_n=top_n,
            batch_size=effective_batch_size,
            i_static_feats=i_static_feats,
        )
        for split in ("train", "val", "test")
    }

    dm.artifacts.train = split_candidates["train"]
    dm.artifacts.val = split_candidates["val"]
    dm.artifacts.test = split_candidates["test"]

    return split_candidates


@torch.no_grad()
def _generate_split_candidates(
    retrieval: Retrieval,
    interactions: pd.DataFrame | None,
    history: History,
    top_n: int,
    batch_size: int,
    i_static_feats: torch.Tensor,
) -> pd.DataFrame | None:
    if interactions is None:
        return None

    df = interactions.reset_index(drop=True).copy()
    if df.empty:
        df[config.CANDIDATE_IDS_COL] = [[] for _ in range(len(df))]
        df[config.CANDIDATE_LABELS_COL] = [[] for _ in range(len(df))]
        df[config.POSITIVE_POSITION_COL] = [0 for _ in range(len(df))]
        return df

    device = retrieval.device
    num_items = int(i_static_feats.size(0))
    effective_k = min(top_n, num_items)

    if effective_k == 0:
        raise RuntimeError("The item catalog must contain at least one item.")

    all_item_ids = torch.arange(num_items, device=device, dtype=torch.long)
    all_item_embs = retrieval.encode_items(all_item_ids)
    pool_size = min(num_items, effective_k + history.items.size(1))

    user_ids = torch.as_tensor(
        df[config.USER_COL].to_numpy(copy=True),
        dtype=torch.long,
    )
    target_item_ids = torch.as_tensor(
        df[config.ITEM_COL].to_numpy(copy=True),
        dtype=torch.long,
    )
    targets = torch.as_tensor(
        df[config.RELEVANT_COL].astype("float32").to_numpy(copy=True),
        dtype=torch.float32,
    )

    candidate_ids: list[list[int]] = []
    candidate_labels: list[list[float]] = []
    positive_positions: list[int] = []

    for start in range(0, len(df), batch_size):
        stop = min(start + batch_size, len(df))
        batch_slice = slice(start, stop)

        query_emb = retrieval.encode_query(
            user_ids=user_ids[batch_slice].to(device),
            history_items=history.items[batch_slice].to(device),
            history_ctx=history.ctx[batch_slice].to(device),
            history_valid_mask=history.valid_mask[batch_slice].to(device),
        )
        scores = query_emb @ all_item_embs.T
        top_item_ids = scores.topk(k=pool_size, dim=1).indices.cpu()

        for offset, top_items in enumerate(top_item_ids):
            row_idx = start + offset
            seen_items = _history_to_seen_items(
                history_items=history.items[row_idx],
                history_valid_mask=history.valid_mask[row_idx],
            )

            row_candidates = _select_candidate_ids(
                top_items=top_items.tolist(),
                seen_items=seen_items,
                top_n=effective_k,
            )

            if len(row_candidates) < effective_k:
                full_ranking = scores[offset].argsort(dim=0, descending=True).cpu()
                row_candidates = _fill_missing_candidates(
                    candidate_ids=row_candidates,
                    ranked_items=full_ranking.tolist(),
                    seen_items=seen_items,
                    top_n=effective_k,
                )

            row_candidates = _ensure_positive_candidate(
                candidate_ids=row_candidates,
                target_item_id=int(target_item_ids[row_idx]),
                target=float(targets[row_idx]),
                top_n=effective_k,
            )

            row_labels, positive_position = _build_candidate_labels(
                candidate_ids=row_candidates,
                target_item_id=int(target_item_ids[row_idx]),
                target=float(targets[row_idx]),
            )

            candidate_ids.append(row_candidates)
            candidate_labels.append(row_labels)
            positive_positions.append(positive_position)

    df[config.CANDIDATE_IDS_COL] = candidate_ids
    df[config.CANDIDATE_LABELS_COL] = candidate_labels
    df[config.POSITIVE_POSITION_COL] = positive_positions
    return df


def _select_candidate_ids(
    top_items: list[int],
    seen_items: set[int],
    top_n: int,
) -> list[int]:
    candidate_ids: list[int] = []

    for item_id in top_items:
        if item_id in seen_items:
            continue
        candidate_ids.append(item_id)
        if len(candidate_ids) == top_n:
            break

    return candidate_ids


def _fill_missing_candidates(
    candidate_ids: list[int],
    ranked_items: list[int],
    seen_items: set[int],
    top_n: int,
) -> list[int]:
    selected = set(candidate_ids)

    for item_id in ranked_items:
        if item_id in selected or item_id in seen_items:
            continue
        candidate_ids.append(item_id)
        selected.add(item_id)
        if len(candidate_ids) == top_n:
            return candidate_ids

    for item_id in ranked_items:
        if item_id in selected:
            continue
        candidate_ids.append(item_id)
        selected.add(item_id)
        if len(candidate_ids) == top_n:
            break

    return candidate_ids


def _ensure_positive_candidate(
    candidate_ids: list[int],
    target_item_id: int,
    target: float,
    top_n: int,
) -> list[int]:
    if target <= 0 or target_item_id in candidate_ids:
        return candidate_ids

    if len(candidate_ids) < top_n:
        candidate_ids.append(target_item_id)
        return candidate_ids

    candidate_ids[-1] = target_item_id
    return candidate_ids


def _history_to_seen_items(
    history_items: torch.Tensor,
    history_valid_mask: torch.Tensor,
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
