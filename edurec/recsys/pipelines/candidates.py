import faiss
import numpy as np
import pandas as pd
import torch

from ... import settings
from ...datasets import ElearningDataModule, UserHistory, Phase
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

    num_items = int(i_static_feats.size(0))
    if num_items == 0:
        raise RuntimeError("The item catalog must contain at least one item.")

    device = retrieval.device
    all_item_ids = torch.arange(num_items, device=device, dtype=torch.long)
    item_embs = retrieval.encode_items(all_item_ids)
    ann_index = _build_ann_index(item_embs)

    bs = batch_size or dm.batch_size
    results: dict[str, pd.DataFrame | None] = {}

    for split in ("train", "val", "test"):
        split_df = _generate_split_candidates(
            split=split,
            retrieval=retrieval,
            interactions=getattr(dm.artifacts, split),
            history=dm.next_item_hist_by_split[split],
            ann_index=ann_index,
            num_items=num_items,
            top_n=top_n,
            batch_size=bs,
        )
        setattr(dm.artifacts, split, split_df)
        results[split] = split_df

    return results


@torch.no_grad()
def _generate_split_candidates(
    split: str,
    retrieval: Retrieval,
    interactions: pd.DataFrame | None,
    history: UserHistory,
    ann_index: faiss.IndexHNSWFlat,
    num_items: int,
    top_n: int,
    batch_size: int,
) -> pd.DataFrame | None:
    if interactions is None:
        return None

    df = interactions.reset_index(drop=True).copy()
    effective_k = min(top_n, num_items)
    force_positive = split == "train"
    exclude_target_from_fallback = split != "train"

    if df.empty:
        df[settings.CANDIDATE_IDS_COL] = [[] for _ in range(len(df))]
        df[settings.CANDIDATE_LABELS_COL] = [[] for _ in range(len(df))]
        df[settings.POSITIVE_POSITION_COL] = [None for _ in range(len(df))]
        df[settings.TARGET_IN_CANDIDATES_COL] = [False for _ in range(len(df))]
        df[settings.TARGET_FORCED_COL] = [False for _ in range(len(df))]
        return df

    search_k = min(
        num_items,
        max(
            effective_k * settings.FAISS_SEARCH_MULTIPLIER,
            effective_k + history.items.size(1),
        ),
    )

    user_ids = torch.as_tensor(
        df[settings.USER_COL].to_numpy(copy=True), dtype=torch.long
    )
    target_item_ids = torch.as_tensor(
        df[settings.ITEM_COL].to_numpy(copy=True), dtype=torch.long
    )
    targets = torch.as_tensor(
        df[settings.RELEVANT_COL].astype("float32").to_numpy(copy=True),
        dtype=torch.float32,
    )

    candidate_ids: list[list[int]] = []
    candidate_labels: list[list[float]] = []
    positive_positions: list[int | None] = []
    target_in_candidates: list[bool] = []
    target_forced: list[bool] = []

    device = retrieval.device

    for start in range(0, len(df), batch_size):
        stop = min(start + batch_size, len(df))
        sl = slice(start, stop)

        query_emb = retrieval.encode_query(
            user_ids=user_ids[sl].to(device),
            history_items=history.items[sl].to(device),
            history_ctx=history.ctx[sl].to(device),
            history_valid_mask=history.valid_mask[sl].to(device),
        )

        ann_candidates = _ann_search(
            index=ann_index,
            query_embeddings=query_emb,
            top_k=search_k,
        )

        for offset, retrieved_ids in enumerate(ann_candidates):
            row_idx = start + offset

            valid_mask = history.valid_mask[row_idx]
            valid_items = history.items[row_idx][valid_mask]
            seen_items = {
                int(item_id) - 1 for item_id in valid_items.tolist() if int(item_id) > 0
            }

            target_item_id = int(target_item_ids[row_idx])
            target = float(targets[row_idx])

            row_candidates, was_forced = _finalize_candidates(
                retrieved_ids=retrieved_ids,
                seen_items=seen_items,
                top_n=effective_k,
                num_items=num_items,
                force_positive=force_positive and target > 0,
                exclude_target_from_fallback=exclude_target_from_fallback,
                target_item_id=target_item_id,
                target=target,
            )

            labels, pos = _build_candidate_labels(
                candidate_ids=row_candidates,
                target_item_id=target_item_id,
                target=target,
            )

            candidate_ids.append(row_candidates)
            candidate_labels.append(labels)
            positive_positions.append(pos)
            target_in_candidates.append(pos is not None)
            target_forced.append(was_forced)

    df[settings.CANDIDATE_IDS_COL] = candidate_ids
    df[settings.CANDIDATE_LABELS_COL] = candidate_labels
    df[settings.POSITIVE_POSITION_COL] = positive_positions
    df[settings.TARGET_IN_CANDIDATES_COL] = target_in_candidates
    df[settings.TARGET_FORCED_COL] = target_forced
    return df


def _finalize_candidates(
    retrieved_ids: list[int],
    seen_items: set[int],
    top_n: int,
    num_items: int,
    force_positive: bool,
    exclude_target_from_fallback: bool,
    target_item_id: int,
    target: float,
) -> tuple[list[int], bool]:
    candidates: list[int] = []
    selected: set[int] = set()
    was_forced = False

    for item_id in retrieved_ids:
        if item_id < 0 or item_id >= num_items:
            continue
        if item_id in seen_items or item_id in selected:
            continue
        candidates.append(item_id)
        selected.add(item_id)
        if len(candidates) == top_n:
            break

    if force_positive and target > 0 and target_item_id not in selected:
        if len(candidates) < top_n:
            candidates.append(target_item_id)
            selected.add(target_item_id)
            was_forced = True
        elif top_n > 0:
            selected.discard(candidates[-1])
            candidates[-1] = target_item_id
            selected.add(target_item_id)
            was_forced = True

    if len(candidates) < top_n:
        for item_id in range(num_items):
            if exclude_target_from_fallback and item_id == target_item_id:
                continue
            if item_id in selected or item_id in seen_items:
                continue
            candidates.append(item_id)
            selected.add(item_id)
            if len(candidates) == top_n:
                break

    if len(candidates) < top_n:
        for item_id in range(num_items):
            if exclude_target_from_fallback and item_id == target_item_id:
                continue
            if item_id in selected:
                continue
            candidates.append(item_id)
            selected.add(item_id)
            if len(candidates) == top_n:
                break

    return candidates, was_forced


def _build_candidate_labels(
    candidate_ids: list[int],
    target_item_id: int,
    target: float,
) -> tuple[list[float], int | None]:
    labels = [0.0] * len(candidate_ids)

    if target <= 0:
        return labels, None

    if target_item_id not in candidate_ids:
        return labels, None

    pos = candidate_ids.index(target_item_id)
    labels[pos] = float(target)
    return labels, pos


def _build_ann_index(item_embeddings: torch.Tensor) -> faiss.IndexHNSWFlat:
    matrix = item_embeddings.detach().cpu().contiguous().numpy().astype(np.float32)
    dim = matrix.shape[1]

    index = faiss.IndexHNSWFlat(dim, settings.FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = max(settings.FAISS_MIN_EF_SEARCH, 40)
    index.add(matrix)  # type: ignore
    return index


def _ann_search(
    index: faiss.IndexHNSWFlat,
    query_embeddings: torch.Tensor,
    top_k: int,
) -> list[list[int]]:
    if top_k <= 0:
        return [[] for _ in range(query_embeddings.size(0))]

    index.hnsw.efSearch = max(settings.FAISS_MIN_EF_SEARCH, top_k * 2)

    _, indices = index.search(
        query_embeddings.detach().cpu().contiguous().numpy().astype(np.float32), top_k
    )  # type: ignore
    return indices.tolist()
