import torch
from torchmetrics import MetricCollection
from torchmetrics.retrieval import (
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)

EVALUATION_PROTOCOL = "one-target-per-interaction-v1"


def build_ranking_metrics(
    topks: list[int],
    prefix: str = "",
    adaptive_k: bool = False,
) -> MetricCollection:
    """Build the full-catalog retrieval metric set for every top-k."""
    metrics = {}
    for k in topks:
        common = {
            "top_k": k,
            "empty_target_action": "neg",
            "aggregation": "mean",
        }
        metrics[f"precision@{k}"] = RetrievalPrecision(
            **common,
            adaptive_k=adaptive_k,
        )
        metrics[f"recall@{k}"] = RetrievalRecall(**common)
        metrics[f"ndcg@{k}"] = RetrievalNormalizedDCG(**common)
        metrics[f"hit@{k}"] = RetrievalHitRate(**common)
        metrics[f"map@{k}"] = RetrievalMAP(**common)
        metrics[f"mrr@{k}"] = RetrievalMRR(**common)
    return MetricCollection(metrics, prefix=prefix)


@torch.no_grad()
def update_ranking_metrics(
    metrics: MetricCollection,
    scores: torch.Tensor,
    target_item_ids: torch.Tensor,
    query_ids: torch.Tensor,
    history_items: torch.Tensor,
    history_mask: torch.Tensor,
    max_k: int,
) -> None:
    """Update one-target-per-query metrics from full-catalog scores."""
    if scores.ndim != 2:
        raise ValueError(
            f"scores must have shape [batch, num_items], got {scores.shape}."
        )

    if not torch.isfinite(scores).all():
        raise FloatingPointError(
            "Model scores contain NaN or infinite values before evaluation."
        )

    batch_size, num_items = scores.shape
    if max_k <= 0 or max_k > num_items:
        raise ValueError(f"max_k must be in [1, {num_items}], got {max_k}.")

    target_item_ids = target_item_ids.reshape(-1).long()
    query_ids = query_ids.reshape(-1).long()

    if target_item_ids.numel() != batch_size:
        raise ValueError("Expected exactly one target item per query.")
    if query_ids.numel() != batch_size:
        raise ValueError("Expected exactly one query ID per query.")
    if query_ids.unique().numel() != query_ids.numel():
        raise ValueError("query_id values must be unique inside each batch.")
    if target_item_ids.numel() and (
        target_item_ids.min() < 0 or target_item_ids.max() >= num_items
    ):
        raise IndexError(
            f"Target item IDs must be in [0, {num_items - 1}] during evaluation."
        )

    eval_scores = mask_seen_items(
        scores=scores.detach().float(),
        history_items=history_items,
        history_mask=history_mask,
        target_item_ids=target_item_ids,
    )

    top_scores, top_item_ids = torch.topk(
        eval_scores,
        k=max_k,
        dim=1,
        largest=True,
        sorted=True,
    )

    top_targets = top_item_ids.eq(target_item_ids.unsqueeze(1))
    indexes = query_ids.unsqueeze(1).expand_as(top_item_ids)

    metrics.update(
        preds=top_scores.reshape(-1),
        target=top_targets.reshape(-1),
        indexes=indexes.reshape(-1),
    )


def mask_seen_items(
    scores: torch.Tensor,
    history_items: torch.Tensor,
    history_mask: torch.Tensor,
    target_item_ids: torch.Tensor,
) -> torch.Tensor:
    """Mask prior interactions while keeping a repeated current target eligible."""
    masked_scores = scores.clone()
    batch_size, num_items = masked_scores.shape

    if history_items.shape != history_mask.shape:
        raise ValueError("history_items and history_mask must have the same shape.")
    if history_items.size(0) != batch_size:
        raise ValueError("History batch size must match the score batch size.")

    history_ids = history_items.long() - 1
    valid_history = history_mask.bool() & history_ids.ge(0) & history_ids.lt(num_items)

    batch_indexes = (
        torch.arange(batch_size, device=masked_scores.device)
        .unsqueeze(1)
        .expand_as(history_ids)
    )

    target_scores = masked_scores.gather(1, target_item_ids.unsqueeze(1))
    masked_scores[
        batch_indexes[valid_history],
        history_ids[valid_history],
    ] = -torch.inf
    masked_scores.scatter_(1, target_item_ids.unsqueeze(1), target_scores)

    return masked_scores
