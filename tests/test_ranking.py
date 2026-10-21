import pytest
import torch

from edurec.recsys.ranking import (
    build_ranking_metrics,
    mask_seen_items,
    update_ranking_metrics,
)


def test_seen_items_are_masked_but_repeated_target_remains_eligible():
    scores = torch.tensor([[5.0, 3.0, 1.0]])
    result = mask_seen_items(
        scores,
        torch.tensor([[1, 2, 0]]),
        torch.tensor([[True, True, False]]),
        torch.tensor([1]),
    )
    assert result.tolist() == [[-torch.inf, 3.0, 1.0]]
    torch.testing.assert_close(scores, torch.tensor([[5.0, 3.0, 1.0]]))


def test_ranking_metrics_use_queries_and_accept_negative_logits():
    metrics = build_ranking_metrics([1])
    update_ranking_metrics(
        metrics,
        torch.tensor([[-1.0, -2.0, -3.0], [-3.0, -1.0, -2.0]]),
        torch.tensor([0, 2]),
        torch.tensor([10, 11]),
        torch.empty(2, 0, dtype=torch.long),
        torch.empty(2, 0, dtype=torch.bool),
        1,
    )
    for value in metrics.compute().values():
        assert value.item() == pytest.approx(0.5)


def test_ranking_rejects_nonfinite_scores():
    with pytest.raises(FloatingPointError, match="NaN or infinite"):
        update_ranking_metrics(
            build_ranking_metrics([1]),
            torch.tensor([[float("nan"), 1.0]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.empty(1, 0, dtype=torch.long),
            torch.empty(1, 0, dtype=torch.bool),
            1,
        )
