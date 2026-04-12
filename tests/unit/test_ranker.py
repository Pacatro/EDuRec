import pytest
import torch

from edurec.recsys.model import Ranker, RankerConfig


@pytest.fixture
def cfg():
    return RankerConfig(
        emb_dim=16,
        n_heads=4,
        n_blocks=2,
        ff_dim=32,
        dropout=0.0,
        max_histoy_len=5,
    )


def test_ranker_forward_multi_candidate(cfg):
    torch.manual_seed(0)

    model = Ranker(cfg).eval()

    B, H, C, D = 3, 5, 4, cfg.emb_dim
    user_emb = torch.randn(B, D)
    history_emb = torch.randn(B, H, D)
    candidate_emb = torch.randn(B, C, D)
    history_valid_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    scores = model(user_emb, history_emb, candidate_emb, history_valid_mask)

    assert scores.shape == (B, C, 1)
    assert torch.isfinite(scores).all()


def test_ranker_forward_single_candidate(cfg):
    torch.manual_seed(0)

    model = Ranker(cfg).eval()

    B, H, D = 2, 5, cfg.emb_dim
    user_emb = torch.randn(B, D)
    history_emb = torch.randn(B, H, D)
    candidate_emb = torch.randn(B, D)
    history_valid_mask = torch.ones(B, H, dtype=torch.bool)

    scores = model(user_emb, history_emb, candidate_emb, history_valid_mask)

    assert scores.shape == (B,)
    assert torch.isfinite(scores).all()


def test_ranker_padding_does_not_affect_scores(cfg):
    torch.manual_seed(0)

    model = Ranker(cfg).eval()

    B, H, C, D = 2, 5, 3, cfg.emb_dim
    user_emb = torch.randn(B, D)
    base_history = torch.randn(B, H, D)
    candidate_emb = torch.randn(B, C, D)
    history_valid_mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, False, False],
        ],
        dtype=torch.bool,
    )

    perturbed_history = base_history.clone()
    perturbed_history[~history_valid_mask] = torch.randn_like(
        perturbed_history[~history_valid_mask]
    )

    with torch.no_grad():
        scores_a = model(user_emb, base_history, candidate_emb, history_valid_mask)
        scores_b = model(user_emb, perturbed_history, candidate_emb, history_valid_mask)

    assert torch.allclose(scores_a, scores_b, atol=1e-6)


def test_ranker_candidate_isolation(cfg):
    torch.manual_seed(0)

    model = Ranker(cfg).eval()

    B, H, C, D = 1, 4, 3, cfg.emb_dim
    user_emb = torch.randn(B, D)
    history_emb = torch.randn(B, H, D)
    history_valid_mask = torch.tensor([[True, True, True, False]], dtype=torch.bool)

    anchor_candidate = torch.randn(B, 1, D)
    other_candidates_a = torch.randn(B, C - 1, D)
    other_candidates_b = torch.randn(B, C - 1, D)

    candidates_a = torch.cat([anchor_candidate, other_candidates_a], dim=1)
    candidates_b = torch.cat([anchor_candidate, other_candidates_b], dim=1)

    with torch.no_grad():
        scores_a = model(user_emb, history_emb, candidates_a, history_valid_mask)
        scores_b = model(user_emb, history_emb, candidates_b, history_valid_mask)

    assert torch.allclose(scores_a[:, 0], scores_b[:, 0], atol=1e-6)
