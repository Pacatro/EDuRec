import pytest
import torch

from edurec.recsys.arquitecture import Ranker, RankerConfig


@pytest.fixture
def cfg():
    return RankerConfig(
        dim_model=16,
        n_heads=4,
        n_blocks=2,
        ff_dim=32,
        dropout=0.1,
        max_history_len=5,
    )


def test_ranker_forward_multi_candidate(cfg):
    torch.manual_seed(0)

    model = Ranker(cfg)

    B, L, K, D = 3, 5, 4, cfg.dim_model

    token_u = torch.randn(B, 1, D)
    tokens_i = torch.randn(B, L, D)
    tokens_c = torch.randn(B, K, D)

    # True = posición válida en el historial
    hist_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    scores = model(
        token_u=token_u,
        tokens_i=tokens_i,
        tokens_c=tokens_c,
        hist_mask=hist_mask,
    )

    assert scores.shape == (B, K)
    assert torch.isfinite(scores).all()


def test_ranker_forward_single_candidate(cfg):
    torch.manual_seed(0)

    model = Ranker(cfg)

    B, L, D = 2, 5, cfg.dim_model

    token_u = torch.randn(B, 1, D)
    tokens_i = torch.randn(B, L, D)
    tokens_c = torch.randn(B, D)  # caso soportado: un solo candidato sin eje K
    hist_mask = torch.ones(B, L, dtype=torch.bool)

    scores = model(
        token_u=token_u,
        tokens_i=tokens_i,
        tokens_c=tokens_c,
        hist_mask=hist_mask,
    )

    assert scores.shape == (B,)
    assert torch.isfinite(scores).all()


def test_ranker_backward(cfg):
    torch.manual_seed(0)

    model = Ranker(cfg)

    B, L, K, D = 2, 5, 3, cfg.dim_model

    token_u = torch.randn(B, 1, D, requires_grad=True)
    tokens_i = torch.randn(B, L, D, requires_grad=True)
    tokens_c = torch.randn(B, K, D, requires_grad=True)
    hist_mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, True, False],
        ],
        dtype=torch.bool,
    )

    scores = model(
        token_u=token_u,
        tokens_i=tokens_i,
        tokens_c=tokens_c,
        hist_mask=hist_mask,
    )

    loss = scores.mean()
    loss.backward()

    assert token_u.grad is not None
    assert tokens_i.grad is not None
    assert tokens_c.grad is not None

    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad


def test_ranker_without_hist_mask(cfg):
    torch.manual_seed(0)

    model = Ranker(cfg)

    B, L, K, D = 2, 5, 2, cfg.dim_model

    token_u = torch.randn(B, 1, D)
    tokens_i = torch.randn(B, L, D)
    tokens_c = torch.randn(B, K, D)

    scores = model(
        token_u=token_u,
        tokens_i=tokens_i,
        tokens_c=tokens_c,
        hist_mask=None,
    )

    assert scores.shape == (B, K)
