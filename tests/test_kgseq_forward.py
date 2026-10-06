"""CPU forward/backward tests for the refactored KGSeq architecture.

They build a tiny synthetic graph and batch, so they run in milliseconds and do
not require any dataset download. The goal is to catch shape/alignment and
ablation regressions in the model wiring.
"""

import unittest
from dataclasses import replace

import torch

from edurec.recsys.archs.kgseq import KGSeq
from edurec.recsys.configs import ModelConfig


def _config(**overrides: object) -> ModelConfig:
    cfg = ModelConfig(
        num_users=5,
        num_items=7,
        num_item_dense_feats=4,
        num_item_text_feats=0,
        num_user_dense_feats=2,
        num_interaction_dense_feats=2,
        user_cat_cardinalities=[4],
        interaction_cat_cardinalities=[3],
        kg_node_counts={"attr::c": 3},
        kg_edge_types=[["item", "c", "attr::c"], ["attr::c", "rev_c", "item"]],
        emb_dim=16,
        gnn_layers=2,
        gnn_heads=2,
        rnn_hidden_dim=8,
        rnn_layers=1,
        hidden_dims=[8],
    )
    return replace(cfg, **overrides)


def _batch() -> dict[str, object]:
    torch.manual_seed(0)
    batch, history = 4, 5
    item_edges = torch.randint(0, 7, (6,))
    attr_edges = torch.randint(0, 3, (6,))
    return {
        "h_ids": torch.randint(0, 8, (batch, history)),
        "h_mask": torch.tensor(
            [
                [True, True, True, False, False],
                [True, False, False, False, False],
                [False, False, False, False, False],
                [True, True, True, True, True],
            ]
        ),
        "edge_index": {
            ("item", "c", "attr::c"): torch.stack([item_edges, attr_edges]),
            ("attr::c", "rev_c", "item"): torch.stack([attr_edges, item_edges]),
        },
        "i_static_feats": torch.randn(7, 4),
        "u_static_feats": torch.randn(5, 2),
        "u_cat_feats": torch.randint(0, 4, (5, 1)),
        "user_ids": torch.randint(0, 5, (batch,)),
        "h_dense": torch.rand(batch, history, 2),
        "h_cat": torch.randint(-1, 3, (batch, history, 1)),
        "h_delta": torch.rand(batch, history) * 100.0,
    }


def _run(cfg: ModelConfig) -> None:
    model = KGSeq(cfg)
    inputs = _batch()
    candidates = torch.randint(0, 7, (4, 3))

    full = model(**inputs)  # type: ignore[arg-type]
    assert full.shape == (4, 7), full.shape
    assert torch.isfinite(full).all()

    cand = model(candidate_item_ids=candidates, **inputs)  # type: ignore[arg-type]
    assert cand.shape == (4, 3), cand.shape
    assert torch.isfinite(cand).all()

    cand.sum().backward()
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.parameters()
    )


class KGSeqForwardTest(unittest.TestCase):
    def test_full_architecture(self) -> None:
        _run(_config())

    def test_item_kg_only_ablation(self) -> None:
        cfg = _config(
            use_user_features=False,
            use_interaction_features=False,
            use_time_features=False,
        )
        assert not cfg.user_state.is_active
        assert not cfg.interaction_context.is_active
        _run(cfg)

    def test_user_features_toggle_disables_profile(self) -> None:
        model = KGSeq(_config(use_user_features=False))
        assert model.user_state.profile is None
        model = KGSeq(_config(use_user_features=True))
        assert model.user_state.profile is not None

    def test_attention_pooling_ablation(self) -> None:
        _run(_config(use_attention_pooling=True))

    def test_id_graph_ablation(self) -> None:
        _run(_config(graph_mode="id"))

    def test_missing_context_and_time_signals(self) -> None:
        model = KGSeq(_config())
        inputs = _batch()
        inputs["h_dense"] = None  # type: ignore[assignment]
        inputs["h_cat"] = None  # type: ignore[assignment]
        inputs["h_delta"] = None  # type: ignore[assignment]
        scores = model(**inputs)  # type: ignore[arg-type]
        assert scores.shape == (4, 7)
        assert torch.isfinite(scores).all()


if __name__ == "__main__":
    unittest.main()
