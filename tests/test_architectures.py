from dataclasses import replace

import pytest
import torch

from edurec.recsys.archs.kgseq import KGSeq
from edurec.recsys.archs.sasrec_text import SASRecText
from edurec.recsys.configs import ModelArch


@pytest.mark.parametrize(
    "arch,has_history",
    [
        (ModelArch.KG_RNN, True),
        (ModelArch.KG_RNN, False),
        (ModelArch.SASREC_TEXT, True),
    ],
)
def test_architecture_scores_candidates_and_backpropagates(
    model_config, arch, has_history
):
    torch.manual_seed(42)
    forward = ("item", "has_topic", "topic")
    reverse = ("topic", "rev_has_topic", "item")
    cfg = replace(
        model_config,
        arch=arch,
        has_history=has_history,
        kg_node_counts={"topic": 2},
        kg_edge_types=[list(forward), list(reverse)],
        gnn_layers=1,
    )
    model = KGSeq(cfg) if arch == ModelArch.KG_RNN else SASRecText(cfg)
    edges = torch.tensor([[0, 1, 2, 3], [0, 1, 0, 1]])
    model.register_static(
        {forward: edges, reverse: edges.flip(0)},
        torch.randn(4, 2),
        torch.empty(2, 0),
        torch.empty(2, 0, dtype=torch.long),
    )
    history = torch.tensor([[1, 2, 0], [3, 0, 0]])
    mask = history > 0
    users = torch.tensor([0, 1])
    candidates = torch.tensor([[0, 3], [1, 2]])
    model.eval()
    with torch.no_grad():
        full = model(history, mask, users)
        selected = model(history, mask, users, candidate_item_ids=candidates)
    assert full.shape == (2, 4)
    assert torch.isfinite(full).all()
    torch.testing.assert_close(selected, full.gather(1, candidates))
    model.train()
    loss = model(history, mask, users).square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
