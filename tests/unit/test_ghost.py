import pytest
import torch
from torch_geometric.data import Data
from unittest.mock import MagicMock

from edurec.recsys.arquitecture import Ghost, GhostConfig


@pytest.fixture
def cfg():
    return GhostConfig(user_dim=8, item_dim=12, hidden_dim=16, max_history_len=10)


def test_ghost(cfg: GhostConfig, monkeypatch):
    mock_gcl = MagicMock()
    mock_ranker = MagicMock()
    monkeypatch.setattr("edurec.recsys.arquitecture.ghost.GCL", lambda _: mock_gcl)
    monkeypatch.setattr(
        "edurec.recsys.arquitecture.ghost.Ranker", lambda _: mock_ranker
    )

    model = Ghost(cfg)

    batch_size = 4
    num_users = 20
    num_items = 50
    history_len = cfg.max_history_len
    num_candidates = 5
    ctx_dim = 5

    u_id = torch.randint(0, num_users, (batch_size,))
    h_ids = torch.randint(0, num_items, (batch_size, history_len))
    h_ctx = torch.randn(batch_size, history_len, ctx_dim)
    c_ids = torch.randint(0, num_items, (batch_size, num_candidates))
    u_static_global = torch.randn(num_users, cfg.user_dim)
    i_static_global = torch.randn(num_items, cfg.item_dim)

    inter_graph = Data(edge_index=torch.tensor([[0], [1]]))
    inter_graph.u_x = u_static_global
    inter_graph.i_x = i_static_global

    u_struct_mock = torch.randn(num_users, cfg.hidden_dim)
    i_struct_mock = torch.randn(num_items, cfg.hidden_dim)
    mock_gcl.return_value = (u_struct_mock, i_struct_mock)

    expected_scores = torch.randn(batch_size, num_candidates)
    mock_ranker.return_value = expected_scores

    scores = model(
        u_id=u_id,
        h_ids=h_ids,
        h_ctx=h_ctx,
        c_ids=c_ids,
        inter_graph=inter_graph,
        u_static_global=u_static_global,
        i_static_global=i_static_global,
    )

    assert scores.shape == (batch_size, num_candidates)
    assert model.proj_ctx.out_features == cfg.hidden_dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
