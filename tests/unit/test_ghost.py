import pytest
import torch
from torch_geometric.data import Data

from edurec.recsys.model import Ghost, GhostConfig


@pytest.fixture
def cfg():
    return GhostConfig(
        num_users=8,
        num_items=12,
        num_ctx_feats=3,
        emb_dim=16,
        num_user_numeric_feats=1,
        num_item_numeric_feats=2,
        user_cat_cardinalities=[5],
        item_cat_cardinalities=[6, 4],
        gnn_layers=1,
        n_heads=4,
        n_blocks=1,
        ff_dim=32,
        dropout=0.0,
    )


def test_ghost_forward(cfg: GhostConfig):
    torch.manual_seed(0)

    model = Ghost(cfg).eval()

    edge_index = torch.tensor(
        [
            [0, 1, 2, cfg.num_users + 0, cfg.num_users + 1, cfg.num_users + 2],
            [cfg.num_users + 0, cfg.num_users + 1, cfg.num_users + 2, 0, 1, 2],
        ],
        dtype=torch.long,
    )
    inter_graph = Data(edge_index=edge_index, num_nodes=cfg.num_users + cfg.num_items)

    u_static = torch.tensor(
        [
            [0.2, 0.0],
            [0.8, 1.0],
            [0.1, 2.0],
            [0.5, 1.0],
            [0.3, 0.0],
            [0.9, 2.0],
            [0.4, 1.0],
            [0.7, 3.0],
        ],
        dtype=torch.float32,
    )
    i_static = torch.tensor(
        [
            [0.1, 0.5, 0.0, 0.0],
            [0.2, 0.3, 1.0, 1.0],
            [0.4, 0.7, 2.0, 0.0],
            [0.6, 0.8, 3.0, 2.0],
            [0.3, 0.2, 1.0, 1.0],
            [0.9, 0.4, 4.0, 2.0],
            [0.5, 0.1, 0.0, 1.0],
            [0.7, 0.9, 2.0, 2.0],
            [0.8, 0.6, 3.0, 0.0],
            [0.2, 0.4, 4.0, 1.0],
            [0.1, 0.7, 1.0, 2.0],
            [0.6, 0.5, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )

    batch_size = 3
    history_len = 4
    num_candidates = 5

    scores = model(
        u_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        h_ids=torch.tensor(
            [
                [1, 2, 0, 0],
                [3, 4, 5, 0],
                [6, 7, 8, 9],
            ],
            dtype=torch.long,
        ),
        h_ctx=torch.randn(batch_size, history_len, cfg.num_ctx_feats),
        h_mask=torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ],
            dtype=torch.bool,
        ),
        c_ids=torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [2, 4, 6, 8, 10],
            ],
            dtype=torch.long,
        ),
        inter_graph=inter_graph,
        u_static_global=u_static,
        i_static_global=i_static,
    )

    assert scores.shape == (batch_size, num_candidates, 1)
    assert torch.isfinite(scores).all()
