import torch
from torch_geometric.data import Data

from edurec.recsys.model import (
    GnnEncoder,
    GnnEncoderConfig,
    InfoNCELoss,
    GhostConfig,
)
from edurec.recsys.engine import RecSys


def test_gnn_encoder_shapes():
    cfg = GnnEncoderConfig(
        num_users=10,
        num_items=5,
        emb_dim=32,
        drop_edges_p=0.2,
        tau=0.1,
        num_layers=2,
    )

    edge_index = torch.tensor(
        [
            [0, 1, 2, 10, 11, 12],
            [10, 11, 12, 0, 1, 2],
        ],
        dtype=torch.long,
    )
    data = Data(edge_index=edge_index, num_nodes=cfg.num_users + cfg.num_items)

    model = GnnEncoder(cfg)

    user_embs, item_embs = model(data)

    assert user_embs.shape == (cfg.num_users, cfg.emb_dim)
    assert item_embs.shape == (cfg.num_items, cfg.emb_dim)
    assert len(model.convs) == cfg.num_layers


def test_info_nce_loss_is_finite():
    torch.manual_seed(0)

    loss_fn = InfoNCELoss(tau=0.1)
    u_emb1 = torch.randn(8, 16)
    i_emb1 = torch.randn(6, 16)
    u_emb2 = torch.randn(8, 16)
    i_emb2 = torch.randn(6, 16)

    loss = loss_fn(u_emb1, i_emb1, u_emb2, i_emb2)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_graph_view_preserves_bidirectional_pairs():
    cfg = GhostConfig(num_users=4, num_items=3, num_ctx_feats=0, edge_dropout=0.5)
    graph = Data(
        edge_index=torch.tensor(
            [
                [0, 1, 2, 4, 5, 6],
                [4, 5, 6, 0, 1, 2],
            ],
            dtype=torch.long,
        ),
        num_nodes=7,
    )
    recsys = RecSys(
        cfg=cfg,
        inter_graph=graph,
        u_static=torch.zeros(cfg.num_users, 0),
        i_static=torch.zeros(cfg.num_items, 0),
    )

    assert graph.edge_index is not None
    view = recsys._create_graph_view(graph.edge_index)

    assert view.size(1) % 2 == 0

    half_edges = view.size(1) // 2
    u2i = view[:, :half_edges]
    i2u = view[:, half_edges:]

    assert torch.equal(u2i[0], i2u[1])
    assert torch.equal(u2i[1], i2u[0])
