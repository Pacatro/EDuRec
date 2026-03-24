import pytest
import torch
from torch_geometric.data import Data

from edurec.datasets.loaders import DatasetName
from edurec.recsys.arquitecture import GnnEncoderConfig, GnnEncoder
from edurec.datasets import ElearningDataModule


def test_gcl():
    cfg = GnnEncoderConfig(
        num_users=10,
        num_items=5,
        embed_dim=32,
        drop_edges_p=0.2,
        tau=0.1,
        num_layers=2,
    )

    num_users = 10
    num_items = 5

    u_x = torch.randn(num_users, cfg.embed_dim)
    i_x = torch.randn(num_items, cfg.embed_dim)

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [10, 11, 12, 13],
        ],
        dtype=torch.long,
    )

    data = Data(edge_index=edge_index)
    data.u_x = u_x
    data.i_x = i_x
    data.num_users = num_users

    model = GnnEncoder(cfg)

    eu_struct, ei_struct = model(data)

    assert eu_struct.shape == (num_users, cfg.dim_hidden)
    assert ei_struct.shape == (num_items, cfg.dim_hidden)
    assert len(model.convs) == cfg.num_layers


def test_gcl_mars():
    dm = ElearningDataModule(
        DatasetName.MARS, batch_size=1, val_ratio=0.2, test_ratio=0.2
    )

    dm.setup()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    graph = dm.create_inter_graph().to(device)

    cfg = GCLConfig(
        dim_user=graph.u_x.shape[1],
        dim_item=graph.i_x.shape[1],
        dim_hidden=32,
        drop_edges_p=0.2,
        tau=0.1,
        num_layers=2,
    )

    model = GCL(cfg).to(device)

    u_struct, i_struct = model(graph)

    assert u_struct.shape == (dm.num_users, cfg.dim_hidden)
    assert i_struct.shape == (dm.num_items, cfg.dim_hidden)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
