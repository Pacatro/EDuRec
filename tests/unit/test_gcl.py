import pytest
import torch
from torch_geometric.data import Data

from edurec.datasets.loaders import DatasetName
from edurec.recsys.arquitecture import GCLConfig, GhostGCL
from edurec.datasets import ElearningDataModule


def test_gcl():
    cfg = GCLConfig(
        dim_user=16,
        dim_item=24,
        dim_hidden=32,
        drop_edges_p=0.2,
        tau=0.1,
    )

    num_users = 10
    num_items = 5

    u_x = torch.randn(num_users, cfg.dim_user)
    i_x = torch.randn(num_items, cfg.dim_item)

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

    model = GhostGCL(cfg)

    eu_struct, ei_struct, loss = model(data)

    assert eu_struct.shape == (num_users, cfg.dim_hidden)
    assert ei_struct.shape == (num_items, cfg.dim_hidden)

    assert loss.dim() == 0
    assert loss.requires_grad
    assert not torch.isnan(loss)


def test_gcl_mars():
    dm = ElearningDataModule(
        DatasetName.MARS, batch_size=1, val_ratio=0.2, test_ratio=0.2
    )

    dm.setup()

    graph = dm.create_inter_graph().to("cuda")

    cfg = GCLConfig(
        dim_user=graph.u_x.shape[1],
        dim_item=graph.i_x.shape[1],
        dim_hidden=32,
        drop_edges_p=0.2,
        tau=0.1,
    )

    model = GhostGCL(cfg).to("cuda")

    u_struct, i_struct, loss = model(graph)

    print(u_struct.shape)
    print(i_struct.shape)
    print(loss)

    assert not True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
