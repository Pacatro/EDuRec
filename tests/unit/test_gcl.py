import pytest
import torch
from torch_geometric.data import Data

from edurec.recsys.arquitecture import GCLConfig, GhostGCL


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
