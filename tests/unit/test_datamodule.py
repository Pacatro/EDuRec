import pytest

from edurec.datasets import DatasetName, ElearningDataModule
from edurec import config


@pytest.fixture
def dm() -> ElearningDataModule:
    return ElearningDataModule(
        DatasetName.MARS, batch_size=1, test_ratio=0.2, val_ratio=0.2
    )


def test_split_mars(dm: ElearningDataModule):
    inter_len = dm.num_interactions
    train_df, val_df, test_df = dm._split_data()

    assert len(train_df) + len(val_df) + len(test_df) == inter_len
    assert len(train_df) == inter_len - len(test_df) - len(val_df)
    assert len(val_df) == len(test_df)


def test_setup(dm: ElearningDataModule):
    dm.setup()
    assert dm.is_processed


def test_create_inter_graph(dm: ElearningDataModule):
    dm.setup()

    train_raw = dm._processed_data["train"]

    assert train_raw is not None

    num_pos_inter = len(train_raw[train_raw[config.RELEVANT_COL] > 0])

    graph = dm.create_inter_graph()

    assert graph.num_edges == 2 * num_pos_inter

    num_users = graph.num_users

    assert graph.edge_index is not None

    item_indices = graph.edge_index[1, :num_pos_inter]

    assert item_indices.min().item() >= num_users

    assert hasattr(graph, "u_x")
    assert hasattr(graph, "i_x")
    assert graph.u_x.shape[0] == num_users


def test_generate_global_history(dm: ElearningDataModule):
    dm.setup()

    global_history = dm._generate_global_history()

    assert global_history is not None
    assert len(global_history) > 0


def test_num_static_feats(dm: ElearningDataModule):
    dm.setup()

    assert dm.num_user_feats == 1
    assert dm.num_item_feats == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
