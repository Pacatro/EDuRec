import pytest

from edurec import config
from edurec.datasets import DatasetName, ElearningDataModule


@pytest.fixture
def dm(tmp_path) -> ElearningDataModule:
    dm = ElearningDataModule(
        DatasetName.MARS, batch_size=1, test_ratio=0.2, val_ratio=0.2
    )
    dm.processed_folder = tmp_path / dm.dataset_name.value
    dm.setup()
    return dm


def test_split_mars():
    dm = ElearningDataModule(
        DatasetName.MARS, batch_size=1, test_ratio=0.2, val_ratio=0.2
    )
    inter_len = dm.num_interactions
    train_df, val_df, test_df = dm._split_data()

    assert len(train_df) + len(val_df) + len(test_df) == inter_len
    assert len(train_df) == inter_len - len(test_df) - len(val_df)
    assert len(val_df) == len(test_df)


def test_setup(dm: ElearningDataModule):
    assert dm.is_processed
    assert dm.artifacts.is_ready
    assert dm.data_processor is not None
    assert len(dm.user_positive_items) > 0
    assert config.RELEVANT_COL in dm.interactions.columns
    assert dm.interactions.groupby(config.USER_COL).size().min() >= dm.min_interactions
    assert set(dm.users_feats[config.USER_COL]) == set(dm.interactions[config.USER_COL])


def test_create_inter_graph(dm: ElearningDataModule):
    train_raw = dm._processed_data["train"]
    assert train_raw is not None

    num_pos_inter = len(train_raw[train_raw[config.RELEVANT_COL] > 0])
    graph = dm.create_inter_graph()

    assert graph.num_edges == 2 * num_pos_inter
    assert graph.edge_index is not None
    assert graph.edge_index[1, :num_pos_inter].min().item() >= graph.num_users
    assert graph.u_x.shape[0] == graph.num_users
    assert graph.i_x.shape[0] == graph.num_items


def test_feature_metadata_and_static_shapes(dm: ElearningDataModule):
    assert dm.num_user_numeric_feats == 0
    assert dm.num_item_numeric_feats == 2
    assert dm.user_cat_cardinalities
    assert dm.item_cat_cardinalities
    assert dm.num_user_feats == 1
    assert dm.num_item_feats == 5
    assert dm.num_ctx_feats == 8
    assert dm.artifacts.train is not None
    assert dm._processed_data["train"] is not None
    assert dm.artifacts.train.equals(dm._processed_data["train"])
    assert dm.artifacts.u_static_feats is not None
    assert dm.artifacts.i_static_feats is not None
    assert dm.artifacts.u_static_feats.shape[0] == dm.num_users
    assert dm.artifacts.i_static_feats.shape[0] == dm.num_items
    assert dm.data_processor is not None
    assert dm.data_processor.feature_metadata["items"].text_cols == [
        "name",
        "description",
    ]
    assert dm.data_processor.feature_metadata["items"].list_cols == [
        "job",
        "software",
        "theme",
    ]
    assert dm.data_processor.feature_metadata["items"].pending_cols == [
        "name",
        "description",
        "job",
        "software",
        "theme",
    ]
    # assert config.TIME_COL in dm.artifacts.train.columns
    # assert config.TIME_COL not in dm.excluded_cols
