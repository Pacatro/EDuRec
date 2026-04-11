import json

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
    assert set(dm.seen_items_by_split) == {"train", "val", "test"}
    assert config.RELEVANT_COL in dm.interactions.columns
    assert dm.interactions.groupby(config.USER_COL).size().min() >= dm.min_interactions
    assert set(dm.users_feats[config.USER_COL]) == set(dm.interactions[config.USER_COL])


def test_seen_items_by_split_are_monotonic(dm: ElearningDataModule):
    users = set()
    for split_seen_items in dm.seen_items_by_split.values():
        users.update(split_seen_items.keys())

    for user_id in users:
        train_seen = dm.seen_items_by_split["train"].get(user_id, set())
        val_seen = dm.seen_items_by_split["val"].get(user_id, set())
        test_seen = dm.seen_items_by_split["test"].get(user_id, set())

        assert train_seen <= val_seen
        assert val_seen <= test_seen


def test_create_inter_graph(dm: ElearningDataModule):
    train_raw = dm._processed_data["train"]
    assert train_raw is not None

    num_pos_inter = len(train_raw[train_raw[config.RELEVANT_COL] > 0])
    graph = dm.create_inter_graph()

    assert graph.num_edges == 2 * num_pos_inter
    assert graph.edge_index is not None
    assert graph.edge_index[1, :num_pos_inter].min().item() >= graph.num_users


def test_feature_metadata_and_static_shapes(dm: ElearningDataModule):
    assert dm.num_user_dense_feats == len(
        dm.data_processor.feature_metadata["users"].dense_cols
    )
    assert dm.num_item_dense_feats == len(
        dm.data_processor.feature_metadata["items"].dense_cols
    )
    assert dm.user_cat_cardinalities
    assert dm.item_cat_cardinalities
    assert dm.artifacts.train is not None
    assert dm._processed_data["train"] is not None
    assert dm.artifacts.train.equals(dm._processed_data["train"])
    assert dm.artifacts.u_static_feats is not None
    assert dm.artifacts.i_static_feats is not None
    assert dm.artifacts.u_static_feats.shape[0] == dm.num_users
    assert dm.artifacts.i_static_feats.shape[0] == dm.num_items
    assert dm.artifacts.u_static_feats.shape[1] == dm.num_user_feats
    assert dm.artifacts.i_static_feats.shape[1] == dm.num_item_feats
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
    assert dm.num_item_dense_feats > len(
        dm.data_processor.feature_metadata["items"].numeric_cols
    )

    ctx_cols = [c for c in dm.artifacts.train.columns if c not in dm.excluded_cols]
    assert dm.num_ctx_feats == len(ctx_cols)
    assert config.TIME_COL in dm.artifacts.train.columns
    assert config.TIME_COL not in ctx_cols
    assert any(col.startswith("time_") for col in ctx_cols)


def test_processed_cache_metadata_compatibility(tmp_path):
    dm = ElearningDataModule(
        DatasetName.MARS,
        batch_size=1,
        test_ratio=0.2,
        val_ratio=0.2,
        use_processed_data=True,
    )
    dm.processed_folder = tmp_path / dm.dataset_name.value
    dm.processed_folder.mkdir(parents=True)

    metadata_path = dm.processed_folder / "preprocess_metadata.json"

    metadata_path.write_text(
        '{"preprocess_cache_version": 1, "feature_types": ["numeric"]}',
        encoding="utf-8",
    )
    assert not dm._has_compatible_cache()

    metadata_path.write_text(
        json.dumps(dm._build_cache_metadata()),
        encoding="utf-8",
    )
    assert dm._has_compatible_cache()
