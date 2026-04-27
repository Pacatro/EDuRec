import json
from typing import Any, cast

import pandas as pd
import pytest
import torch

from edurec import settings
from edurec.datasets import DatasetName, ElearningDataModule
from edurec.datasets.datamodule import ProcessedArtifacts


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
    assert dm.artifacts.train is not None
    assert dm.artifacts.val is not None
    assert dm.artifacts.test is not None
    assert set(dm.seen_items_by_split) == {"train", "val", "test"}
    assert set(dm.next_item_hist_by_split) == {"train", "val", "test"}
    assert settings.RELEVANT_COL in dm.interactions.columns
    assert (
        dm.interactions.groupby(settings.USER_COL).size().min() >= dm.min_interactions
    )
    assert set(dm.users_feats[settings.USER_COL]) == set(
        dm.interactions[settings.USER_COL]
    )
    assert settings.INTERACTION_ORDER_COL in dm.artifacts.train.columns
    assert len(dm.next_item_hist_by_split["train"].items) == len(dm.artifacts.train)
    assert len(dm.next_item_hist_by_split["val"].items) == len(dm.artifacts.val)
    assert len(dm.next_item_hist_by_split["test"].items) == len(dm.artifacts.test)


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

    num_pos_inter = len(train_raw[train_raw[settings.RELEVANT_COL] > 0])
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
    assert settings.TIME_COL in dm.artifacts.train.columns
    assert settings.TIME_COL not in ctx_cols
    assert settings.INTERACTION_ORDER_COL not in ctx_cols
    assert any(col.startswith("time_") for col in ctx_cols)


def _build_synthetic_dm(
    train_df,
    val_df,
    test_df,
    num_users: int = 2,
    num_items: int = 5,
) -> ElearningDataModule:
    dm = ElearningDataModule(
        DatasetName.ITM, batch_size=1, test_ratio=0.2, val_ratio=0.2
    )
    dm.artifacts = ProcessedArtifacts(
        train=train_df,
        val=val_df,
        test=test_df,
        u_static_feats=torch.zeros((num_users, 1), dtype=torch.float32),
        i_static_feats=torch.zeros((num_items, 1), dtype=torch.float32),
        data_processor=cast(Any, object()),
    )
    dm._build_runtime_state()
    return dm


def test_history_prefix_repeated_item_uses_immediate_prior_prefix():
    train_df = pd.DataFrame(
        [
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 0,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-01T00:00:00Z",
                "ctx_value": 0.1,
                settings.INTERACTION_ORDER_COL: 0,
            },
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 1,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-02T00:00:00Z",
                "ctx_value": 0.2,
                settings.INTERACTION_ORDER_COL: 1,
            },
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 0,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-03T00:00:00Z",
                "ctx_value": 0.3,
                settings.INTERACTION_ORDER_COL: 2,
            },
        ]
    )
    empty_df = train_df.iloc[0:0].copy()
    dm = _build_synthetic_dm(train_df, empty_df, empty_df, num_items=3)

    train_history = dm.next_item_hist_by_split["train"]
    assert train_history.valid_mask[0].sum().item() == 0
    assert train_history.items[1, :1].tolist() == [1]
    assert train_history.items[2, :2].tolist() == [1, 2]


def test_history_includes_non_relevant_prior_interactions():
    train_df = pd.DataFrame(
        [
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 2,
                settings.RELEVANT_COL: 0,
                settings.TIME_COL: "2024-01-01T00:00:00Z",
                "ctx_value": 0.1,
                settings.INTERACTION_ORDER_COL: 0,
            },
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 4,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-02T00:00:00Z",
                "ctx_value": 0.2,
                settings.INTERACTION_ORDER_COL: 1,
            },
        ]
    )
    empty_df = train_df.iloc[0:0].copy()
    dm = _build_synthetic_dm(train_df, empty_df, empty_df, num_items=6)

    train_history = dm.next_item_hist_by_split["train"]
    assert train_history.valid_mask[0].sum().item() == 0
    assert train_history.valid_mask[1, :1].tolist() == [True]
    assert train_history.items[1, :1].tolist() == [3]
    assert train_history.ctx[1, 0, 0].item() == pytest.approx(0.1)


def test_current_interaction_never_appears_in_its_own_history():
    train_df = pd.DataFrame(
        [
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 1,
                settings.RELEVANT_COL: 0,
                settings.TIME_COL: "2024-01-01T00:00:00Z",
                "ctx_value": 0.1,
                settings.INTERACTION_ORDER_COL: 0,
            },
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 1,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-02T00:00:00Z",
                "ctx_value": 0.2,
                settings.INTERACTION_ORDER_COL: 1,
            },
        ]
    )
    empty_df = train_df.iloc[0:0].copy()
    dm = _build_synthetic_dm(train_df, empty_df, empty_df, num_items=3)

    train_history = dm.next_item_hist_by_split["train"]
    assert train_history.valid_mask[0].sum().item() == 0
    assert train_history.valid_mask[1, :1].tolist() == [True]
    assert train_history.items[0, 0].item() == 0
    assert train_history.items[1, :1].tolist() == [2]


def test_history_prefixes_differ_within_val_when_train_has_no_item():
    train_df = pd.DataFrame(
        columns=[
            settings.USER_COL,
            settings.ITEM_COL,
            settings.RELEVANT_COL,
            settings.TIME_COL,
            "ctx_value",
            settings.INTERACTION_ORDER_COL,
        ]
    )
    val_df = pd.DataFrame(
        [
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 0,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-01T00:00:00Z",
                "ctx_value": 0.1,
                settings.INTERACTION_ORDER_COL: 0,
            },
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 1,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-02T00:00:00Z",
                "ctx_value": 0.2,
                settings.INTERACTION_ORDER_COL: 1,
            },
        ]
    )
    empty_df = val_df.iloc[0:0].copy()
    dm = _build_synthetic_dm(train_df, val_df, empty_df, num_items=3)

    val_history = dm.next_item_hist_by_split["val"]
    assert val_history.valid_mask[0].sum().item() == 0
    assert val_history.items[1, :1].tolist() == [1]


def test_val_history_inherits_non_relevant_events_from_train():
    train_df = pd.DataFrame(
        [
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 2,
                settings.RELEVANT_COL: 0,
                settings.TIME_COL: "2024-01-01T00:00:00Z",
                "ctx_value": 0.1,
                settings.INTERACTION_ORDER_COL: 0,
            }
        ]
    )
    val_df = pd.DataFrame(
        [
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 4,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-02T00:00:00Z",
                "ctx_value": 0.2,
                settings.INTERACTION_ORDER_COL: 1,
            }
        ]
    )
    empty_df = val_df.iloc[0:0].copy()
    dm = _build_synthetic_dm(train_df, val_df, empty_df, num_items=6)

    val_history = dm.next_item_hist_by_split["val"]
    assert val_history.valid_mask[0, :1].tolist() == [True]
    assert val_history.items[0, :1].tolist() == [3]


def test_test_history_includes_train_val_and_prior_test():
    train_df = pd.DataFrame(
        [
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 0,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-01T00:00:00Z",
                "ctx_value": 0.1,
                settings.INTERACTION_ORDER_COL: 0,
            }
        ]
    )
    val_df = pd.DataFrame(
        [
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 1,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-02T00:00:00Z",
                "ctx_value": 0.2,
                settings.INTERACTION_ORDER_COL: 1,
            }
        ]
    )
    test_df = pd.DataFrame(
        [
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 2,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-03T00:00:00Z",
                "ctx_value": 0.3,
                settings.INTERACTION_ORDER_COL: 2,
            },
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 3,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-04T00:00:00Z",
                "ctx_value": 0.4,
                settings.INTERACTION_ORDER_COL: 3,
            },
        ]
    )
    dm = _build_synthetic_dm(train_df, val_df, test_df, num_items=5)

    test_history = dm.next_item_hist_by_split["test"]
    assert test_history.items[0, :2].tolist() == [1, 2]
    assert test_history.items[1, :3].tolist() == [1, 2, 3]


def test_test_history_inherits_non_relevant_events_from_train_and_val():
    train_df = pd.DataFrame(
        [
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 0,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-01T00:00:00Z",
                "ctx_value": 0.1,
                settings.INTERACTION_ORDER_COL: 0,
            },
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 2,
                settings.RELEVANT_COL: 0,
                settings.TIME_COL: "2024-01-02T00:00:00Z",
                "ctx_value": 0.2,
                settings.INTERACTION_ORDER_COL: 1,
            },
        ]
    )
    val_df = pd.DataFrame(
        [
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 1,
                settings.RELEVANT_COL: 0,
                settings.TIME_COL: "2024-01-03T00:00:00Z",
                "ctx_value": 0.3,
                settings.INTERACTION_ORDER_COL: 2,
            }
        ]
    )
    test_df = pd.DataFrame(
        [
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 3,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-04T00:00:00Z",
                "ctx_value": 0.4,
                settings.INTERACTION_ORDER_COL: 3,
            }
        ]
    )
    dm = _build_synthetic_dm(train_df, val_df, test_df, num_items=5)

    test_history = dm.next_item_hist_by_split["test"]
    assert test_history.valid_mask[0, :3].tolist() == [True, True, True]
    assert test_history.items[0, :3].tolist() == [1, 3, 2]


def test_history_fallback_uses_interaction_order_without_timestamp():
    train_df = pd.DataFrame(
        [
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 2,
                settings.RELEVANT_COL: 1,
                "ctx_value": 0.3,
                settings.INTERACTION_ORDER_COL: 2,
            },
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 0,
                settings.RELEVANT_COL: 1,
                "ctx_value": 0.1,
                settings.INTERACTION_ORDER_COL: 0,
            },
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 1,
                settings.RELEVANT_COL: 1,
                "ctx_value": 0.2,
                settings.INTERACTION_ORDER_COL: 1,
            },
        ]
    )
    empty_df = train_df.iloc[0:0].copy()
    dm = _build_synthetic_dm(train_df, empty_df, empty_df, num_items=4)

    train_history = dm.next_item_hist_by_split["train"]
    assert train_history.valid_mask[1].sum().item() == 0
    assert train_history.items[2, :1].tolist() == [1]
    assert train_history.items[0, :2].tolist() == [1, 2]


def test_history_truncates_to_max_history_len(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "MAX_HISTORY_LEN", 2)

    train_df = pd.DataFrame(
        [
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 0,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-01T00:00:00Z",
                "ctx_value": 0.1,
                settings.INTERACTION_ORDER_COL: 0,
            },
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 1,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-02T00:00:00Z",
                "ctx_value": 0.2,
                settings.INTERACTION_ORDER_COL: 1,
            },
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 2,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-03T00:00:00Z",
                "ctx_value": 0.3,
                settings.INTERACTION_ORDER_COL: 2,
            },
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 3,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-04T00:00:00Z",
                "ctx_value": 0.4,
                settings.INTERACTION_ORDER_COL: 3,
            },
        ]
    )
    empty_df = train_df.iloc[0:0].copy()
    dm = _build_synthetic_dm(train_df, empty_df, empty_df, num_items=5)

    train_history = dm.next_item_hist_by_split["train"]
    assert train_history.items.shape[1] == 2
    assert train_history.items[3, :2].tolist() == [2, 3]


def test_history_truncates_with_mixed_relevant_and_non_relevant_events(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings, "MAX_HISTORY_LEN", 2)

    train_df = pd.DataFrame(
        [
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 0,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-01T00:00:00Z",
                "ctx_value": 0.1,
                settings.INTERACTION_ORDER_COL: 0,
            },
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 1,
                settings.RELEVANT_COL: 0,
                settings.TIME_COL: "2024-01-02T00:00:00Z",
                "ctx_value": 0.2,
                settings.INTERACTION_ORDER_COL: 1,
            },
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 2,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-03T00:00:00Z",
                "ctx_value": 0.3,
                settings.INTERACTION_ORDER_COL: 2,
            },
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 3,
                settings.RELEVANT_COL: 1,
                settings.TIME_COL: "2024-01-04T00:00:00Z",
                "ctx_value": 0.4,
                settings.INTERACTION_ORDER_COL: 3,
            },
        ]
    )
    empty_df = train_df.iloc[0:0].copy()
    dm = _build_synthetic_dm(train_df, empty_df, empty_df, num_items=5)

    train_history = dm.next_item_hist_by_split["train"]
    assert train_history.items.shape[1] == 2
    assert train_history.items[3, :2].tolist() == [2, 3]
    assert train_history.ctx[3, 0, 0].item() == pytest.approx(0.2)
    assert train_history.ctx[3, 1, 0].item() == pytest.approx(0.3)


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
