import pandas as pd
import pytest
import torch

from edurec import settings
from edurec.datasets.adapters.queries import build_histories
from edurec.datasets.pipeline.prepare import (
    deduplicate_interactions,
    filter_sparse,
)
from edurec.datasets.pipeline.splits import split_data
from edurec.datasets.torch_data import RecSysDataset


def test_temporal_split_preserves_order_and_all_events():
    frame = pd.DataFrame(
        {
            settings.USER_COL: [0] * 5,
            settings.ITEM_COL: [4, 0, 3, 1, 2],
            settings.TIME_COL: [50, 10, 40, 20, 30],
        }
    )
    train, val, test = split_data(frame, 0.2, 0.2, 3, 42)
    assert train[settings.ITEM_COL].tolist() == [0, 1, 2]
    assert val[settings.ITEM_COL].tolist() == [3]
    assert test[settings.ITEM_COL].tolist() == [4]
    assert len(train) + len(val) + len(test) == len(frame)


def test_deduplication_keeps_latest_event():
    frame = pd.DataFrame(
        {
            settings.USER_COL: [0, 0, 0],
            settings.ITEM_COL: [1, 2, 1],
            settings.TIME_COL: [30, 20, 10],
        }
    )
    result = deduplicate_interactions(frame)
    assert result.set_index(settings.ITEM_COL)[settings.TIME_COL].to_dict() == {
        1: 30,
        2: 20,
    }
    assert len(frame) == 3


def test_histories_exclude_target_isolate_users_and_truncate_oldest():
    splits = {
        "train": pd.DataFrame(
            {settings.USER_COL: [0, 1, 0], settings.ITEM_COL: [2, 9, 4]}
        ),
        "val": pd.DataFrame({settings.USER_COL: [0], settings.ITEM_COL: [5]}),
        "test": pd.DataFrame({settings.USER_COL: [0], settings.ITEM_COL: [6]}),
    }
    histories = build_histories(splits, max_history=2)
    assert histories["train"][0].tolist() == [[0, 0], [0, 0], [3, 0]]
    assert histories["val"][0].tolist() == [[3, 5]]
    assert histories["test"][0].tolist() == [[5, 6]]
    assert histories["test"][2].tolist() == [[2, 3]]
    assert histories["train"][1].tolist() == [
        [False, False],
        [False, False],
        [True, False],
    ]


def test_dataset_gathers_context_and_preserves_second_precision_in_gaps():
    frame = pd.DataFrame({settings.USER_COL: [0], settings.ITEM_COL: [2]})
    dataset = RecSysDataset(
        frame,
        torch.tensor([[1, 2, 0]]),
        torch.tensor([[True, True, False]]),
        torch.tensor([[0, 1, -1]]),
        torch.tensor([[3.0], [7.0], [99.0]]),
        torch.tensor([[1], [2], [9]]),
        torch.tensor(
            [1_700_000_000.0, 1_700_000_003.0, 1_700_000_010.0], dtype=torch.float64
        ),
        negative_item_ids=torch.tensor([[0, 1]]),
    )
    query = dataset[0]
    assert len(dataset) == 1
    assert query.target_item_id.item() == 2
    assert query.history_dense_features.tolist() == [[3.0], [7.0], [0.0]]
    assert query.history_cat_features.tolist() == [[1], [2], [0]]
    assert query.history_delta_times.tolist() == [0.0, 3.0, 0.0]
    assert query.negative_item_ids.tolist() == [0, 1]


def test_filter_sparse_reports_a_threshold_that_removes_every_interaction():
    users = pd.DataFrame({settings.USER_COL: [0, 1]})
    items = pd.DataFrame({settings.ITEM_COL: [0, 1]})
    interactions = pd.DataFrame(
        {settings.USER_COL: [0, 0, 1], settings.ITEM_COL: [0, 1, 1]}
    )

    with pytest.raises(
        ValueError, match=r"min_interactions=3 but the densest user has 2"
    ):
        filter_sparse(users, items, interactions, min_interactions=3)
