import pandas as pd
import pytest
import torch

from edurec import settings
from edurec.datasets import RecSysDataset
from edurec.datasets.preprocessing.prepare import (
    deduplicate_interactions,
    filter_sparse,
)
from edurec.datasets.preprocessing.splits import split_data
from edurec.datasets.queries import build_histories


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


@pytest.mark.parametrize(
    "dataset, module_name, loader_name",
    [
        ("doris", "doris", "load_doris"),
        ("itm", "itm", "load_itm"),
        ("coco", "coco", "load_coco"),
        ("explicit_mars", "mars", "load_mars"),
        ("implicit_mars", "mars", "load_mars"),
    ],
)
def test_source_preparation_applies_shared_filters(
    dataset, module_name, loader_name, raw_communities, monkeypatch
):
    from edurec.datasets import DataConfig, DatasetName, sources
    from edurec.datasets.sources.registry import prepare_raw_data

    raw = raw_communities
    events = raw.interactions.copy()
    # Add an isolated user/item that shared support filtering must remove.
    events.loc[len(events)] = [99, 99, 0, 1.0]
    users = pd.concat([raw.user_features, pd.DataFrame({settings.USER_COL: [99]})])
    items = pd.concat(
        [raw.item_features, pd.DataFrame({settings.ITEM_COL: [99], "concept": [[]]})]
    )
    if dataset == "doris":
        events[settings.TIME_COL] = "20-21"
    elif module_name == "mars":
        events[settings.TIME_COL] = "2020-01-01T00:00:00Z"
    loaded = raw._replace(interactions=events, user_features=users, item_features=items)
    module = getattr(sources, module_name)
    monkeypatch.setattr(module, loader_name, lambda *a, **kw: loaded)

    prepared = prepare_raw_data(DataConfig(dataset=DatasetName(dataset)))

    assert len(prepared.interactions) == len(raw.interactions)
    assert prepared.user_features.user_id.tolist() == raw.user_features.user_id.tolist()
    assert prepared.item_features.item_id.tolist() == raw.item_features.item_id.tolist()
    # Preparation must not modify the loaded tables in place.
    assert len(loaded.interactions) == len(raw.interactions) + 1


def test_doris_preparation_discards_invalid_scores_and_years(make_raw, monkeypatch):
    from edurec.datasets import DataConfig, DatasetName
    from edurec.datasets.sources import doris

    raw = make_raw({0: [0, 1, 2, 3]}, {i: [] for i in range(4)})
    raw.interactions[settings.RATING_COL] = [4, "invalid", 3, None]
    raw.interactions[settings.TIME_COL] = ["20-21", "20-21", "invalid", "20-21"]
    monkeypatch.setattr(doris, "load_doris", lambda: raw)

    prepared = doris.prepare_doris(
        DataConfig(dataset=DatasetName.DORIS, remove_sparse=False)
    )

    assert prepared.interactions.item_id.tolist() == [0]
    assert prepared.interactions.timestamp.tolist() == [2020]
    assert len(raw.interactions) == 4


@pytest.mark.parametrize("mode", ["explicit", "implicit"])
def test_mars_preparation_discards_invalid_and_negative_dates(
    make_raw, monkeypatch, mode
):
    from edurec.datasets import DataConfig, DatasetName
    from edurec.datasets.sources import mars
    from edurec.datasets.sources.registry import prepare_raw_data

    raw = make_raw({0: [0, 1, 2, 3]}, {i: [] for i in range(4)})
    raw.interactions[settings.TIME_COL] = ["2020-01-01", "invalid", "1960-01-01", None]
    monkeypatch.setattr(mars, "load_mars", lambda data_type: raw)

    with pytest.warns(UserWarning, match="Dropping 3 MARS interactions"):
        prepared = prepare_raw_data(
            DataConfig(dataset=DatasetName(f"{mode}_mars"), remove_sparse=False)
        )

    assert prepared.interactions.item_id.tolist() == [0]
    assert prepared.interactions.timestamp.tolist() == [1577836800]


def test_coco_review_loading_discards_missing_fields_and_invalid_dates(
    tmp_path, monkeypatch
):
    from edurec.datasets.sources import coco

    reviews = pd.DataFrame(
        {
            "learner_id": [1, None, 1, 1, 1, 1],
            "course_id": [1, 2, None, 4, 5, 6],
            "learner_rating": [4, 4, 4, None, 4, 4],
            "learner_timestamp": ["2020-01-01"] * 4 + [None, "invalid"],
        }
    )
    monkeypatch.setattr(coco.pd, "read_csv", lambda *a, **kw: reviews.copy())

    interactions = coco._load_coco_interactions(tmp_path)

    assert interactions.item_id.tolist() == [1]
    assert interactions.timestamp.tolist() == [1577836800]


@pytest.mark.parametrize(
    "temporal, deduplicate, expected",
    [(True, False, 4), (True, True, 3), (False, False, 3)],
)
def test_common_preparation_preserves_deduplication_policy(
    make_raw, temporal, deduplicate, expected
):
    from edurec.datasets import DataConfig, DatasetName
    from edurec.datasets.preprocessing.prepare import apply_common_filters

    raw = make_raw({0: [0, 1, 2, 0]}, {i: [] for i in range(3)})
    if not temporal:
        raw = raw._replace(
            interactions=raw.interactions.drop(columns=settings.TIME_COL)
        )
    prepared = apply_common_filters(
        raw,
        DataConfig(
            dataset=DatasetName.COCO, remove_sparse=False, deduplicate=deduplicate
        ),
    )
    assert len(prepared.interactions) == expected
    if temporal and deduplicate:
        assert (
            prepared.interactions.loc[
                prepared.interactions.item_id == 0, "timestamp"
            ].item()
            == 3
        )


def test_sampling_report_does_not_control_common_budget(make_raw):
    from edurec.datasets import DataConfig, DatasetName
    from edurec.datasets.preprocessing.prepare import apply_common_filters

    raw = make_raw({u: list(range(5)) for u in range(4)}, {i: [] for i in range(5)})
    raw = raw._replace(reports={"source_sampling": {"selection": "informational"}})
    config = DataConfig(
        dataset=DatasetName.COCO, max_interactions=5, remove_sparse=False
    )

    prepared = apply_common_filters(raw, config)

    assert len(prepared.interactions) == 5
    assert prepared.reports["interaction_budget"]["selection"] == "whole_users_sha256"
    assert "interaction_budget" not in raw.reports
