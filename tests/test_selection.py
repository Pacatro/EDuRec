import hashlib
import json

import pandas as pd
import pytest

from edurec import settings
from edurec.datasets import DatasetName, ElearningDataModule
from edurec.datasets.pipeline.prepare import select_dataset, selection_metadata
from edurec.datasets.pipeline.splits import split_data
from edurec.datasets.registry import RawData


def make_raw(histories, concepts):
    events = pd.DataFrame(
        [
            (user, item, time, 1.0)
            for user, items in histories.items()
            for time, item in enumerate(items)
        ],
        columns=[
            settings.USER_COL,
            settings.ITEM_COL,
            settings.TIME_COL,
            settings.RATING_COL,
        ],
    )
    return RawData(
        interactions=events,
        user_features=pd.DataFrame({settings.USER_COL: list(histories)}),
        item_features=pd.DataFrame(
            {settings.ITEM_COL: list(concepts), "concept": list(concepts.values())}
        ),
        schema={"items": {"graph_list": ["concept"], "community_fields": ["concept"]}},
    )


@pytest.fixture
def raw_communities():
    histories = {user: [0, 1, 2, 3] for user in range(4)}
    histories.update({user: [4, 5, 6, 7] for user in range(4, 7)})
    return make_raw(histories, {item: ["a" if item < 4 else "b"] for item in range(8)})


def test_initial_filter_removes_inactive_users_and_items(raw_communities):
    extra = make_raw({99: [0, 1, 2, 3, 99], 100: [99]}, {99: ["a"]})
    raw = raw_communities._replace(
        interactions=pd.concat(
            [raw_communities.interactions, extra.interactions], ignore_index=True
        ),
        user_features=pd.concat(
            [raw_communities.user_features, extra.user_features], ignore_index=True
        ),
        item_features=pd.concat(
            [raw_communities.item_features, extra.item_features], ignore_index=True
        ),
    )
    result = select_dataset(raw)
    assert 99 not in set(result.interactions[settings.ITEM_COL])
    assert 100 not in set(result.interactions[settings.USER_COL])
    # The initial activity filter defines the denominator: user 99 retains 4/4.
    assert result.selection_report["min_history_retention"] == 1


def test_unlimited_grows_only_one_connected_community(raw_communities):
    result = select_dataset(raw_communities)
    assert set(result.item_features[settings.ITEM_COL]) == {0, 1, 2, 3}
    assert len(result.interactions) == 16
    assert result.selection_report["connected"]
    assert result.selection_report["policy"] == "community"


def test_growth_uses_shared_users_even_without_shared_semantics():
    raw = make_raw(
        {0: [0, 2], 1: [0, 2], 2: [0], 10: [1]}, {0: ["a"], 1: ["a"], 2: ["b"]}
    )
    result = select_dataset(
        raw, max_items=2, min_interactions=1, min_history_retention=0.1
    )
    # Item 1 shares a concept, but item 2 shares two users with the seed.
    assert set(result.item_features[settings.ITEM_COL]) == {0, 2}


def test_growth_uses_semantics_even_without_shared_users():
    raw = make_raw(
        {0: [0], 1: [0], 2: [0], 10: [1], 11: [2], 12: [2]},
        {0: ["a"], 1: ["a"], 2: ["b"]},
    )
    result = select_dataset(raw, max_items=2, min_interactions=1)
    assert set(result.item_features[settings.ITEM_COL]) == {0, 1}


def test_item_budget_and_retention_preserve_original_event_values(raw_communities):
    result = select_dataset(raw_communities, max_items=3)
    assert len(result.item_features) == 3
    assert result.selection_report["min_history_retention"] == 0.75
    expected = raw_communities.interactions.loc[
        raw_communities.interactions[settings.USER_COL].isin(
            result.user_features[settings.USER_COL]
        )
        & raw_communities.interactions[settings.ITEM_COL].isin(
            result.item_features[settings.ITEM_COL]
        )
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(result.interactions, expected)
    with pytest.raises(ValueError, match="Community is empty"):
        select_dataset(raw_communities, max_items=3, min_history_retention=1)


def test_user_and_interaction_budgets_do_not_cut_histories(raw_communities):
    result = select_dataset(raw_communities, max_users=3, max_interactions=13)
    assert len(result.user_features) == 3
    assert len(result.interactions) == 12
    for user, history in result.interactions.groupby(settings.USER_COL):
        pd.testing.assert_frame_equal(
            history.reset_index(drop=True),
            raw_communities.interactions.loc[
                raw_communities.interactions[settings.USER_COL] == user
            ].reset_index(drop=True),
        )


def test_support_is_rechecked_after_budget(raw_communities):
    with pytest.raises(ValueError, match="Community is empty"):
        select_dataset(raw_communities, max_users=2)


def test_final_connectivity_uses_only_surviving_users():
    raw = make_raw(
        {0: [0, 1], 1: [0, 1], 2: [3, 4], 3: [3, 4], 4: [1, 2], 5: [1, 2]},
        {0: ["a"], 1: ["a"], 2: ["a", "b"], 3: ["b"], 4: ["b"]},
    )
    # Choose a seed whose cohort budget drops both learners supporting bridge 2.
    seed = next(
        seed
        for seed in range(1000)
        if set(
            sorted(
                range(6),
                key=lambda user: hashlib.sha256(
                    f"{seed}:int:{user}".encode()
                ).hexdigest(),
            )[:4]
        )
        == {0, 1, 2, 3}
    )
    result = select_dataset(raw, seed=seed, max_users=4, min_interactions=2)
    assert len(result.item_features) == 2
    assert result.selection_report["connected"]
    assert result.selection_report["min_item_users"] == 2
    assert result.selection_report["min_user_items"] == 2
    assert result.selection_report["min_history_retention"] == 1


def test_duplicate_events_do_not_create_distinct_user_support(raw_communities):
    events = raw_communities.interactions.loc[
        raw_communities.interactions[settings.USER_COL] == 0
    ]
    raw = raw_communities._replace(
        interactions=pd.concat([events] * 4, ignore_index=True)
    )
    with pytest.raises(ValueError, match="minimum activity"):
        select_dataset(raw)


def test_selection_is_independent_of_source_order_ratings_and_time(raw_communities):
    first = select_dataset(raw_communities, max_items=3)
    events = raw_communities.interactions.sample(frac=1, random_state=10).copy()
    events[settings.RATING_COL] = -100
    events[settings.TIME_COL] *= -1
    shuffled = raw_communities._replace(
        interactions=events,
        item_features=raw_communities.item_features.sample(frac=1, random_state=11),
    )
    second = select_dataset(shuffled, max_items=3)
    assert set(first.item_features[settings.ITEM_COL]) == set(
        second.item_features[settings.ITEM_COL]
    )
    assert set(first.user_features[settings.USER_COL]) == set(
        second.user_features[settings.USER_COL]
    )


def test_coco_categories_and_instructors_use_the_same_selector(raw_communities):
    items = raw_communities.item_features.drop(columns="concept").copy()
    items["second_level_category"] = ["a"] * 4 + ["b"] * 4
    items["instructor"] = [["i1"]] * 4 + [["i2"]] * 4
    items["language"] = "en"
    raw = raw_communities._replace(
        item_features=items,
        schema={
            "items": {
                "cat": ["second_level_category", "language"],
                "graph_list": ["instructor"],
                "community_fields": ["second_level_category", "instructor"],
            }
        },
    )
    assert len(select_dataset(raw).item_features) == 4
    assert len(select_dataset(raw, community_fields=["language"]).item_features) == 8


def test_temporal_splits_follow_selection(raw_communities):
    selected = select_dataset(raw_communities)
    train, val, test = split_data(selected.interactions, 0.2, 0.2, 3, 42)
    assert len(train) + len(val) + len(test) == len(selected.interactions)
    for user in selected.user_features[settings.USER_COL]:
        assert (
            train.loc[train[settings.USER_COL] == user, settings.TIME_COL].max()
            < val.loc[val[settings.USER_COL] == user, settings.TIME_COL].min()
        )
        assert (
            val.loc[val[settings.USER_COL] == user, settings.TIME_COL].max()
            < test.loc[test[settings.USER_COL] == user, settings.TIME_COL].min()
        )


def test_all_selection_parameters_are_part_of_cache_identity():
    params = {
        "dataset": DatasetName.COCO,
        "batch_size": 8,
        "test_ratio": 0.2,
        "val_ratio": 0.2,
    }
    base = ElearningDataModule(**params).cache_params["selection"]
    for change in [
        {"max_items": 10},
        {"max_users": 100},
        {"max_interactions": 1000},
        {"community_fields": ["instructor"]},
        {"min_history_retention": 0.9},
        {"min_interactions": 5},
        {"selection_seed": 7},
    ]:
        assert ElearningDataModule(**params, **change).cache_params["selection"] != base
    assert json.loads(json.dumps(base)) == base


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget(budget):
    with pytest.raises(ValueError, match="max_interactions"):
        selection_metadata(None, 42, budget)


def test_pipeline_and_cache_use_the_same_community_parameters(
    raw_communities, monkeypatch, tmp_path
):
    from edurec.datasets import datamodule

    monkeypatch.setattr(settings, "PROCESSED_FOLDER", str(tmp_path))
    monkeypatch.setattr(datamodule, "load_raw_data", lambda dataset: raw_communities)
    monkeypatch.setattr(ElearningDataModule, "_source_metadata", lambda self: {})
    params = {
        "dataset": DatasetName.COCO,
        "batch_size": 8,
        "test_ratio": 0.2,
        "val_ratio": 0.2,
        "max_items": 3,
        "random_state": 42,
    }
    built = ElearningDataModule(**params)
    built.setup()
    assert built.num_items == 3
    assert built.num_users == 4
    assert built.num_interactions == 12
    assert built.cache_params["selection_report"]["min_history_retention"] == 0.75
    cached = ElearningDataModule(**params, use_processed_data=True)
    assert cached._compatible_cache()
    cached.setup()
    assert cached.num_interactions == built.num_interactions
    assert (
        cached.cache_params["selection_report"]
        == built.cache_params["selection_report"]
    )
    different = ElearningDataModule(
        **{**params, "max_items": 4}, use_processed_data=True
    )
    assert not different._compatible_cache()
