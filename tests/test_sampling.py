import argparse
from dataclasses import replace

import pandas as pd
import pytest

from edurec import settings
from edurec.datasets import DataConfig, DatasetName, ProcessedData
from edurec.datasets.preprocessing.build import build_processed
from edurec.datasets.preprocessing.sampling import (
    prune_concepts,
    reduce_dataset,
    sample_users,
)
from edurec.datasets.preprocessing.splits import split_data
from scripts.common import add_preprocessing_args, data_config


@pytest.fixture
def sampling_raw(make_raw):
    raw = make_raw(
        {u: list(range((4, 8, 16)[u % 3])) for u in range(90)},
        {i: ["shared", *[f"{i}-{j}" for j in range(6)]] for i in range(18)},
    )
    raw.item_features["field"] = [[f"field-{i % 3}"] for i in range(18)]
    raw.schema["items"]["list"] = ["field"]
    return raw._replace(
        graph_entities={
            "concept": pd.DataFrame(
                {
                    "id": [
                        "shared",
                        *[f"{i}-{j}" for i in range(18) for j in range(6)],
                    ],
                }
            )
        }
    )


@pytest.fixture
def sampling_config():
    return DataConfig(
        dataset=DatasetName.MOOCCUBEX,
        remove_sparse=False,
        test_ratio=0.2,
        val_ratio=0.2,
        random_state=42,
        max_interactions=350,
        sampling_seed=42,
        max_exclusive_concepts=2,
    )


def test_reduction_preserves_original_splits_and_catalog(
    sampling_raw,
    sampling_config,
    tmp_path,
):
    original = split_data(
        sampling_raw.interactions,
        0.2,
        0.2,
        sampling_config.effective_min_interactions,
        42,
    )
    reduced, splits = reduce_dataset(sampling_raw, original, sampling_config)
    selected = set(reduced.user_features[settings.USER_COL])
    assert 0 < len(selected) < 90
    for before, after in zip(original, splits):
        pd.testing.assert_frame_equal(
            before.loc[before[settings.USER_COL].isin(selected)].reset_index(drop=True),
            after,
        )
    pd.testing.assert_frame_equal(
        sampling_raw.item_features.drop(columns="concept"),
        reduced.item_features.drop(columns="concept"),
    )
    assert len(reduced.interactions) == sum(map(len, splits))

    result = build_processed(sampling_raw, sampling_config)
    assert result.data.num_items == 18  # Includes courses never seen in train.
    assert result.data.num_interactions < len(sampling_raw.interactions)
    report = result.data.reports["reduction_report"]
    assert report["sampling"]["interactions_after"] == result.data.num_interactions
    assert report["sampling"]["coverage_after"]["courses_without_train"] >= 2
    assert report["concepts"]["shared_before"] == report["concepts"]["shared_after"]
    result.data.save(tmp_path / "reduced")
    loaded = ProcessedData.load(tmp_path / "reduced")
    assert loaded.reports == result.data.reports
    assert loaded.manifest == result.data.manifest


def test_selection_ignores_held_out_values_and_lengths(sampling_raw, sampling_config):
    train, val, test = split_data(sampling_raw.interactions, 0.2, 0.2, 5, 42)
    reduced, _ = reduce_dataset(sampling_raw, (train, val, test), sampling_config)
    # Change test popularity, fields observed there, ratings, and history lengths.
    future = pd.concat([test] * 10, ignore_index=True).assign(item_id=17, rating=999)
    changed, _ = reduce_dataset(sampling_raw, (train, val, future), sampling_config)
    pd.testing.assert_frame_equal(reduced.user_features, changed.user_features)
    pd.testing.assert_frame_equal(reduced.item_features, changed.item_features)


def test_stratified_sampling_is_reproducible_and_retains_activity_groups(
    sampling_raw,
    sampling_config,
):
    train, _, _ = split_data(sampling_raw.interactions, 0.2, 0.2, 5, 42)
    selected = sample_users(train, sampling_config)
    assert selected == sample_users(
        train.sample(frac=1, random_state=7), sampling_config
    )
    assert selected != sample_users(train, replace(sampling_config, sampling_seed=7))
    counts = train.groupby(settings.USER_COL).size()
    assert set(counts.loc[list(selected)]) == set(counts)
    assert selected <= sample_users(
        train, replace(sampling_config, max_interactions=500)
    )
    assert sample_users(train, replace(sampling_config, max_interactions=10000)) == set(
        counts.index
    )
    with pytest.raises(ValueError, match="too small"):
        sample_users(train, replace(sampling_config, max_interactions=1))


@pytest.mark.parametrize("limit", [0, 2, 100])
def test_concept_pruning_preserves_shared_nodes_and_other_relations(
    sampling_raw, limit
):
    reduced, report = prune_concepts(sampling_raw, limit, 42)
    expected_per_course = 1 + min(limit, 6)
    assert reduced.item_features["concept"].map(len).eq(expected_per_course).all()
    assert reduced.item_features["concept"].map(lambda tokens: "shared" in tokens).all()
    assert report["after"] == 1 + 18 * min(limit, 6)
    assert report["edges_after"] == 18 * expected_per_course
    assert len(reduced.graph_entities["concept"]) == report["after"]
    pd.testing.assert_series_equal(
        reduced.item_features["field"],
        sampling_raw.item_features["field"],
    )
    # Raw source tables and entity dictionaries remain untouched.
    assert sampling_raw.item_features["concept"].map(len).eq(7).all()
    assert len(sampling_raw.graph_entities["concept"]) == 109
    reordered = sampling_raw._replace(
        item_features=sampling_raw.item_features.iloc[::-1],
    )
    other, _ = prune_concepts(reordered, limit, 42)
    assert other.item_features.set_index(settings.ITEM_COL)["concept"].to_dict() == (
        reduced.item_features.set_index(settings.ITEM_COL)["concept"].to_dict()
    )


def test_no_reduction_is_a_noop(sampling_raw, sampling_config):
    config = replace(
        sampling_config, max_interactions=None, max_exclusive_concepts=None
    )
    splits = split_data(sampling_raw.interactions, 0.2, 0.2, 5, 42)
    raw, result = reduce_dataset(sampling_raw, splits, config)
    assert raw is sampling_raw
    assert result is splits


@pytest.mark.parametrize(
    "options",
    [
        {"max_interactions": 0},
        {"max_exclusive_concepts": -1},
        {"sampling_seed": -1},
    ],
)
def test_invalid_reduction_options(options):
    with pytest.raises(ValueError):
        DataConfig(dataset=DatasetName.MOOCCUBEX, **options)


def test_cli_and_cache_identity_include_reduction_options(sampling_config):
    parser = argparse.ArgumentParser()
    add_preprocessing_args(parser)
    args = parser.parse_args(
        [
            "--max-interactions",
            "500000",
            "--sampling-seed",
            "7",
            "--max-exclusive-concepts",
            "10",
            "--no-remove-sparse",
        ]
    )
    config = data_config(args, DatasetName.MOOCCUBEX)
    assert config.max_interactions == 500000
    assert config.max_exclusive_concepts == 10
    assert config.sampling_seed == 7
    assert not config.remove_sparse
    manifest = sampling_config.manifest("temporal")
    for change in (
        {"sampling_seed": 7},
        {"max_interactions": 500},
        {"max_exclusive_concepts": 10},
    ):
        assert manifest.incompatibility(
            replace(sampling_config, **change).manifest("temporal")
        )
