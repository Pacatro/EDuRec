import argparse
from dataclasses import replace

import pandas as pd
import pytest

from edurec import settings
from edurec.datasets import DataConfig, DatasetName, ProcessedData
from edurec.datasets.preprocessing.build import build_processed
from edurec.datasets.preprocessing.reduction import (
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
    )


def test_reduction_preserves_splits_and_restricts_catalog(
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
    assert set(reduced.item_features[settings.ITEM_COL]) == set(
        reduced.interactions[settings.ITEM_COL]
    )
    assert set(reduced.graph_entities["concept"]["id"]) == {
        c for concepts in reduced.item_features["concept"] for c in concepts
    }
    assert reduced.item_features["concept"].map(len).eq(7).all()
    assert len(reduced.interactions) == sum(map(len, splits))

    result = build_processed(sampling_raw, sampling_config)
    assert result.data.num_items == 16  # Drops courses absent from selected histories.
    assert result.data.num_interactions < len(sampling_raw.interactions)
    report = result.data.reports["reduction_report"]
    assert report["sampling"]["interactions_after"] == result.data.num_interactions
    assert report["catalog"]["items_after"] == 16
    assert "concepts" not in report
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
    # User selection stays TRAIN-only; the catalog follows retained events.
    assert 17 in set(changed.item_features[settings.ITEM_COL])


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


def test_no_reduction_is_a_noop(sampling_raw, sampling_config):
    config = replace(
        sampling_config, max_interactions=None
    )
    splits = split_data(sampling_raw.interactions, 0.2, 0.2, 5, 42)
    raw, result = reduce_dataset(sampling_raw, splits, config)
    assert raw is sampling_raw
    assert result is splits


@pytest.mark.parametrize(
    "options",
    [
        {"max_interactions": 0},
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
            "--min-concept-support",
            "2",
            "--max-concept-support",
            "10",
            "--max-concepts-per-course",
            "50",
            "--no-remove-sparse",
        ]
    )
    config = data_config(args, DatasetName.MOOCCUBEX)
    assert config.max_interactions == 500000
    assert config.sampling_seed == 7
    assert config.min_concept_support == 2
    assert config.max_concept_support == 10
    assert config.max_concepts_per_course == 50
    assert not config.remove_sparse
    manifest = sampling_config.manifest("temporal")
    for change in (
        {"sampling_seed": 7},
        {"max_interactions": 500},
        {"min_concept_support": 2},
        {"max_concept_support": 10},
        {"max_concepts_per_course": 50},
    ):
        assert manifest.incompatibility(
            replace(sampling_config, **change).manifest("temporal")
        )
