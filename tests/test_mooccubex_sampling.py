import argparse
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from edurec import settings
from edurec.datasets import DataConfig, DatasetName, ElearningDataModule, ProcessedData
from edurec.datasets.preprocessing import prepare, text
from edurec.datasets.preprocessing.sampling import reduce_concepts, sample_users
from edurec.datasets.sources import mooccubex
from edurec.datasets.sources.registry import prepare_raw_data
from scripts.common import add_preprocessing_args, data_config


def test_activity_sampling_preserves_proportions_seed_and_input_order():
    sizes = {f"{size}-{i}": size for size in (1, 3, 8) for i in range(100)}
    selected, report = sample_users(sizes, 120, 42)
    reordered, _ = sample_users(dict(reversed(list(sizes.items()))), 120, 42)
    other_seed, _ = sample_users(sizes, 120, 7)

    assert selected == reordered
    assert selected != other_seed
    assert sum(sizes[u] for u in selected) == 120
    assert [s["users"] for s in report["strata"].values()] == [10, 10, 10]
    assert report["users_before"] == 300
    assert report["interactions_before"] == 1200


def test_activity_sampling_budget_edges():
    sizes = {"short": 2, "long": 7}
    assert sample_users(sizes, 100, 42)[0] == set(sizes)
    assert sample_users(sizes, 3, 42)[0] == {"short"}
    with pytest.raises(ValueError, match="complete user history"):
        sample_users(sizes, 1, 42)


def test_concept_reduction_keeps_shared_and_other_relations():
    items = pd.DataFrame(
        {
            "item_id": ["a", "b", "c"],
            "concept": [["shared", "a1", "a2", "a2"], ["shared", "b1"], []],
            "teacher": [["t1"], ["t2"], []],
            "school": [["s"], ["s"], []],
            "field": [["f"], ["g"], []],
        }
    )
    reduced, report = reduce_concepts(items, 1, 42)
    again, _ = reduce_concepts(items.iloc[::-1], 1, 42)
    pd.testing.assert_frame_equal(reduced, again.iloc[::-1])
    pd.testing.assert_frame_equal(
        reduced.drop(columns="concept"), items.drop(columns="concept")
    )
    assert "shared" in reduced.concept.iloc[0]
    assert "shared" in reduced.concept.iloc[1]
    assert reduced.concept.map(len).tolist() == [2, 2, 0]
    assert items.concept.iloc[0] == ["shared", "a1", "a2", "a2"]
    assert report["concepts_before"] == 4
    assert report["concepts"] == 3
    shared_only, _ = reduce_concepts(items, 0, 42)
    assert shared_only.concept.tolist() == [["shared"], ["shared"], []]


@pytest.fixture
def sampling_source(tmp_path, monkeypatch):
    entities = tmp_path / "raw/mooccubex/entities"
    relations = entities.parent / "relations"
    entities.mkdir(parents=True)
    relations.mkdir()
    courses = [
        {
            "id": f"C_{i}",
            "name": f"Course {i}",
            "field": ["area"],
            "about": "",
            "prerequisites": "",
            "resource": [{"resource_id": f"V_{i}"}],
        }
        for i in range(7)
    ]
    records = []
    for size in (1, 3, 6):
        for i in range(20):
            records.append(
                {
                    "id": f"u-{size}-{i}",
                    "gender": i % 2,
                    "school": "school",
                    "year_of_birth": 1990,
                    "course_order": list(range(size)),
                    "enroll_time": [
                        f"2020-01-{j + 1:02} 00:00:00" for j in range(size)
                    ],
                }
            )
    # First record wins, even when a duplicate has a different history.
    records.append({**records[0], "course_order": [6], "enroll_time": ["2021-01-01"]})
    records.append(
        {
            "id": "extreme",
            "course_order": list(range(7)),
            "enroll_time": ["2020-01-01"] * 7,
        }
    )
    for name, rows in [("course", courses), ("user", records), ("concept", [])]:
        (entities / f"{name}.json").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    (relations / "concept-course.txt").write_text(
        "".join(
            f"shared\tC_{i}\nprivate-{i}-a\tC_{i}\nprivate-{i}-b\tC_{i}\n"
            for i in range(7)
        ),
        encoding="utf-8",
    )
    for field, entity in [("teacher", "T"), ("school", "S")]:
        (relations / f"course-{field}.txt").write_text(
            "".join(f"C_{i}\t{entity}_{i}\n" for i in range(7)), encoding="utf-8"
        )
    monkeypatch.setattr(settings, "RAW_DATA_FOLDER", tmp_path / "raw")
    monkeypatch.setattr(settings, "MOOCCUBEX_MAX_USER_ENROLLMENTS", 6)
    return entities


def test_source_sampling_is_early_and_preserves_complete_histories(
    sampling_source,
    monkeypatch,
):
    full = mooccubex.load_mooccubex()
    config = DataConfig(
        dataset=DatasetName.MOOCCUBEX,
        max_interactions=40,
        max_exclusive_concepts=1,
        remove_sparse=False,
    )
    original_load = mooccubex._load_mooccubex_enrollments

    def selected_only(path, catalog, selected_users=None):
        assert selected_users is not None
        assert len(selected_users) < 60
        return original_load(path, catalog, selected_users)

    monkeypatch.setattr(mooccubex, "_load_mooccubex_enrollments", selected_only)
    # The source selects users before loading metadata. Common preparation
    # must not make another selection after support filtering.
    monkeypatch.setattr(
        prepare, "limit_interactions", lambda *a: pytest.fail("second budget selection")
    )
    raw = prepare_raw_data(config)
    users = set(raw.user_features.user_id)
    expected = full.interactions.loc[full.interactions.user_id.isin(users)]
    pd.testing.assert_frame_equal(raw.interactions, expected.reset_index(drop=True))
    assert 0 < len(raw.interactions) <= 40
    assert set(raw.item_features.item_id) == set(raw.interactions.item_id)
    assert "C_6" not in set(raw.item_features.item_id)
    assert "extreme" not in users
    assert raw.item_features.concept.map(len).le(2).all()
    assert raw.reports["source_sampling"]["users_before"] == 60
    assert raw.reports["source_sampling"]["interactions_before"] == 200


def test_sampled_datamodule_cache_reports_splits_and_entities(
    sampling_source,
    monkeypatch,
):
    config = DataConfig(
        dataset=DatasetName.MOOCCUBEX,
        max_interactions=40,
        max_exclusive_concepts=1,
        remove_sparse=False,
    )
    # Exercise entity metadata pruning, without downloading any text encoder.
    monkeypatch.setattr(settings, "MOOCCUBEX_ENCODE_CONCEPT_TEXT", True)
    encoded_lengths = []

    def encode(docs):
        encoded_lengths.append(len(docs))
        return np.zeros((len(docs), 4), dtype=np.float32)

    monkeypatch.setattr(text, "encode_documents", encode)
    dm = ElearningDataModule(config, batch_size=4)
    with pytest.warns(UserWarning, match="lack metadata"):
        dm._ensure_data()
    assert dm.data.num_interactions <= 40
    assert dm.data.num_items == dm.build.raw.interactions.item_id.nunique()
    concepts = {c for values in dm.data.item_features.concept for c in values}
    assert set(dm.data.graph_node_features["concept"].index) == concepts
    assert max(encoded_lengths) <= max(
        dm.data.num_items, len(concepts), dm.data.num_users
    )
    assert dm.data.reports["interaction_budget"]["selection"].startswith(
        "whole_users_activity"
    )
    assert dm.data.reports["concept_sampling"]["concepts"] == len(concepts)
    for user in dm.data.val.user_id.unique():
        train = dm.data.train.loc[dm.data.train.user_id == user, "timestamp"]
        val = dm.data.val.loc[dm.data.val.user_id == user, "timestamp"]
        test = dm.data.test.loc[dm.data.test.user_id == user, "timestamp"]
        assert train.max() < val.min() <= val.max() < test.min()
    loaded = ProcessedData.load(dm.processed_folder)
    assert loaded.reports == dm.data.reports
    assert loaded.manifest.incompatibility(
        replace(config, sampling_seed=7).manifest("temporal")
    )
    assert loaded.manifest.incompatibility(
        replace(config, max_exclusive_concepts=0).manifest("temporal")
    )
    cached = ElearningDataModule(config, batch_size=4, use_processed_data=True)
    monkeypatch.setattr(
        mooccubex, "_iter_mooccubex_users", lambda *a: pytest.fail("raw reread")
    )
    cached._ensure_data()
    assert cached.data.num_interactions == dm.data.num_interactions


def test_sampling_cli_and_configuration_validation():
    parser = argparse.ArgumentParser()
    add_preprocessing_args(parser)
    args = parser.parse_args(
        [
            "--max-interactions",
            "1000",
            "--sampling-seed",
            "7",
            "--max-exclusive-concepts",
            "8",
            "--no-remove-sparse",
        ]
    )
    config = data_config(args, DatasetName.MOOCCUBEX)
    assert config.sampling_seed == 7
    assert config.max_exclusive_concepts == 8
    assert not config.remove_sparse
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError, match="nonnegative"):
            replace(config, max_exclusive_concepts=value)
    with pytest.raises(ValueError, match="only supported"):
        replace(config, dataset=DatasetName.COCO)
    with pytest.raises(TypeError, match="integer"):
        replace(config, sampling_seed=True)
