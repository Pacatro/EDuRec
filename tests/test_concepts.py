from dataclasses import replace

import pandas as pd
import pytest

from edurec.datasets import DataConfig, DatasetName, RawData
from edurec.datasets.preprocessing.build import build_processed
from edurec.datasets.preprocessing.reduction import (
    apply_common_filters,
    reduce_dataset,
)


def _raw(concepts: dict[int, list[str]]) -> RawData:
    return RawData(
        interactions=pd.DataFrame(),
        item_features=pd.DataFrame(
            {"item_id": list(concepts), "concept": list(concepts.values())}
        ),
        user_features=pd.DataFrame({"user_id": []}),
        schema={"items": {"graph_list": ["concept"]}},
        graph_entities={
            "concept": pd.DataFrame(
                {
                    "id": sorted({c for values in concepts.values() for c in values}),
                    "name": ["nombre"] * len(
                        {c for values in concepts.values() for c in values}
                    ),
                }
            )
        },
    )


def _config(**overrides) -> DataConfig:
    defaults = {
        "dataset": DatasetName.COCO,
        "random_state": 42,
        "min_interactions": 3,
    }
    return DataConfig(**{**defaults, **overrides})


def _splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    empty = pd.DataFrame()
    return empty, empty.copy(), empty.copy()


def _pruned(raw: RawData, config: DataConfig) -> RawData:
    result, _ = reduce_dataset(raw, _splits(), config)
    return result


def _concepts(raw: RawData) -> list[list[str]]:
    return raw.item_features["concept"].tolist()


def _entity_ids(raw: RawData) -> list[str]:
    entities = raw.graph_entities
    assert entities is not None
    return entities["concept"]["id"].tolist()


def _pruning_report(raw: RawData) -> dict:
    reports = raw.reports
    assert reports is not None
    return reports["reduction_report"]["concepts"]


def test_min_support_removes_rare_concepts_and_prunes_metadata():
    raw = _raw({1: ["shared", "x"], 2: ["shared", "y"], 3: ["z"]})

    result = _pruned(raw, _config(min_concept_support=2))

    assert _concepts(result) == [["shared"], ["shared"], []]
    assert _entity_ids(result) == ["shared"]
    report = _pruning_report(result)
    assert report["concepts_before"] == 4
    assert report["concepts_after"] == 1
    assert report["edges_before"] == 5
    assert report["edges_after"] == 2
    assert report["courses_without_concepts_after"] == 1


def test_max_support_removes_ubiquitous_concepts():
    raw = _raw({1: ["shared", "x"], 2: ["shared", "y"], 3: ["z"]})

    result = _pruned(raw, _config(max_concept_support=1))

    assert _concepts(result) == [["x"], ["y"], ["z"]]
    assert _entity_ids(result) == ["x", "y", "z"]


def test_per_course_cap_ranks_by_support_and_is_reproducible():
    raw = _raw(
        {
            1: ["shared", "x1", "x2", "x3"],
            2: ["shared", "y1"],
        }
    )
    config = _config(max_concepts_per_course=2)

    first = _pruned(raw, config)
    second = _pruned(raw, config)

    assert _concepts(first) == _concepts(second)
    assert _concepts(first)[0][0] == "shared"
    assert len(_concepts(first)[0]) == 2
    assert _concepts(first)[1] == ["shared", "y1"]
    assert _pruning_report(first)["edges_after"] == 4


def test_cap_zero_removes_all_concepts():
    raw = _raw({1: ["a", "b"], 2: ["a"]})

    result = _pruned(raw, _config(max_concepts_per_course=0))

    assert _concepts(result) == [[], []]
    assert _entity_ids(result) == []
    assert _pruning_report(result)["concepts_after"] == 0


def test_disabled_policy_is_a_noop():
    raw = _raw({1: ["a"], 2: ["b"]})
    splits = _splits()

    result, result_splits = reduce_dataset(raw, splits, _config())

    assert result is raw
    assert result_splits is splits


def test_invalid_concept_pruning_options():
    with pytest.raises(ValueError, match="min_concept_support"):
        _config(min_concept_support=0)
    with pytest.raises(ValueError, match="max_concept_support"):
        _config(min_concept_support=3, max_concept_support=2)
    with pytest.raises(ValueError, match="max_concepts_per_course"):
        _config(max_concepts_per_course=-1)


def test_concept_pruning_invalidates_cache_identity():
    base = _config().manifest("temporal")
    changed = replace(_config(), min_concept_support=2).manifest("temporal")
    assert base.incompatibility(changed) is not None


def test_build_processed_applies_concept_pruning(make_raw):
    raw = make_raw(
        {user: [0, 1, 2, 3] for user in range(4)},
        {item: ["shared"] for item in range(4)},
    )
    raw.item_features.loc[0, "concept"] = ["unique"]

    config = _config(min_concept_support=2)
    result = build_processed(apply_common_filters(raw, config), config)

    concepts = {
        concept for values in result.item_features["concept"] for concept in values
    }
    assert concepts == {"shared"}
    assert result.reports["reduction_report"]["concepts"]["concepts_after"] == 1
