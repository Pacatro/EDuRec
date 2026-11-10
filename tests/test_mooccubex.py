import json
from pathlib import Path

import pandas as pd
import pytest

from edurec import settings
from edurec.datasets import DataConfig, DatasetName
from edurec.datasets.preprocessing.build import build_processed
from edurec.datasets.preprocessing.reduction import apply_common_filters
from edurec.datasets.sources import mooccubex


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


@pytest.fixture
def raw_mooccubex(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    entities = root / "mooccubex" / "entities"
    relations = root / "mooccubex" / "relations"
    entities.mkdir(parents=True)
    relations.mkdir(parents=True)

    courses = [
        {
            "id": "C_1",
            "name": "Curso real",
            "field": ["A"],
            "prerequisites": "",
            "about": "",
            "resource": [{"resource_id": "V_1"}],
        },
        {
            "id": "C_2",
            "name": "测试课程",
            "field": ["A"],
            "prerequisites": "",
            "about": "",
            "resource": [{"resource_id": "V_2"}],
        },
        {
            "id": "C_3",
            "name": "Sin recursos",
            "field": ["A"],
            "prerequisites": "",
            "about": "",
            "resource": [],
        },
        {
            "id": "C_4",
            "name": "Solo lectura",
            "field": ["A"],
            "prerequisites": "",
            "about": "",
            "resource": [{"resource_id": "R_1"}],
        },
        {
            "id": "C_5",
            "name": "软件测试",
            "field": ["A"],
            "prerequisites": "",
            "about": "",
            "resource": [{"resource_id": "Ex_1"}],
        },
        {
            "id": "C_6",
            "name": "Otro curso",
            "field": ["A"],
            "prerequisites": "",
            "about": "",
            "resource": [{"resource_id": "V_3"}],
        },
    ]
    users = [
        {
            "id": "U_1",
            "name": "Uno",
            "gender": 1,
            "school": "S_1",
            "year_of_birth": 1990,
            "course_order": [1, 2, 3, 4, 5, 6],
            "enroll_time": [f"2020-01-0{day} 00:00:00" for day in range(1, 7)],
        },
        {
            "id": "U_2",
            "name": "Dos",
            "gender": 0,
            "school": "S_1",
            "year_of_birth": 1991,
            "course_order": [1],
            "enroll_time": ["2020-01-01 00:00:00"],
        },
        {
            "id": "U_3",
            "name": "Tres",
            "gender": 0,
            "school": "S_1",
            "year_of_birth": 1992,
            "course_order": [3],
            "enroll_time": ["2020-01-01 00:00:00"],
        },
        {
            "id": "U_4",
            "name": "Cuatro",
            "gender": 0,
            "school": "S_1",
            "year_of_birth": 1993,
            "course_order": [2],
            "enroll_time": ["2020-01-01 00:00:00"],
        },
        {
            "id": "U_5",
            "name": "Cinco",
            "gender": 0,
            "school": "S_1",
            "year_of_birth": 1994,
            "course_order": [5, 6],
            "enroll_time": ["2020-01-01 00:00:00", "2020-01-02 00:00:00"],
        },
        {
            "id": "U_6",
            "name": "Seis",
            "gender": 0,
            "school": "S_1",
            "year_of_birth": 1995,
            "course_order": [4],
            "enroll_time": ["2020-01-01 00:00:00"],
        },
    ]
    concepts = [
        {"id": "K_1", "name": "Concepto 1", "context": ""},
        {"id": "K_5", "name": "Concepto 5", "context": ""},
        {"id": "K_6", "name": "Concepto 6", "context": ""},
    ]

    _write_jsonl(entities / "course.json", courses)
    _write_jsonl(entities / "user.json", users)
    _write_jsonl(entities / "concept.json", concepts)
    (relations / "concept-course.txt").write_text(
        "K_1\tC_1\nK_2\tC_2\nK_5\tC_5\nK_6\tC_6\n", encoding="utf-8"
    )
    (relations / "course-teacher.txt").write_text(
        "C_1\tT_1\nC_2\tT_2\nC_5\tT_5\nC_6\tT_6\n", encoding="utf-8"
    )
    (relations / "course-school.txt").write_text(
        "C_1\tS_1\nC_2\tS_1\nC_5\tS_1\nC_6\tS_1\n", encoding="utf-8"
    )

    monkeypatch.setattr(settings, "RAW_DATA_FOLDER", root)
    monkeypatch.setattr(settings, "MOOCCUBEX_MAX_USER_ENROLLMENTS", 2)
    return mooccubex.load_mooccubex()


def test_mooccubex_source_filters_courses_users_and_extreme_enrollments(
    raw_mooccubex,
):
    raw = raw_mooccubex

    assert set(raw.item_features[settings.ITEM_COL]) == {"C_1", "C_5", "C_6"}
    assert set(raw.user_features[settings.USER_COL]) == {"U_2", "U_5"}
    assert len(raw.interactions) == 3
    # A real course about software testing must survive the placeholder filter.
    assert "C_5" in set(raw.item_features[settings.ITEM_COL])
    assert raw.item_features.set_index(settings.ITEM_COL).loc["C_1", "teacher"] == [
        "T_1"
    ]
    assert raw.item_features.set_index(settings.ITEM_COL).loc["C_1", "concept"] == [
        "K_1"
    ]


def test_mooccubex_defaults_to_five_connections(make_raw):
    assert DataConfig(dataset=DatasetName.MOOCCUBEX).effective_min_interactions == 5

    histories = {user: [0, 1, 2, 3, 4] for user in range(10)}
    raw = make_raw(histories, {item: [] for item in range(5)})
    config = DataConfig(dataset=DatasetName.MOOCCUBEX, random_state=42)

    result = build_processed(apply_common_filters(raw, config), config)
    again = build_processed(apply_common_filters(raw, config), config)

    assert result.num_users == 10
    assert result.num_items == 5
    assert result.num_interactions == 50
    assert result.manifest is not None
    assert result.manifest.config["min_interactions"] == 5
    for split, frame in result.splits().items():
        pd.testing.assert_frame_equal(frame, again.splits()[split])


def test_source_filter_settings_invalidate_the_manifest(monkeypatch):
    config = DataConfig(dataset=DatasetName.MOOCCUBEX)
    manifest = config.manifest("temporal")
    monkeypatch.setattr(settings, "MOOCCUBEX_MAX_USER_ENROLLMENTS", 50)

    assert manifest.incompatibility(config.manifest("temporal")) is not None


def test_course_streaming_keeps_first_duplicate_and_required_columns(tmp_path):
    path = tmp_path / "course.json"
    _write_jsonl(
        path,
        [
            {
                "id": "C_1",
                "name": "Curso real",
                "field": ["A"],
                "prerequisites": "",
                "about": "",
                "resource": [{"resource_id": "V_1"}],
                "unused": "not loaded",
            },
            {
                "id": "C_1",
                "name": "Duplicado",
                "resource": [{"resource_id": "V_2"}],
            },
            {
                "id": "C_2",
                "name": "测试课程",
                "resource": [{"resource_id": "V_3"}],
            },
            {"id": "C_3", "name": "Sin recursos", "resource": []},
        ],
    )

    items = mooccubex._load_mooccubex_courses(path)

    assert list(items.columns) == [
        settings.ITEM_COL,
        "name",
        "field",
        "prerequisites",
        "about",
        "resource",
    ]
    assert set(items[settings.ITEM_COL]) == {"C_1"}
    assert items.iloc[0]["name"] == "Curso real"


def test_streaming_reduction_loads_only_selected_courses_and_entities(
    raw_mooccubex, monkeypatch,
):
    from edurec.datasets.preprocessing.reduction import sample_users
    from edurec.datasets.preprocessing.splits import split_data

    entities = settings.RAW_DATA_FOLDER / "mooccubex" / "entities"
    monkeypatch.setattr(settings, "MOOCCUBEX_MAX_USER_ENROLLMENTS", 100)
    monkeypatch.setattr(settings, "MOOCCUBEX_ENCODE_CONCEPT_TEXT", True)
    # Distinct course sets make it possible to observe real graph reduction.
    courses = json.loads((entities / "course.json").read_text().splitlines()[0])
    records = []
    for i in range(12):
        record = dict(courses, id=f"C_{i + 10}", name=f"Course {i}")
        records.append(record)
    _write_jsonl(entities / "course.json", records)
    _write_jsonl(entities / "user.json", [
        {
            "id": f"U_{u}", "course_order": list(range(10 + u % 4 * 3, 13 + u % 4 * 3)),
            "enroll_time": [f"2020-01-0{d}" for d in (1, 2, 3)],
        }
        for u in range(12)
    ])
    relations = entities.parent / "relations"
    (relations / "concept-course.txt").write_text("".join(
        f"K_{i}\tC_{i}\n" for i in range(10, 22)
    ))
    _write_jsonl(entities / "concept.json", [
        {"id": f"K_{i}", "name": str(i), "context": ""} for i in range(10, 22)
    ])
    config = DataConfig(
        dataset=DatasetName.MOOCCUBEX, min_interactions=3,
        max_interactions=5, remove_sparse=False, random_state=42,
    )
    full = mooccubex.load_mooccubex()
    splits = split_data(full.interactions, config.test_ratio, config.val_ratio, 3, 42)
    expected = sample_users(splits[0], config)
    calls = []
    original = mooccubex._load_course_relations

    def observe(path, course_ids, course_column):
        calls.append(set(course_ids))
        return original(path, course_ids, course_column)

    monkeypatch.setattr(mooccubex, "_load_course_relations", observe)
    reduced = mooccubex.prepare_mooccubex(config)
    assert reduced.source_reduced
    assert set(reduced.user_features[settings.USER_COL]) == expected
    expected_events = full.interactions.loc[full.interactions[settings.USER_COL].isin(expected)]
    pd.testing.assert_frame_equal(reduced.interactions, expected_events.reset_index(drop=True))
    catalog = set(expected_events[settings.ITEM_COL])
    assert len(catalog) < len(full.item_features)
    report = reduced.reports["reduction_report"]
    assert report["sampling"]["users_after"] == len(expected)
    assert report["catalog"]["items_after"] == len(catalog)
    assert calls == [catalog] * 3
    assert set(reduced.graph_entities["concept"]["id"]) == {
        c for values in reduced.item_features["concept"] for c in values
    }
    # Building must not sample a second time or discard held-out-only courses.
    reduced.schema["items"]["text"] = []
    reduced.schema["items"]["graph_text"] = {}
    built = build_processed(reduced, config)
    assert built.num_interactions == len(expected_events)
    assert built.num_items == len(catalog)


def test_streamed_sparse_filter_matches_in_memory_kcore(raw_mooccubex, monkeypatch):
    from dataclasses import replace

    entities = settings.RAW_DATA_FOLDER / "mooccubex" / "entities"
    monkeypatch.setattr(settings, "MOOCCUBEX_MAX_USER_ENROLLMENTS", 100)
    _write_jsonl(entities / "user.json", [
        {
            "id": f"U_{u}", "course_order": [1, 5, 6] if u < 4 else [1],
            "enroll_time": ["2020-01-01", "2020-01-02", "2020-01-03"] if u < 4 else ["2020-01-01"],
        }
        for u in range(8)
    ])
    config = DataConfig(dataset=DatasetName.MOOCCUBEX, min_interactions=3)
    full = mooccubex.load_mooccubex()
    expected = apply_common_filters(full, config)
    streamed = mooccubex.prepare_mooccubex(config)
    pd.testing.assert_frame_equal(streamed.interactions, expected.interactions)
    # The original enrollment cap applies before sparse catalog filtering.
    monkeypatch.setattr(settings, "MOOCCUBEX_MAX_USER_ENROLLMENTS", 2)
    with pytest.raises(ValueError, match="removed every interaction"):
        mooccubex.prepare_mooccubex(replace(config, max_interactions=10))
