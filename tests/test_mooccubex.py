import json
from pathlib import Path

import pandas as pd
import pytest

from edurec import settings
from edurec.datasets import DataConfig, DatasetName
from edurec.datasets.preprocessing.build import build_processed
from edurec.datasets.preprocessing.prepare import limit_interactions
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
    config = DataConfig(
        dataset=DatasetName.MOOCCUBEX,
        max_interactions=25,
        random_state=42,
    )

    result = build_processed(raw, config)
    again = build_processed(raw, config)

    assert result.data.num_users == 5
    assert result.data.num_items == 5
    assert result.data.num_interactions == 25
    assert result.data.manifest is not None
    assert result.data.manifest.config["min_interactions"] == 5
    assert result.data.manifest.config["max_interactions"] == 25
    assert result.data.reports["interaction_budget"] == {
        "requested": 25,
        "interactions": 25,
        "users": 5,
        "items": 5,
        "selection": "whole_users_sha256",
    }
    assert result.raw.interactions.groupby(settings.USER_COL).size().eq(5).all()
    pd.testing.assert_frame_equal(result.raw.interactions, again.raw.interactions)


def test_limit_interactions_keeps_whole_user_histories():
    frame = pd.DataFrame(
        {
            settings.USER_COL: ["a"] * 3 + ["b"] * 2 + ["c"] * 4,
            settings.ITEM_COL: list(range(9)),
        }
    )

    limited = limit_interactions(frame, 5)

    assert len(limited) <= 5
    for user, count in limited[settings.USER_COL].value_counts().items():
        assert count == int((frame[settings.USER_COL] == user).sum())


def test_limit_interactions_rejects_an_impossible_budget():
    frame = pd.DataFrame(
        {
            settings.USER_COL: ["a"] * 3,
            settings.ITEM_COL: [0, 1, 2],
        }
    )

    with pytest.raises(ValueError, match="complete user history"):
        limit_interactions(frame, 2)


def test_source_filter_settings_invalidate_the_manifest(monkeypatch):
    config = DataConfig(dataset=DatasetName.MOOCCUBEX)
    manifest = config.manifest("temporal")
    monkeypatch.setattr(settings, "MOOCCUBEX_MAX_USER_ENROLLMENTS", 50)

    assert manifest.incompatibility(config.manifest("temporal")) is not None


def test_max_interactions_validation():
    with pytest.raises(ValueError, match="positive integer"):
        DataConfig(dataset=DatasetName.COCO, max_interactions=0)

    with pytest.raises(ValueError, match="at least min_interactions"):
        DataConfig(dataset=DatasetName.COCO, max_interactions=2)
