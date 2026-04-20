import numpy as np
import pandas as pd
import pytest
import joblib

from edurec import config
from edurec.datasets import DataProcessor
from edurec.datasets.data_processor import SentenceEmbeddingTransformer


def _has_suffix(columns: list[str], suffix: str) -> bool:
    return any(col.endswith(suffix) for col in columns)


def _get_matching_column(columns: list[str], suffix: str) -> str:
    return next(col for col in columns if col.endswith(suffix))


def test_data_processor_processes_all_feature_types(fake_sentence_model):
    users_train = pd.DataFrame(
        {
            config.USER_COL: [1, 2, 3, 4],
            "age": [20, 30, 25, 40],
            "job": ["teacher", "student", "teacher", None],
            "bio": [
                "teaches statistics and machine learning",
                "studies recommender systems",
                "works on data mining",
                "focuses on analytics",
            ],
            "skills": [
                "python,teaching",
                "sql|research",
                "ml;evaluation",
                "analysis,mentoring",
            ],
        }
    )
    users_val = pd.DataFrame(
        {
            config.USER_COL: [5, 6],
            "age": [35, 22],
            "job": ["admin", "student"],
            "bio": ["manages course operations", "likes practical labs"],
            "skills": ["coordination,reporting", "python|debugging"],
        }
    )
    users_test = pd.DataFrame(
        {
            config.USER_COL: [7],
            "age": [28],
            "job": ["guest"],
            "bio": ["reviews learning content"],
            "skills": ["feedback"],
        }
    )

    items_train = pd.DataFrame(
        {
            config.ITEM_COL: [101, 102, 103, 104],
            "language": ["en", "es", "en", "es"],
            "nb_views": [100, 200, 150, 180],
            "description": [
                "this is a practical machine learning course",
                "hands on introduction to data analysis",
                "comprehensive tutorial for recommendation systems",
                "learn python basics for analytics projects",
            ],
            "tags": [
                "ml,python",
                "analysis|statistics",
                "recsys;evaluation",
                "python,data",
            ],
        }
    )
    items_val = pd.DataFrame(
        {
            config.ITEM_COL: [105],
            "language": ["it"],
            "nb_views": [170],
            "description": ["introductory course about modern statistics"],
            "tags": ["etl,sql"],
        }
    )
    items_test = pd.DataFrame(
        {
            config.ITEM_COL: [106],
            "language": ["de"],
            "nb_views": [120],
            "description": ["advanced class on recommendation pipelines"],
            "tags": ["debugging"],
        }
    )

    interactions_train = pd.DataFrame(
        {
            config.USER_COL: [1, 2, 3, 4, 1],
            config.ITEM_COL: [101, 102, 103, 104, 102],
            "watch_percentage": [0.6, 0.8, 0.5, 0.9, 0.3],
            "semester": ["spring", "fall", "spring", "fall", "spring"],
            "feedback": [
                "helpful explanation and examples",
                "clear structure for the topic",
                "too short but still useful",
                "great content and pacing",
                "needs more practical exercises",
            ],
            "skills": [
                "python, ml",
                "statistics|visualization",
                "recsys;evaluation",
                "python,data",
                "experiments",
            ],
            config.TIME_COL: [
                "2025-01-01T08:00:00Z",
                "2025-01-02T10:00:00Z",
                "2025-01-03T12:00:00Z",
                "2025-01-04T14:00:00Z",
                "2025-01-05T16:00:00Z",
            ],
            config.RATING_COL: [4, 5, 3, 5, 2],
            config.RELEVANT_COL: [1, 1, 0, 1, 0],
        }
    )
    interactions_val = pd.DataFrame(
        {
            config.USER_COL: [5],
            config.ITEM_COL: [105],
            "watch_percentage": [0.7],
            "semester": ["winter"],
            "feedback": ["solid summary of the lesson"],
            "skills": ["sql,etl"],
            config.TIME_COL: ["2025-01-06T18:00:00Z"],
            config.RATING_COL: [4],
            config.RELEVANT_COL: [1],
        }
    )
    interactions_test = pd.DataFrame(
        {
            config.USER_COL: [7],
            config.ITEM_COL: [106],
            "watch_percentage": [0.4],
            "semester": ["summer"],
            "feedback": ["good but a bit repetitive"],
            "skills": ["debugging"],
            config.TIME_COL: ["2025-01-07T20:00:00Z"],
            config.RATING_COL: [3],
            config.RELEVANT_COL: [0],
        }
    )

    schema = {
        "users": {
            "bin": [],
            "num": ["age"],
            "cat": ["job"],
            "text": ["bio"],
            "list": ["skills"],
        },
        "items": {
            "bin": [],
            "num": ["nb_views"],
            "cat": ["language"],
            "text": ["description"],
            "list": ["tags"],
        },
        "inter": {
            "bin": [],
            "num": ["watch_percentage", config.RATING_COL],
            "cat": ["semester"],
            "text": ["feedback"],
            "list": ["skills"],
        },
    }

    processor = DataProcessor(schema=schema)
    processor.fit(users_train, items_train, interactions_train)

    train_processed = processor.transform(users_train, items_train, interactions_train)
    val_processed = processor.transform(users_val, items_val, interactions_val)
    test_processed = processor.transform(users_test, items_test, interactions_test)

    assert train_processed.users is not None
    assert val_processed.users is not None
    assert test_processed.users is not None

    assert train_processed.interactions is not None
    assert val_processed.interactions is not None
    assert test_processed.interactions is not None

    assert train_processed.items is not None
    assert val_processed.items is not None
    assert test_processed.items is not None

    assert len(train_processed.users) == len(users_train)
    assert len(val_processed.users) == len(users_val)
    assert len(test_processed.users) == len(users_test)

    assert train_processed.items is not None

    assert list(train_processed.users.columns) == list(val_processed.users.columns)
    assert list(train_processed.items.columns) == list(test_processed.items.columns)

    assert not train_processed.users.isna().any().any()
    assert np.isfinite(train_processed.users.values).all()
    assert np.isfinite(train_processed.items.values).all()
    numeric_interactions = train_processed.interactions.drop(columns=[config.TIME_COL])
    assert np.isfinite(numeric_interactions.values).all()

    pd.testing.assert_index_equal(train_processed.users.index, users_train.index)
    pd.testing.assert_index_equal(
        train_processed.interactions.index, interactions_train.index
    )

    assert -1 in val_processed.users.values

    assert val_processed.interactions is not None
    assert test_processed.interactions is not None
    assert -1 in val_processed.interactions.values
    assert -1 in test_processed.interactions.values

    assert train_processed.items.shape[1] > 2
    assert processor.feature_metadata["users"].numeric_cols == ["age"]
    assert _has_suffix(processor.feature_metadata["users"].categorical_cols, "job")
    assert processor.feature_metadata["users"].text_cols == ["bio"]
    assert processor.feature_metadata["users"].list_cols == ["skills"]
    assert _has_suffix(processor.feature_metadata["users"].dense_cols, "age")
    assert len(processor.feature_metadata["users"].dense_cols) > 1
    assert processor.feature_metadata["items"].numeric_cols == ["nb_views"]
    assert _has_suffix(processor.feature_metadata["items"].categorical_cols, "language")
    assert processor.feature_metadata["items"].text_cols == ["description"]
    assert processor.feature_metadata["items"].list_cols == ["tags"]
    assert _has_suffix(processor.feature_metadata["items"].dense_cols, "nb_views")
    assert len(processor.feature_metadata["items"].dense_cols) > 1
    assert processor.feature_metadata["inter"].text_cols == ["feedback"]
    assert processor.feature_metadata["inter"].list_cols == ["skills"]
    assert processor.feature_metadata["inter"].time_cols == [config.TIME_COL]
    assert _has_suffix(
        processor.feature_metadata["inter"].dense_cols, "watch_percentage"
    )
    assert _has_suffix(processor.feature_metadata["inter"].dense_cols, "time_hour")
    user_job_col = _get_matching_column(
        processor.feature_metadata["users"].categorical_cols,
        "job",
    )
    assert (
        processor.feature_metadata["users"].categorical_cardinalities[user_job_col] >= 1
    )

    users_groups = processor.column_groups["users"]
    items_groups = processor.column_groups["items"]
    inter_groups = processor.column_groups["inter"]

    assert users_groups["input"] == ["age", "job", "bio", "skills"]
    assert items_groups["input"] == ["nb_views", "language", "description", "tags"]
    assert items_groups["text"] == ["description"]
    assert items_groups["list"] == ["tags"]

    assert inter_groups["time"] == [config.TIME_COL]
    assert inter_groups["text"] == ["feedback"]
    assert inter_groups["list"] == ["skills"]
    assert inter_groups["input"] == [
        "watch_percentage",
        "semester",
        "feedback",
        "skills",
        config.TIME_COL,
    ]
    assert config.RATING_COL not in inter_groups["input"]
    assert config.RELEVANT_COL not in inter_groups["input"]
    assert config.USER_COL not in inter_groups["input"]
    assert config.ITEM_COL not in inter_groups["input"]
    assert config.TIME_COL in train_processed.interactions.columns
    assert _has_suffix(train_processed.interactions.columns.tolist(), "time_hour")
    assert train_processed.interactions.shape[1] > len(inter_groups["passthrough"])
    assert len(fake_sentence_model) == 1
    assert config.TEXT_EMBEDDING_MODEL in fake_sentence_model


def test_data_processor_respects_global_feature_toggle(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        config,
        "PREPROCESS_FEATURE_TYPES",
        ("numeric", "categorical"),
    )

    users = pd.DataFrame(
        {
            config.USER_COL: [1, 2],
            "age": [20, 30],
            "job": ["teacher", "student"],
            "bio": ["teaches statistics", "studies recommender systems"],
            "skills": ["python,teaching", "sql|research"],
        }
    )
    items = pd.DataFrame(
        {
            config.ITEM_COL: [101, 102],
            "language": ["en", "es"],
            "nb_views": [100, 200],
            "description": ["machine learning course", "data analysis lab"],
            "tags": ["ml,python", "analysis|statistics"],
        }
    )
    interactions = pd.DataFrame(
        {
            config.USER_COL: [1, 2],
            config.ITEM_COL: [101, 102],
            "watch_percentage": [0.6, 0.8],
            "semester": ["spring", "fall"],
            "feedback": ["helpful explanation", "clear structure"],
            "skills": ["python,ml", "statistics|visualization"],
            config.TIME_COL: ["2025-01-01T08:00:00Z", "2025-01-02T10:00:00Z"],
            config.RATING_COL: [4, 5],
            config.RELEVANT_COL: [1, 1],
        }
    )
    schema = {
        "users": {
            "bin": [],
            "num": ["age"],
            "cat": ["job"],
            "text": ["bio"],
            "list": ["skills"],
        },
        "items": {
            "bin": [],
            "num": ["nb_views"],
            "cat": ["language"],
            "text": ["description"],
            "list": ["tags"],
        },
        "inter": {
            "bin": [],
            "num": ["watch_percentage"],
            "cat": ["semester"],
            "text": ["feedback"],
            "list": ["skills"],
        },
    }

    processor = DataProcessor(schema=schema)
    processor.fit(users, items, interactions)
    processed = processor.transform(users=users, items=items, interactions=interactions)

    assert processed.users is not None
    assert processed.items is not None
    assert processed.interactions is not None

    assert processor.column_groups["users"]["input"] == ["age", "job"]
    assert processor.column_groups["items"]["input"] == ["nb_views", "language"]
    assert processor.column_groups["inter"]["input"] == ["watch_percentage", "semester"]

    assert processor.feature_metadata["users"].dense_cols == ["num__age"]
    assert processor.feature_metadata["items"].dense_cols == ["num__nb_views"]
    assert processor.feature_metadata["inter"].dense_cols == ["num__watch_percentage"]

    assert "bio" not in processed.users.columns
    assert "description" not in processed.items.columns
    assert "time_hour" not in processed.interactions.columns
    assert config.TIME_COL in processed.interactions.columns


def test_data_processor_normalizes_mixed_type_categorical_columns():
    schema = {
        "users": {
            "bin": [],
            "num": [],
            "cat": ["segment"],
            "text": [],
            "list": [],
        },
        "items": {
            "bin": [],
            "num": [],
            "cat": ["grade", "prerequisite"],
            "text": [],
            "list": [],
        },
        "inter": {
            "bin": [],
            "num": [],
            "cat": ["semester"],
            "text": [],
            "list": [],
        },
    }
    items = pd.DataFrame(
        {
            config.ITEM_COL: [101, 102, 103, 104],
            "grade": [2.0, 3.0, np.nan, ""],
            "prerequisite": [None, "algebra", "", "history"],
        }
    )
    users = pd.DataFrame({config.USER_COL: [1], "segment": ["A"]})
    interactions = pd.DataFrame(
        {config.USER_COL: [1], config.ITEM_COL: [101], "semester": ["1.0"]}
    )

    processor = DataProcessor(schema=schema)
    processor.fit(
        users_train=users,
        items_train=items,
        interactions_train=interactions,
    )

    processed = processor.transform(
        users=users,
        items=items,
        interactions=interactions,
    )

    assert processed.items is not None
    assert np.isfinite(processed.items.drop(columns=[config.ITEM_COL]).to_numpy()).all()
    assert _has_suffix(
        processor.feature_metadata["items"].categorical_cols,
        "grade",
    )


def test_sentence_embedding_transformer_cleans_truncates_and_serializes(
    tmp_path,
    fake_sentence_model,
):
    transformer = SentenceEmbeddingTransformer(
        cols=["title", "description"],
        model_name=config.TEXT_EMBEDDING_MODEL,
        embedding_dim=config.TEXT_EMBEDDING_DIM,
        batch_size=config.TEXT_EMBEDDING_BATCH_SIZE,
        max_tokens=8,
    )
    df = pd.DataFrame(
        {
            "title": ["<b>Hello</b>\nWorld"],
            "description": [
                "Visit https://example.com\tfor MORE details and extra tokens here"
            ],
        }
    )

    transformer.fit(df)
    transformed = transformer.transform(df)
    transformed_array = np.asarray(transformed)

    assert transformed.shape == (1, config.TEXT_EMBEDDING_DIM)
    assert np.isfinite(transformed_array).all()
    assert transformer.get_feature_names_out().shape == (config.TEXT_EMBEDDING_DIM,)

    model = fake_sentence_model[config.TEXT_EMBEDDING_MODEL]
    assert model.last_texts == ["title: hello world [SEP] description: visit for more"]

    path = tmp_path / "text_transformer.joblib"
    joblib.dump(transformer, path)
    loaded = joblib.load(path)
    reloaded = loaded.transform(df)

    np.testing.assert_allclose(transformed_array, np.asarray(reloaded))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
