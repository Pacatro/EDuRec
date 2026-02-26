import numpy as np
import pandas as pd
import pytest

from edurec import config
from edurec.datasets.data_processor import DataProcessor


def test_data_processor():
    users_train = pd.DataFrame(
        {
            config.USER_COL: [1, 2, 3, 4],
            "age": [20, 30, 25, 40],
            "job": ["teacher", "student", "teacher", None],
        }
    )
    users_val = pd.DataFrame(
        {
            config.USER_COL: [5, 6],
            "age": [35, 22],
            "job": ["admin", "student"],
        }
    )
    users_test = pd.DataFrame(
        {
            config.USER_COL: [7],
            "age": [28],
            "job": ["guest"],
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
        }
    )
    items_val = pd.DataFrame(
        {
            config.ITEM_COL: [105],
            "language": ["it"],
            "nb_views": [170],
            "description": ["introductory course about modern statistics"],
        }
    )
    items_test = pd.DataFrame(
        {
            config.ITEM_COL: [106],
            "language": ["de"],
            "nb_views": [120],
            "description": ["advanced class on recommendation pipelines"],
        }
    )

    interactions_train = pd.DataFrame(
        {
            config.USER_COL: [1, 2, 3, 4, 1],
            config.ITEM_COL: [101, 102, 103, 104, 102],
            "watch_percentage": [0.6, 0.8, 0.5, 0.9, 0.3],
            "semester": ["spring", "fall", "spring", "fall", "spring"],
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
            "text": [],
            "list": [],
        },
        "items": {
            "bin": [],
            "num": ["nb_views"],
            "cat": ["language"],
            "text": ["description"],
            "list": [],
        },
        "inter": {
            "bin": [],
            "num": ["watch_percentage", config.RATING_COL],
            "cat": ["semester"],
            "text": [],
            "list": [],
        },
    }

    processor = DataProcessor(schema=schema)
    processor.fit(users_train, items_train, interactions_train)

    train_processed = processor.transform(users_train, items_train, interactions_train)
    val_processed = processor.transform(users_val, items_val, interactions_val)
    test_processed = processor.transform(users_test, items_test, interactions_test)

    print(train_processed.interactions.head())

    assert isinstance(train_processed.users, pd.DataFrame)
    assert isinstance(train_processed.items, pd.DataFrame)
    assert isinstance(train_processed.interactions, pd.DataFrame)

    assert len(train_processed.users) == len(users_train)
    assert len(val_processed.users) == len(users_val)
    assert len(test_processed.users) == len(users_test)

    assert list(train_processed.users.columns) == list(val_processed.users.columns)
    assert list(train_processed.items.columns) == list(test_processed.items.columns)

    assert not train_processed.users.isna().any().any()
    assert np.isfinite(train_processed.users.values).all()
    assert np.isfinite(train_processed.items.values).all()
    assert np.isfinite(train_processed.interactions.values).all()

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


if __name__ == "__main__":
    test_data_processor()
    pytest.main([__file__, "-v"])
