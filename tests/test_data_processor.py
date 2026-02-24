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

    processor = DataProcessor()
    processor.fit(users_train, items_train, interactions_train)

    train_processed = processor.transform(users_train, items_train, interactions_train)
    val_processed = processor.transform(users_val, items_val, interactions_val)
    test_processed = processor.transform(users_test, items_test, interactions_test)
    print(train_processed.X_items[0])

    assert train_processed.X_users.shape[0] == len(users_train)
    assert val_processed.X_users.shape[0] == len(users_val)
    assert test_processed.X_users.shape[0] == len(users_test)

    assert train_processed.X_items.shape[0] == len(items_train)
    assert val_processed.X_items.shape[0] == len(items_val)
    assert test_processed.X_items.shape[0] == len(items_test)

    assert train_processed.X_interactions is not None
    assert val_processed.X_interactions is not None
    assert test_processed.X_interactions is not None

    assert train_processed.X_users.shape[1] == val_processed.X_users.shape[1]
    assert train_processed.X_users.shape[1] == test_processed.X_users.shape[1]
    assert train_processed.X_items.shape[1] == val_processed.X_items.shape[1]
    assert train_processed.X_items.shape[1] == test_processed.X_items.shape[1]
    assert (
        train_processed.X_interactions.shape[1]
        == val_processed.X_interactions.shape[1]
        == test_processed.X_interactions.shape[1]
    )

    assert np.isfinite(train_processed.X_users).all()
    assert np.isfinite(train_processed.X_items).all()
    assert np.isfinite(train_processed.X_interactions).all()

    # Unknown categories in val/test should be handled by OrdinalEncoder with -1.
    assert -1 in val_processed.X_users
    assert -1 in val_processed.X_interactions


if __name__ == "__main__":
    test_data_processor()
    pytest.main([__file__, "-v"])
