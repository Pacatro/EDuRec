import numpy as np
import pandas as pd
import pytest

from edurec import config
from edurec.datasets import DataProcessor, load_data, DatasetName
from edurec.datasets.loaders import RawDataset


def test_data_loaders_mars():
    n_users = 131247
    n_items = 2618
    n_interactions = 88998

    raw_dataset = load_data(DatasetName.MARS)

    assert isinstance(raw_dataset, RawDataset)

    # Check dataset cardinality
    assert len(raw_dataset.u_feats) == n_users
    assert len(raw_dataset.i_feats) == n_items
    assert len(raw_dataset.interactions) == n_interactions

    # Check dataset features
    assert set(raw_dataset.u_feats.columns) == set(["user_id", "job"])
    assert set(raw_dataset.i_feats.columns) == set(
        [
            "item_id",
            "language",
            "name",
            "nb_views",
            "description",
            "created_at",
            "difficulty",
            "job",
            "software",
            "theme",
            "duration",
            "item_type",
        ]
    )
    assert set(raw_dataset.interactions.columns) == set(
        [
            config.USER_COL,
            config.ITEM_COL,
            "watch_percentage",
            config.TIME_COL,
            config.RATING_COL,
            config.RELEVANT_COL,
        ]
    )


def test_data_loaders_itm():
    n_users = 476
    n_items = 70
    n_interactions = 5230

    raw_dataset = load_data(DatasetName.ITM)

    assert isinstance(raw_dataset, RawDataset)

    # Check dataset cardinality
    assert len(raw_dataset.u_feats) == n_users
    assert len(raw_dataset.i_feats) == n_items
    assert len(raw_dataset.interactions) == n_interactions

    # Check dataset features
    assert set(raw_dataset.u_feats.columns) == set(
        [config.USER_COL, "gender", "age", "married"]
    )
    assert set(raw_dataset.i_feats.columns) == set(
        [config.ITEM_COL, "title", "url", "descriptions"]
    )
    assert set(raw_dataset.interactions.columns) == set(
        [
            config.USER_COL,
            config.ITEM_COL,
            config.RATING_COL,
            config.RELEVANT_COL,
            "class",
            "lockdown",
            "data",
            "ease",
            "app",
            "semester",
        ]
    )


# def test_data_preprocessing():
#     train_df = pd.DataFrame(
#         {
#             config.USER_COL: [123, 2324, 4343, 123, 2324],
#             config.ITEM_COL: [145, 232, 343, 343, 145],
#             "a": [1.0, 2.0, 3.0, 4.0, 5.0],
#             "b": ["a", "b", "c", "a", "b"],
#         }
#     )
#     val_df = pd.DataFrame(
#         {
#             config.USER_COL: [123, 4343],
#             config.ITEM_COL: [145, 232],
#             "a": [6.0, 7.0],
#             "b": ["c", "a"],
#         }
#     )
#     test_df = pd.DataFrame(
#         {
#             config.USER_COL: [2324, 2324],
#             config.ITEM_COL: [343, 145],
#             "a": [8.0, 9.0],
#             "b": ["b", "c"],
#         }
#     )
#
#     preprocessor = DataProcessor(
#         numeric_cols=["a"],
#         categorical_cols=["b"],
#         text_cols=[],
#         list_cols=[],
#         id_cols=[config.USER_COL, config.ITEM_COL],
#     )
#
#     train_processed, val_processed, test_processed = preprocessor.fit_transform(
#         train_df=train_df,
#         val_df=val_df,
#         test_df=test_df,
#     )
#
#     assert "a" in preprocessor.numeric_cols
#     assert "b" in preprocessor.categorical_cols
#
#     assert preprocessor.numeric_cols == ["a"]
#     assert preprocessor.categorical_cols == ["b"]
#     assert preprocessor.id_cols == [config.USER_COL, config.ITEM_COL]
#
#     assert isinstance(train_processed, pd.DataFrame)
#     assert isinstance(val_processed, pd.DataFrame)
#     assert isinstance(test_processed, pd.DataFrame)
#
#     assert set(train_processed.columns) == set(train_df.columns)
#     assert set(val_processed.columns) == set(val_df.columns)
#     assert set(test_processed.columns) == set(test_df.columns)
#
#     assert train_processed["a"].dtype == np.float32
#     assert train_processed["b"].dtype == np.int64
#     assert train_processed[config.USER_COL].dtype == np.int64
#     assert train_processed[config.ITEM_COL].dtype == np.int64
#
#     assert train_processed["a"].min() >= 0.0
#     assert train_processed["a"].max() <= 1.0


if __name__ == "__main__":
    test_data_loaders_itm()
    pytest.main([__file__, "-v"])
