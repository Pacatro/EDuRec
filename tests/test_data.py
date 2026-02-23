import numpy as np
import pandas as pd
import pytest
import torch

from edurec import config
from edurec.datasets import DataProcessor, DatasetName, load_data
from edurec.datasets.data_processor import get_column_types
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


def test_data_processor_mars():
    raw_dataset = load_data(DatasetName.MARS)
    merged_df = DataProcessor.merge_raw_features(
        interactions_df=raw_dataset.interactions,
        users_df=raw_dataset.u_feats,
        items_df=raw_dataset.i_feats,
    )

    assert "user_job" in merged_df.columns
    assert "item_name" in merged_df.columns

    train_df = merged_df.iloc[:300].reset_index(drop=True)
    val_df = merged_df.iloc[300:360].reset_index(drop=True)
    test_df = merged_df.iloc[360:420].reset_index(drop=True)

    numeric_cols, categorical_lengths, list_cols, text_cols = get_column_types(train_df)

    preprocessor = DataProcessor(
        numeric_cols=numeric_cols,
        categorical_cols=list(categorical_lengths.keys()),
        text_cols=text_cols,
        list_cols=list_cols,
        id_cols=[config.USER_COL, config.ITEM_COL],
        has_time=config.TIME_COL in train_df.columns,
    )

    train_processed, val_processed, test_processed = preprocessor.fit_transform(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    assert train_processed is not None
    assert val_processed is not None
    assert test_processed is not None
    assert train_processed[config.RATING_COL].dtype == np.float32
    assert train_processed[config.RELEVANT_COL].dtype == bool
    assert config.USER_COL in train_processed.columns
    assert config.ITEM_COL in train_processed.columns

    full_processed = pd.concat(
        [train_processed, val_processed, test_processed], ignore_index=True
    )
    user_df, item_df = preprocessor.split_entity_feature_frames(full_processed)
    user_tensors, item_tensors = preprocessor.build_entity_tensors(full_processed)
    print(item_df[item_df[config.ITEM_COL] == 0])
    print(item_tensors[0])

    assert user_df[config.USER_COL].is_unique
    assert item_df[config.ITEM_COL].is_unique
    assert user_tensors.dtype == torch.float32
    assert item_tensors.dtype == torch.float32
    assert user_tensors.ndim == 2
    assert item_tensors.ndim == 2
    assert user_tensors.shape[0] == len(user_df)
    assert item_tensors.shape[0] == len(item_df)
    assert user_tensors.shape[1] == (len(user_df.columns) - 1)
    assert item_tensors.shape[1] == (len(item_df.columns) - 1)


def test_data_processor_itm():
    raw_dataset = load_data(DatasetName.ITM)
    merged_df = DataProcessor.merge_raw_features(
        interactions_df=raw_dataset.interactions,
        users_df=raw_dataset.u_feats,
        items_df=raw_dataset.i_feats,
    )

    assert "user_gender" in merged_df.columns
    assert "item_title" in merged_df.columns

    train_df = merged_df.iloc[:300].reset_index(drop=True)
    val_df = merged_df.iloc[300:360].reset_index(drop=True)
    test_df = merged_df.iloc[360:420].reset_index(drop=True)

    numeric_cols, categorical_lengths, list_cols, text_cols = get_column_types(train_df)

    preprocessor = DataProcessor(
        numeric_cols=numeric_cols,
        categorical_cols=list(categorical_lengths.keys()),
        text_cols=text_cols,
        list_cols=list_cols,
        id_cols=[config.USER_COL, config.ITEM_COL],
        has_time=config.TIME_COL in train_df.columns,
    )

    train_processed, val_processed, test_processed = preprocessor.fit_transform(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    assert train_processed is not None
    assert val_processed is not None
    assert test_processed is not None
    assert train_processed[config.RATING_COL].dtype == np.float32
    assert train_processed[config.RELEVANT_COL].dtype == bool
    assert config.USER_COL in train_processed.columns
    assert config.ITEM_COL in train_processed.columns

    full_processed = pd.concat(
        [train_processed, val_processed, test_processed], ignore_index=True
    )
    user_df, item_df = preprocessor.split_entity_feature_frames(full_processed)
    user_tensors, item_tensors = preprocessor.build_entity_tensors(full_processed)

    assert user_df[config.USER_COL].is_unique
    assert item_df[config.ITEM_COL].is_unique
    assert user_tensors.dtype == torch.float32
    assert item_tensors.dtype == torch.float32
    assert user_tensors.ndim == 2
    assert item_tensors.ndim == 2
    assert user_tensors.shape[0] == len(user_df)
    assert item_tensors.shape[0] == len(item_df)
    assert user_tensors.shape[1] == (len(user_df.columns) - 1)
    assert item_tensors.shape[1] == (len(item_df.columns) - 1)


if __name__ == "__main__":
    test_data_processor_mars()
    pytest.main([__file__, "-v"])
