import pytest

from edurec import config
from edurec.datasets import DatasetName, load_raw_data
from edurec.datasets.loaders import RawDataset


def test_data_loaders_mars():
    n_users = 131247
    n_items = 2618
    n_interactions = 88998

    raw_dataset = load_raw_data(DatasetName.MARS)

    assert isinstance(raw_dataset, RawDataset)

    # Check dataset cardinality
    assert len(raw_dataset.u_feats) == n_users
    assert len(raw_dataset.i_feats) == n_items
    assert len(raw_dataset.interactions) == n_interactions
    assert config.RELEVANT_COL not in raw_dataset.interactions.columns

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
            "Difficulty",
            "Job",
            "Software",
            "Theme",
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
        ]
    )


def test_data_loaders_itm():
    n_users = 476
    n_items = 70
    n_interactions = 5230

    raw_dataset = load_raw_data(DatasetName.ITM)

    assert isinstance(raw_dataset, RawDataset)

    # Check dataset cardinality
    assert len(raw_dataset.u_feats) == n_users
    assert len(raw_dataset.i_feats) == n_items
    assert len(raw_dataset.interactions) == n_interactions
    assert config.RELEVANT_COL not in raw_dataset.interactions.columns

    # Check dataset features
    assert set(raw_dataset.u_feats.columns) == set(
        [config.USER_COL, " Gender", " Age", "Married"]
    )
    assert set(raw_dataset.i_feats.columns) == set(
        [config.ITEM_COL, "Title", "Descriptions"]
    )
    assert set(raw_dataset.interactions.columns) == set(
        [
            config.USER_COL,
            config.ITEM_COL,
            config.RATING_COL,
            "Class",
            "Lockdown",
            "Data",
            "Ease",
            "App",
            "Semester",
        ]
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
