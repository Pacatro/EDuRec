import pytest

from edurec import config
from edurec.datasets import DatasetName, load_data
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
        [config.ITEM_COL, "title", "descriptions"]
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
