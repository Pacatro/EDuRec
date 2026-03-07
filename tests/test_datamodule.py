from pathlib import Path

import pytest

from edurec.datasets import DatasetName, ElearningDataModule
from edurec import config


def test_split_mars():
    dm = ElearningDataModule(
        DatasetName.MARS, batch_size=1, test_ratio=0.2, val_ratio=0.2
    )
    inter_len = dm.num_interactions
    train_df, val_df, test_df = dm._split_data()

    assert len(train_df) + len(val_df) + len(test_df) == inter_len
    assert len(train_df) == inter_len - len(test_df) - len(val_df)
    assert len(val_df) == len(test_df)


def test_setup():
    dm = ElearningDataModule(
        DatasetName.MARS, batch_size=1, test_ratio=0.2, val_ratio=0.2
    )
    dm.setup()
    assert dm.is_processed


def test_create_inter_graph():
    dm = ElearningDataModule(
        DatasetName.MARS, batch_size=1, test_ratio=0.2, val_ratio=0.2
    )

    dm.setup()

    train_raw = dm._processed_data["train"]

    assert train_raw is not None

    num_pos_inter = len(train_raw[train_raw[config.RELEVANT_COL] > 0])
    graph = dm.create_inter_graph()
    num_interacciones = graph["user", "interacts", "item"].num_edges
    num_rev_interacciones = graph["item", "rev_interacts", "user"].num_edges

    assert num_interacciones == num_rev_interacciones
    assert num_interacciones == num_pos_inter


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
