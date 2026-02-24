import pytest

from edurec.datasets import DatasetName
from edurec.recsys.datamodule import ElearningDataModule


def test_split():
    dm = ElearningDataModule(
        DatasetName.MARS, batch_size=1, test_ratio=0.2, val_ratio=0.2
    )
    inter_len = len(dm.interactions)
    train_df, val_df, test_df = dm._split_data()

    assert len(train_df) + len(val_df) + len(test_df) == inter_len
    assert len(train_df) == inter_len - len(test_df) - len(val_df)
    assert len(val_df) == len(test_df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
