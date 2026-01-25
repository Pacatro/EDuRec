import numpy as np
import pandas as pd
import pytest
import torch

from edurec import config
from edurec.datasets import ElearningDataset, Preprocessor


def test_elearning_dataset_getitem():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        }
    )

    ds = ElearningDataset(df)

    elem = ds[0]
    assert isinstance(elem, dict)
    assert set(elem.keys()) == {"a", "b", "c"}
    assert isinstance(elem["a"], torch.Tensor)
    assert isinstance(elem["b"], torch.Tensor)
    assert isinstance(elem["c"], torch.Tensor)
    assert torch.equal(elem["a"], torch.tensor(1))
    assert torch.equal(elem["b"], torch.tensor(4))
    assert torch.equal(elem["c"], torch.tensor(7))

    elem = ds[2]
    assert torch.equal(elem["a"], torch.tensor(3))
    assert torch.equal(elem["b"], torch.tensor(6))
    assert torch.equal(elem["c"], torch.tensor(9))


def test_data_preprocessing():
    train_df = pd.DataFrame(
        {
            config.USER_COL: [123, 2324, 4343, 123, 2324],
            config.ITEM_COL: [145, 232, 343, 343, 145],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": ["a", "b", "c", "a", "b"],
        }
    )
    val_df = pd.DataFrame(
        {
            config.USER_COL: [123, 4343],
            config.ITEM_COL: [145, 232],
            "a": [6.0, 7.0],
            "b": ["c", "a"],
        }
    )
    test_df = pd.DataFrame(
        {
            config.USER_COL: [2324, 2324],
            config.ITEM_COL: [343, 145],
            "a": [8.0, 9.0],
            "b": ["b", "c"],
        }
    )

    preprocessor = Preprocessor(["a"], ["b"], [config.USER_COL, config.ITEM_COL])

    train_processed, val_processed, test_processed = preprocessor.fit_transform(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    assert "a" in preprocessor.numeric_cols
    assert "b" in preprocessor.categorical_cols

    assert preprocessor.numeric_cols == ["a"]
    assert preprocessor.categorical_cols == ["b"]
    assert preprocessor.id_cols == [config.USER_COL, config.ITEM_COL]

    assert isinstance(train_processed, pd.DataFrame)
    assert isinstance(val_processed, pd.DataFrame)
    assert isinstance(test_processed, pd.DataFrame)

    assert set(train_processed.columns) == set(train_df.columns)
    assert set(val_processed.columns) == set(val_df.columns)
    assert set(test_processed.columns) == set(test_df.columns)

    assert train_processed["a"].dtype == np.float32
    assert train_processed["b"].dtype == np.int64
    assert train_processed[config.USER_COL].dtype == np.int64
    assert train_processed[config.ITEM_COL].dtype == np.int64

    assert train_processed["a"].min() >= 0.0
    assert train_processed["a"].max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
