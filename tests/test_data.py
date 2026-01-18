import pandas as pd
import torch
from edurec.core.data import ElearningDataset
import pytest


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
