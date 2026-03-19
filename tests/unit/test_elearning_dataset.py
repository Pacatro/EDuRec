import pytest
import torch

from edurec.datasets import DatasetName, ElearningDataModule, ElearningDataset
from edurec import config


@pytest.fixture
def dm() -> ElearningDataModule:
    dm = ElearningDataModule(
        DatasetName.MARS, batch_size=1, test_ratio=0.2, val_ratio=0.2
    )

    dm.setup()

    return dm


def test_elearning_dataset(dm: ElearningDataModule):
    ds = dm.train_ds

    assert isinstance(ds, ElearningDataset)

    elem = ds[0]

    assert isinstance(elem, dict)

    expected_keys = [
        "user_id",
        "history_items",
        "history_ctx",
        "candidates",
        "mask",
        "target",
    ]

    assert set(elem.keys()) == set(expected_keys)
    assert all(isinstance(elem[k], torch.Tensor) for k in expected_keys)
    assert elem["history_items"].shape == torch.Size([config.MAX_HISTORY_LEN])
