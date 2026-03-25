import pytest
import torch

from edurec import config
from edurec.datasets import DatasetName, ElearningDataModule, ElearningDataset


@pytest.fixture
def dm() -> ElearningDataModule:
    datamodule = ElearningDataModule(
        DatasetName.MARS, batch_size=1, test_ratio=0.2, val_ratio=0.2
    )
    datamodule.setup()
    return datamodule


def test_elearning_dataset_contract(dm: ElearningDataModule):
    ds = dm.train_ds

    assert isinstance(ds, ElearningDataset)
    assert len(ds) > 0

    elem = ds[0]
    expected_keys = {
        "query_id",
        "user_id",
        "history_items",
        "history_ctx",
        "history_valid_mask",
        "candidate_ids",
        "candidate_labels",
        "positive_position",
    }

    assert set(elem.keys()) == expected_keys
    assert all(isinstance(elem[k], torch.Tensor) for k in expected_keys)
    assert elem["history_items"].shape == torch.Size([config.MAX_HISTORY_LEN])
    assert elem["history_valid_mask"].shape == torch.Size([config.MAX_HISTORY_LEN])
    assert elem["candidate_ids"].min().item() >= 1
    assert elem["candidate_labels"].sum().item() == 1.0
    assert elem["positive_position"].item() == int(elem["candidate_labels"].argmax())


def test_negative_sampling_excludes_known_positives(dm: ElearningDataModule):
    ds = dm.train_ds

    for idx in range(min(25, len(ds))):
        elem = ds[idx]
        user_id = int(elem["user_id"])
        candidate_ids = elem["candidate_ids"] - 1
        positive_position = int(elem["positive_position"])

        negatives = torch.cat(
            [
                candidate_ids[:positive_position],
                candidate_ids[positive_position + 1 :],
            ]
        )

        assert elem["candidate_labels"].sum().item() == 1.0
        assert all(
            int(neg_item) not in dm.user_positive_items[user_id] for neg_item in negatives
        )
