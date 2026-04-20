import numpy as np
import pandas as pd
import pytest
import torch

from edurec import settings
from edurec.datasets import DatasetName, ElearningDataModule, RankerDataset
from edurec.datasets.elearnig_dataset import History


@pytest.fixture
def dm() -> ElearningDataModule:
    datamodule = ElearningDataModule(
        DatasetName.MARS, batch_size=1, test_ratio=0.2, val_ratio=0.2
    )
    datamodule.setup()
    return datamodule


def test_elearning_dataset_contract(dm: ElearningDataModule):
    ds = dm.train_ds

    assert isinstance(ds, RankerDataset)
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
    assert elem["history_items"].shape == torch.Size([settings.MAX_HISTORY_LEN])
    assert elem["history_ctx"].shape == torch.Size(
        [settings.MAX_HISTORY_LEN, dm.train_ds.num_ctx_feats]
    )
    assert elem["history_valid_mask"].shape == torch.Size([settings.MAX_HISTORY_LEN])
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
            int(neg_item) not in dm.seen_items_by_split["train"][user_id]
            for neg_item in negatives
        )


def test_negative_sampling_is_split_aware(dm: ElearningDataModule):
    val_ds = dm.val_ds
    dm.setup("test")
    test_ds = dm.test_ds

    for idx in range(min(25, len(val_ds))):
        elem = val_ds[idx]
        user_id = int(elem["user_id"])
        candidate_ids = elem["candidate_ids"] - 1
        positive_position = int(elem["positive_position"])

        negatives = torch.cat(
            [
                candidate_ids[:positive_position],
                candidate_ids[positive_position + 1 :],
            ]
        )

        assert all(
            int(neg_item) not in dm.seen_items_by_split["val"].get(user_id, set())
            for neg_item in negatives
        )

    for idx in range(min(25, len(test_ds))):
        elem = test_ds[idx]
        user_id = int(elem["user_id"])
        candidate_ids = elem["candidate_ids"] - 1
        positive_position = int(elem["positive_position"])

        negatives = torch.cat(
            [
                candidate_ids[:positive_position],
                candidate_ids[positive_position + 1 :],
            ]
        )

        assert all(
            int(neg_item) not in dm.seen_items_by_split["test"].get(user_id, set())
            for neg_item in negatives
        )


def test_negative_sampling_allows_future_items_but_never_current_positive(
    monkeypatch: pytest.MonkeyPatch,
):
    interactions = pd.DataFrame(
        {
            settings.USER_COL: [0],
            settings.ITEM_COL: [1],
            settings.RELEVANT_COL: [1.0],
        }
    )

    dataset = RankerDataset(
        interactions=interactions,
        precomputed_history=History(
            items=torch.zeros((1, settings.MAX_HISTORY_LEN), dtype=torch.long),
            ctx=torch.zeros((1, settings.MAX_HISTORY_LEN, 0), dtype=torch.float32),
            valid_mask=torch.zeros((1, settings.MAX_HISTORY_LEN), dtype=torch.bool),
        ),
        seen_items_by_user={0: {0}},
        num_ctx_feats=0,
        all_item_ids=np.array([0, 1, 2, 3], dtype=np.int64),
        n_negatives=1,
    )

    def fake_choice(a, size, replace=False):
        assert size == 1
        assert replace is False
        assert 0 not in a
        assert 1 not in a
        assert 2 in a
        return np.array([2], dtype=np.int64)

    monkeypatch.setattr(np.random, "choice", fake_choice)

    elem = dataset[0]
    candidate_ids = elem["candidate_ids"] - 1
    positive_position = int(elem["positive_position"])
    negatives = torch.cat(
        [
            candidate_ids[:positive_position],
            candidate_ids[positive_position + 1 :],
        ]
    )

    assert 2 in negatives.tolist()
    assert 1 not in negatives.tolist()
