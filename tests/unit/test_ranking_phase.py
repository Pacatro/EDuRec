from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from edurec import settings
from edurec.datasets import DatasetName, ElearningDataModule, Phase, RankerDataset
from edurec.datasets.datamodule import ProcessedArtifacts
from edurec.datasets.history import History


def test_ranker_dataset_recovers_positive_position_from_labels_when_missing():
    interactions = pd.DataFrame(
        {
            settings.USER_COL: [0],
            settings.ITEM_COL: [2],
            settings.RELEVANT_COL: [1.0],
            settings.CANDIDATE_IDS_COL: [[1, 2, 3]],
            settings.CANDIDATE_LABELS_COL: [[0.0, 1.0, 0.0]],
            settings.POSITIVE_POSITION_COL: [np.nan],
        }
    )

    dataset = RankerDataset(
        interactions=interactions,
        precomputed_history=History(
            items=torch.zeros((1, 2), dtype=torch.long),
            ctx=torch.zeros((1, 2, 0), dtype=torch.float32),
            valid_mask=torch.zeros((1, 2), dtype=torch.bool),
        ),
        num_ctx_feats=0,
    )

    example = dataset[0]

    assert example.positive_positions.item() == 1


def test_datamodule_ranking_phase_uses_only_positive_rows():
    dm = ElearningDataModule(
        DatasetName.ITM,
        batch_size=2,
        test_ratio=0.2,
        val_ratio=0.2,
    )
    dm.phase = Phase.RANKING

    train_df = pd.DataFrame(
        [
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 0,
                settings.RELEVANT_COL: 0.0,
                settings.CANDIDATE_IDS_COL: [0, 1],
                settings.CANDIDATE_LABELS_COL: [0.0, 0.0],
                settings.POSITIVE_POSITION_COL: np.nan,
                settings.INTERACTION_ORDER_COL: 0,
                "ctx_value": 0.1,
            },
            {
                settings.USER_COL: 0,
                settings.ITEM_COL: 1,
                settings.RELEVANT_COL: 1.0,
                settings.CANDIDATE_IDS_COL: [1, 2],
                settings.CANDIDATE_LABELS_COL: [1.0, 0.0],
                settings.POSITIVE_POSITION_COL: 0,
                settings.INTERACTION_ORDER_COL: 1,
                "ctx_value": 0.2,
            },
        ]
    )

    empty_df = train_df.iloc[0:0].copy()
    dm.artifacts = ProcessedArtifacts(
        train=train_df,
        val=empty_df,
        test=empty_df,
        u_static_feats=torch.zeros((1, 1), dtype=torch.float32),
        i_static_feats=torch.zeros((3, 1), dtype=torch.float32),
        data_processor=cast(Any, object()),
    )
    dm.next_item_hist_by_split = {
        "train": History(
            items=torch.tensor([[1, 0], [1, 2]], dtype=torch.long),
            ctx=torch.zeros((2, 2, 1), dtype=torch.float32),
            valid_mask=torch.tensor([[True, False], [True, True]]),
        ),
        "val": History(
            items=torch.zeros((0, 2), dtype=torch.long),
            ctx=torch.zeros((0, 2, 1), dtype=torch.float32),
            valid_mask=torch.zeros((0, 2), dtype=torch.bool),
        ),
        "test": History(
            items=torch.zeros((0, 2), dtype=torch.long),
            ctx=torch.zeros((0, 2, 1), dtype=torch.float32),
            valid_mask=torch.zeros((0, 2), dtype=torch.bool),
        ),
    }

    dataset = dm._make_dataset("train")

    assert isinstance(dataset, RankerDataset)
    assert len(dataset) == 1

    example = dataset[0]
    assert example.user_id.item() == 0
    assert example.candidate_labels.tolist() == [1.0, 0.0]
    assert example.positive_positions.item() == 0
    assert example.history_items.tolist() == [1, 2]
