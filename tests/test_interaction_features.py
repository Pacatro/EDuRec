"""Unit tests for on-the-fly interaction context and time-gap gathering.

These lock the alignment between history items, interaction features and the
computed ``delta_t = t - t_{t-1}`` (with delta 0 at the first valid step).
"""

import unittest

import pandas as pd
import torch

from edurec import settings
from edurec.datasets.recsys_dataset import RecSysDataset


class InteractionContextGatherTest(unittest.TestCase):
    def _dataset(self) -> RecSysDataset:
        interactions = pd.DataFrame(
            {
                settings.USER_COL: [0, 0],
                settings.ITEM_COL: [5, 6],
            }
        )
        dense = torch.tensor([[1.0], [2.0], [3.0]])
        cat = torch.tensor([[0], [1], [2]])
        timestamps = torch.tensor([100.0, 130.0, 150.0])
        # Row 0: no history. Row 1: events 0 and 1 (timestamps 100 then 130).
        history_items = torch.tensor([[0, 0, 0], [1, 2, 0]])
        history_mask = torch.tensor(
            [[False, False, False], [True, True, False]]
        )
        context_index = torch.tensor([[-1, -1, -1], [0, 1, -1]])
        return RecSysDataset(
            interactions=interactions,
            history_items=history_items,
            history_valid_mask=history_mask,
            history_context_index=context_index,
            interaction_dense=dense,
            interaction_cat=cat,
            interaction_timestamps=timestamps,
        )

    def test_empty_history_has_zero_context_and_time(self) -> None:
        query = self._dataset()[0]
        self.assertEqual(query.history_dense_features.shape, (3, 1))
        self.assertEqual(float(query.history_dense_features.abs().sum()), 0.0)
        self.assertEqual(float(query.history_delta_times.abs().sum()), 0.0)

    def test_context_and_delta_align_with_history(self) -> None:
        query = self._dataset()[1]
        torch.testing.assert_close(
            query.history_dense_features[:, 0], torch.tensor([1.0, 2.0, 0.0])
        )
        torch.testing.assert_close(
            query.history_cat_features[:, 0], torch.tensor([0, 1, 0])
        )
        torch.testing.assert_close(
            query.history_timestamps, torch.tensor([100.0, 130.0, 0.0])
        )
        # First valid step is 0; the next is the 30s gap; padding stays 0.
        torch.testing.assert_close(
            query.history_delta_times, torch.tensor([0.0, 30.0, 0.0])
        )

    def test_epoch_second_gaps_survive_float64(self) -> None:
        # float32 cannot represent a 30-second gap at ~1.7e9 seconds.
        base = 1_700_000_000.0
        interactions = pd.DataFrame(
            {settings.USER_COL: [0], settings.ITEM_COL: [5]}
        )
        dataset = RecSysDataset(
            interactions=interactions,
            history_items=torch.tensor([[1, 2]]),
            history_valid_mask=torch.tensor([[True, True]]),
            history_context_index=torch.tensor([[0, 1]]),
            interaction_dense=torch.zeros(2, 1, dtype=torch.float64),
            interaction_cat=torch.zeros(2, 1, dtype=torch.long),
            interaction_timestamps=torch.tensor(
                [base, base + 30.0], dtype=torch.float64
            ),
        )
        query = dataset[0]
        torch.testing.assert_close(
            query.history_delta_times, torch.tensor([0.0, 30.0])
        )


if __name__ == "__main__":
    unittest.main()
