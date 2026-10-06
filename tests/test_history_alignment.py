"""Regression tests for chronological history construction.

These tests define the leakage guarantee of the kg_rnn architecture: every
history field is aligned by step and only contains events strictly before the
current row. They are fast, CPU-only and do not load any pretrained model.
"""

import unittest

import pandas as pd
import torch

from edurec import settings
from edurec.datasets.user_history import build_histories


def _frame(pairs: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            settings.USER_COL: [user for user, _ in pairs],
            settings.ITEM_COL: [item for _, item in pairs],
        }
    )


class HistoryAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.splits = {
            "train": _frame([(0, 0), (0, 1), (1, 5)]),
            "val": _frame([(0, 2)]),
            "test": _frame([(0, 3), (1, 6)]),
        }
        self.global_items = [
            int(item)
            for split in ("train", "val", "test")
            for item in self.splits[split][settings.ITEM_COL]
        ]
        self.max_history = 8
        self.histories = build_histories(
            self.splits, max_history=self.max_history, enabled=True
        )
        self.offsets = {"train": 0, "val": 3, "test": 4}

    def test_shapes_and_dtypes(self) -> None:
        for split, df in self.splits.items():
            items, mask, context = self.histories[split]
            self.assertEqual(items.shape, (len(df), self.max_history))
            self.assertEqual(mask.shape, (len(df), self.max_history))
            self.assertEqual(context.shape, (len(df), self.max_history))
            self.assertEqual(items.dtype, torch.long)
            self.assertEqual(mask.dtype, torch.bool)
            self.assertEqual(context.dtype, torch.long)

    def test_every_history_entry_is_a_past_event(self) -> None:
        for split, df in self.splits.items():
            items, mask, context = self.histories[split]
            for row in range(len(df)):
                global_index = self.offsets[split] + row
                valid = mask[row].nonzero(as_tuple=False).flatten().tolist()
                for step in valid:
                    past = int(context[row, step])
                    # Strictly before the current row: no self/future leakage.
                    self.assertLess(past, global_index)
                    # The item at that step matches the same global event.
                    self.assertEqual(
                        int(items[row, step]), self.global_items[past] + 1
                    )
                # Padding is explicit and consistent across the three fields.
                for step in range(len(valid), self.max_history):
                    self.assertEqual(int(context[row, step]), -1)
                    self.assertEqual(int(items[row, step]), 0)
                    self.assertFalse(bool(mask[row, step]))

    def test_target_never_appears_in_its_own_history(self) -> None:
        for split, df in self.splits.items():
            items, mask, _ = self.histories[split]
            for row in range(len(df)):
                target = int(df[settings.ITEM_COL].iloc[row]) + 1
                valid_items = [
                    int(items[row, step])
                    for step in mask[row].nonzero(as_tuple=False).flatten().tolist()
                ]
                self.assertNotIn(target, valid_items)

    def test_later_splits_see_earlier_history(self) -> None:
        _, val_mask, _ = self.histories["val"]
        self.assertEqual(int(val_mask[0].sum()), 2)  # both train events of user 0
        _, test_mask, _ = self.histories["test"]
        self.assertEqual(int(test_mask[0].sum()), 3)  # two train + one val
        self.assertEqual(int(test_mask[1].sum()), 1)  # user 1 train event

    def test_disabled_returns_empty_width(self) -> None:
        hist = build_histories(self.splits, max_history=self.max_history, enabled=False)
        for split, df in self.splits.items():
            items, mask, context = hist[split]
            self.assertEqual(items.shape, (len(df), 0))
            self.assertEqual(mask.shape, (len(df), 0))
            self.assertEqual(context.shape, (len(df), 0))
            self.assertEqual(items.dtype, torch.long)
            self.assertEqual(mask.dtype, torch.bool)
            self.assertEqual(context.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
