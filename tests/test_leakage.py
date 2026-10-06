"""Explicit leakage-prevention tests for the kg_rnn refactor.

They assert the two structural guarantees:
1. The item knowledge graph cannot consume interactions at all.
2. Repeated ``(user, item)`` events are preserved by default and only collapsed
   when requested or when no timestamp can order them.
"""

import inspect
import unittest

import pandas as pd

from edurec import settings
from edurec.datasets.knowledge_graph import build_knowledge_graph
from edurec.datasets.preprocessing import split_data


class KnowledgeGraphLeakageTest(unittest.TestCase):
    def test_builder_cannot_accept_interactions(self) -> None:
        params = list(inspect.signature(build_knowledge_graph).parameters)
        self.assertEqual(params, ["item_frame", "item_feats", "processor"])
        self.assertNotIn("train_interactions", params)
        self.assertNotIn("interactions", params)


class DeduplicationPolicyTest(unittest.TestCase):
    def _events(self, with_time: bool) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                settings.USER_COL: [0, 0, 0, 0],
                settings.ITEM_COL: [1, 1, 2, 3],
            }
        )
        if with_time:
            frame[settings.TIME_COL] = [10, 20, 30, 40]
        return frame

    def test_timestamps_preserve_distinct_events_by_default(self) -> None:
        train, val, test = split_data(
            self._events(with_time=True),
            test_ratio=0.2,
            val_ratio=0.1,
            min_interactions=3,
            random_state=0,
            deduplicate=False,
        )
        self.assertEqual(len(train) + len(val) + len(test), 4)

    def test_forced_deduplication_collapses_pairs(self) -> None:
        train, val, test = split_data(
            self._events(with_time=True),
            test_ratio=0.2,
            val_ratio=0.1,
            min_interactions=3,
            random_state=0,
            deduplicate=True,
        )
        self.assertEqual(len(train) + len(val) + len(test), 3)

    def test_missing_timestamp_always_deduplicates(self) -> None:
        train, val, test = split_data(
            self._events(with_time=False),
            test_ratio=0.2,
            val_ratio=0.1,
            min_interactions=3,
            random_state=0,
            deduplicate=False,
        )
        self.assertEqual(len(train) + len(val) + len(test), 3)


if __name__ == "__main__":
    unittest.main()
