"""Whole-history sampling must match a sorted prefix without staging to disk."""

import hashlib
import json
import random
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from edurec.datasets.mooccubex import (
    MooccubexSelection,
    _collect_candidates,
    select_enrollments,
)


class TwoPassSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "user.json"
        self.catalog = {f"C_{i}" for i in range(1, 32)}
        self.items = pd.DataFrame(
            {
                "item_id": sorted(self.catalog),
                "concept": [["K_shared"] for _ in self.catalog],
                "field": [["math"] for _ in self.catalog],
            }
        )
        self.cfg = MooccubexSelection(
            max_interactions=25,
            candidate_multiplier=3,
            min_user_courses=3,
            min_course_users=1,
            min_history_retention=1.0,
            max_courses=None,
        )

    @staticmethod
    def record(index, size):
        return {
            "id": f"U_{index}",
            "gender": 1,
            "school": "school",
            "year_of_birth": 1990,
            "course_order": list(range(1, size + 1)),
            "enroll_time": [f"2020-01-{i:02d}" for i in range(1, size + 1)],
        }

    def write(self, records):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in records))

    def collect(self, records, cfg=None):
        self.write(records)
        cfg = cfg or self.cfg
        budget = cfg.max_interactions
        return _collect_candidates(
            self.path, self.catalog, self.catalog, self.catalog, cfg, budget, Counter()
        )

    def test_heap_matches_sort_then_prefix_for_variable_history_lengths(self):
        for seed in range(30):
            rng = random.Random(seed)
            records = [self.record(i, rng.randint(3, 31)) for i in range(100)]
            eligible = [
                record for record in records if len(record["course_order"]) <= 25
            ]
            ordered = sorted(
                eligible,
                key=lambda record: (
                    hashlib.sha256(f"42:{record['id']}".encode()).hexdigest(),
                    record["id"],
                ),
            )
            expected, used = [], 0
            for record in ordered:
                if used + len(record["course_order"]) > 75:
                    break
                expected.append(record["id"])
                used += len(record["course_order"])
            rng.shuffle(records)
            candidates = self.collect(records)
            self.assertEqual(
                [candidate.priority[1] for candidate in candidates], expected
            )
            self.assertLessEqual(
                sum(len(candidate.history) for candidate in candidates), 75
            )

    def test_final_selection_is_order_independent_and_never_truncates(self):
        records = [self.record(i, 5) for i in range(80)]
        self.write(records)
        first, users, report = select_enrollments(self.path, self.items, self.cfg)
        self.write(list(reversed(records)))
        second, _, _ = select_enrollments(self.path, self.items, self.cfg)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 25)
        self.assertTrue(first.groupby("user_id").size().eq(5).all())
        self.assertEqual(len(users), 5)
        self.assertEqual(report["scanned_users"], 80)
        self.assertEqual(report["eligible_source_users"], 80)
        self.assertLessEqual(report["candidate_pool_interactions"], 75)

    def test_duplicate_users_do_not_inflate_support_or_counts(self):
        records = [self.record(i, 5) for i in range(10)]
        self.write(records + records)
        events, _, report = select_enrollments(self.path, self.items, self.cfg)
        self.assertEqual(report["duplicate_user_records"], 10)
        self.assertEqual(report["eligible_source_users"], 10)
        self.assertEqual(report["scanned_users"], 20)
        self.assertFalse(events.duplicated(["user_id", "item_id"]).any())

    def test_unlimited_budget_keeps_all_users_and_oversized_histories(self):
        records = [self.record(i, 31) for i in range(10)]
        candidates = self.collect(records, replace(self.cfg, max_interactions=None))
        self.assertEqual(len(candidates), 10)
        self.assertEqual(sum(len(candidate.history) for candidate in candidates), 310)

    def test_community_excludes_disconnected_courses_and_keeps_histories(self):
        items = self.items.copy()
        items["concept"] = items["item_id"].map(
            lambda course: ["K_shared"] if course in {"C_1", "C_2", "C_3"} else []
        )
        self.write([self.record(i, 3) for i in range(10)] + [self.record(10, 5)])
        cfg = replace(
            self.cfg, max_courses=10, seed_course="C_1", max_interactions=None
        )
        events, users, report = select_enrollments(self.path, items, cfg)
        self.assertEqual(set(events["item_id"]), {"C_1", "C_2", "C_3"})
        self.assertEqual(len(users), 10)
        self.assertTrue(events.groupby("user_id").size().eq(3).all())
        self.assertEqual(report["selected_catalog_size_before_core"], 3)

    def test_no_temporary_storage_and_exactly_two_source_passes(self):
        self.write([self.record(i, 5) for i in range(10)])
        original = Path.open
        reads = []

        def open_path(path, *args, **kwargs):
            if path == self.path:
                reads.append(path)
            return original(path, *args, **kwargs)

        with (
            patch.object(Path, "open", open_path),
            patch.object(
                tempfile,
                "TemporaryDirectory",
                side_effect=AssertionError("temporary storage"),
            ),
        ):
            select_enrollments(self.path, self.items, self.cfg)
        self.assertEqual(len(reads), 2)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])


if __name__ == "__main__":
    unittest.main()
