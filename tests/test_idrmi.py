"""Regression tests for the IDRMI baseline.

The tests build a tiny synthetic module so they run on CPU without any cached
dataset. They focus on the properties the evaluation protocol relies on:
leakage-free factor tables, finite scores, candidate/full-catalog consistency
and a working training step.
"""

import unittest

import torch

from edurec.datasets import RecSysQuery
from edurec.evaluation.baselines.idrmi import (
    IDRMIConfig,
    IDRMIModule,
    _build_interest_tables,
)
from edurec.recsys.configs import TrainConfig

NUM_USERS = 24
NUM_ITEMS = 12
NUM_ATTR = 6
EMB_DIM = 8


def _module(train_interactions: torch.Tensor) -> IDRMIModule:
    torch.manual_seed(0)
    item_feats = torch.randn(NUM_ITEMS, 4)
    item_edges = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 0, 1, 2, 3, 4]], dtype=torch.long
    )
    attr_edges = item_edges.flip(0)
    edge_index = {
        ("item", "has", "attr::x"): item_edges,
        ("attr::x", "rev_has", "item"): attr_edges,
    }
    return IDRMIModule(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        i_static_feats=item_feats,
        edge_index=edge_index,
        node_counts={"attr::x": NUM_ATTR},
        train_interactions=train_interactions,
        cfg=IDRMIConfig(emb_dim=EMB_DIM, top_users=3, top_items=2, ngcf_layers=2),
        train_cfg=TrainConfig(topks=[5], epochs=1, patience=1),
        val_topk=5,
    )


def _train_interactions() -> torch.Tensor:
    pairs = []
    for user in range(NUM_USERS):
        for item in range(user % 4, NUM_ITEMS, 3):
            pairs.append((user, item))
    return torch.tensor(pairs, dtype=torch.long)


class BaselineRegistryTests(unittest.TestCase):
    def test_idrmi_is_registered(self) -> None:
        from edurec.evaluation import baseline_labels, get_baseline

        self.assertIn("IDRMI", baseline_labels())
        self.assertIsNotNone(get_baseline("IDRMI"))

    def test_labels_are_unique(self) -> None:
        from edurec.evaluation import baseline_labels

        labels = baseline_labels()
        self.assertEqual(len(labels), len(set(labels)))


class InterestTableTests(unittest.TestCase):
    def test_tables_use_train_interactions_only(self) -> None:
        interactions = _train_interactions()
        matrix, course_match = _build_interest_tables(
            interactions, NUM_USERS, NUM_ITEMS
        )

        self.assertEqual(tuple(matrix.shape), (NUM_USERS, NUM_ITEMS))
        self.assertEqual(tuple(course_match.shape), (NUM_USERS, NUM_ITEMS))
        # A held-out pair never appears in the train-only interaction matrix.
        matrix_cpu = matrix
        self.assertEqual(float(matrix_cpu[0, 1].item()), 0.0)
        # Every distinct train pair is present exactly once.
        unique_pairs = {tuple(pair) for pair in interactions.tolist()}
        self.assertEqual(int(matrix_cpu.sum().item()), len(unique_pairs))

    def test_course_match_is_finite_and_bounded(self) -> None:
        _, course_match = _build_interest_tables(
            _train_interactions(), NUM_USERS, NUM_ITEMS
        )
        self.assertTrue(torch.isfinite(course_match).all())
        self.assertGreaterEqual(float(course_match.min().item()), 0.0)
        self.assertLessEqual(float(course_match.max().item()), 1.0 + 1e-6)


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _module(_train_interactions())
        self.module.eval()
        self.users = torch.arange(6, dtype=torch.long)

    def test_scores_are_finite_and_shaped(self) -> None:
        scores = self.module.score(
            self.users, torch.arange(NUM_ITEMS, dtype=torch.long)
        )
        self.assertEqual(tuple(scores.shape), (6, NUM_ITEMS))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue((scores > 0).all())

    def test_candidate_and_full_catalog_scores_agree(self) -> None:
        full = self.module.score(self.users, torch.arange(NUM_ITEMS, dtype=torch.long))
        candidates = torch.tensor([3, 1, 7, 0], dtype=torch.long)
        subset = self.module.score(self.users, candidates)
        expected = full[:, candidates]
        self.assertTrue(torch.allclose(subset, expected, atol=1e-4))

    def test_scoring_is_deterministic_in_eval(self) -> None:
        candidates = torch.arange(NUM_ITEMS, dtype=torch.long)
        first = self.module.score(self.users, candidates)
        second = self.module.score(self.users, candidates)
        self.assertTrue(torch.equal(first, second))

    def test_normalise_mix_range(self) -> None:
        mix = torch.tensor([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
        normalized = IDRMIModule._normalize_mix(mix)
        self.assertAlmostEqual(float(normalized.min().item()), 0.5, places=5)
        self.assertAlmostEqual(float(normalized.max().item()), 1.5, places=5)
        self.assertTrue(torch.isfinite(normalized).all())


class TrainingStepTests(unittest.TestCase):
    def test_training_step_backpropagates(self) -> None:
        module = _module(_train_interactions())
        module.train()
        module._refresh_embedding_factors()

        batch = RecSysQuery(
            query_id=torch.arange(4, dtype=torch.long),
            user_id=torch.arange(4, dtype=torch.long),
            history_items=torch.zeros(4, 0, dtype=torch.long),
            history_valid_mask=torch.zeros(4, 0, dtype=torch.bool),
            history_dense_features=torch.zeros(4, 0, 1),
            history_cat_features=torch.zeros(4, 0, 1, dtype=torch.long),
            history_timestamps=torch.zeros(4, 0),
            history_delta_times=torch.zeros(4, 0),
            target_item_id=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            negative_item_ids=torch.tensor(
                [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.long
            ),
        )

        loss = module.training_step(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(module.ngcf.weights[0].grad)
        self.assertTrue(torch.isfinite(module.ngcf.weights[0].grad).all())


if __name__ == "__main__":
    unittest.main()
