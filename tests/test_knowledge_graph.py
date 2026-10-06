"""Regression tests for the item-only knowledge graph builder.

The tests are self-contained and fast: they build a tiny :class:`DataProcessor`
without text columns so no sentence-transformer model is loaded, and they never
touch the network or a GPU.
"""

import unittest
from typing import Any

import pandas as pd
import torch

from edurec import settings
from edurec.datasets.dataprocessor import DataProcessor
from edurec.datasets.knowledge_graph import build_knowledge_graph


def _edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return {
        (int(source), int(target))
        for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist())
    }


def _tiny_schema() -> dict[str, dict[str, Any]]:
    return {
        "users": {"cat": ["gender"]},
        "items": {
            "cat": ["category", "subcategory", "title"],
            "list": ["tags", "prerequisites"],
            "refs": {"prerequisites": "title"},
            "cooc": [("subcategory", "category")],
        },
        "inter": {"num": ["rating"]},
    }


class KnowledgeGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        # Exclude text so DataProcessor never reaches the embedding model.
        self._prev_feature_types = settings.PREPROCESS_FEATURE_TYPES
        settings.PREPROCESS_FEATURE_TYPES = (
            "numeric",
            "categorical",
            "list",
            "time",
        )

        self.items = pd.DataFrame(
            {
                settings.ITEM_COL: ["i1", "i2", "i3", "i4", "i5"],
                "category": ["math", "math", "science", "science", "math"],
                "subcategory": [
                    "algebra",
                    "calculus",
                    "physics",
                    "chemistry",
                    "algebra",
                ],
                "title": [
                    "Algebra I",
                    "Calculus I",
                    "Physics",
                    "Chemistry",
                    "Algebra II",
                ],
                "tags": [
                    ["basics", "equations"],
                    ["limits"],
                    ["mechanics"],
                    ["atoms"],
                    ["equations"],
                ],
                "prerequisites": [
                    [],
                    ["Algebra I"],
                    ["Calculus I"],
                    [],
                    ["Algebra I"],
                ],
            }
        )
        self.users = pd.DataFrame(
            {settings.USER_COL: ["u1", "u2"], "gender": ["f", "m"]}
        )
        self.interactions = pd.DataFrame(
            {
                settings.USER_COL: ["u1", "u2"],
                settings.ITEM_COL: ["i1", "i3"],
                "rating": [4.0, 5.0],
            }
        )

        self.processor = DataProcessor(schema=_tiny_schema()).fit(
            self.users, self.items, self.interactions
        )
        self.graph = build_knowledge_graph(
            self.items,
            torch.randn(len(self.items), 2),
            self.processor,
        )

    def tearDown(self) -> None:
        settings.PREPROCESS_FEATURE_TYPES = self._prev_feature_types

    def test_has_no_user_nodes_or_edges(self) -> None:
        self.assertNotIn("user", self.graph.node_types)
        for source, _, target in self.graph.edge_types:
            self.assertNotEqual(source, "user")
            self.assertNotEqual(target, "user")

    def test_reference_edges_have_distinct_reverse(self) -> None:
        self.assertIn(("item", "ref::prerequisites", "item"), self.graph.edge_types)
        self.assertIn(
            ("item", "rev_ref::prerequisites", "item"), self.graph.edge_types
        )

        forward = _edge_set(
            self.graph["item", "ref::prerequisites", "item"].edge_index
        )
        reverse = _edge_set(
            self.graph["item", "rev_ref::prerequisites", "item"].edge_index
        )
        self.assertGreater(len(forward), 0)
        self.assertEqual(reverse, {(target, source) for source, target in forward})

    def test_attribute_edges_exist_both_directions(self) -> None:
        self.assertIn(("item", "category", "attr::category"), self.graph.edge_types)
        self.assertIn(
            ("attr::category", "rev_category", "item"), self.graph.edge_types
        )

        forward = _edge_set(
            self.graph["item", "category", "attr::category"].edge_index
        )
        reverse = _edge_set(
            self.graph["attr::category", "rev_category", "item"].edge_index
        )
        self.assertEqual(reverse, {(target, source) for source, target in forward})

    def test_no_self_loops_in_same_type_edges(self) -> None:
        # Self-loops are only well defined for same-type edges; numeric index
        # collisions across different node types are not self-loops.
        for source, relation, target in self.graph.edge_types:
            if source != target:
                continue
            edge_index = self.graph[source, relation, target].edge_index
            self.assertFalse(
                bool((edge_index[0] == edge_index[1]).any()),
                msg=f"Self loop found in {(source, relation, target)}",
            )


if __name__ == "__main__":
    unittest.main()
