"""Regression tests for the static-feature registration refactor.

Static, batch-invariant tensors live on the architecture as buffers and are
attached once with ``register_static``; ``forward`` only receives per-batch
data. Item embeddings are cached during evaluation and invalidated on
``train()``.
"""

import unittest

import torch
from torch_geometric.data import HeteroData

from edurec.datasets.recsys_dataset import RecSysQuery
from edurec.recsys.archs import KGSeq, SASRecText
from edurec.recsys.archs.modules.kg_encoder import EdgeType
from edurec.recsys.configs import ModelArch, ModelConfig

EMB_DIM = 8
NUM_ITEMS = 7
NUM_USERS = 5
EDGE_TYPE: EdgeType = ("item", "has", "attr")
ARCHES = (ModelArch.KG_RNN, ModelArch.SASREC_TEXT)


def make_cfg(arch: ModelArch = ModelArch.KG_RNN) -> ModelConfig:
    return ModelConfig(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        num_item_dense_feats=6,
        num_item_text_feats=4,
        num_user_dense_feats=3,
        num_interaction_dense_feats=2,
        user_cat_cardinalities=[4],
        interaction_cat_cardinalities=[5],
        kg_node_counts={"attr": 4},
        kg_edge_types=[list(EDGE_TYPE)],
        emb_dim=EMB_DIM,
        arch=arch,
        hidden_dims=[EMB_DIM],
        rnn_hidden_dim=EMB_DIM,
        transformer_hidden_dim=EMB_DIM,
        gnn_heads=2,
    )


def static_inputs() -> tuple[dict[EdgeType, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_index = {
        EDGE_TYPE: torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 0, 1, 2]], dtype=torch.long
        )
    }
    return (
        edge_index,
        torch.randn(NUM_ITEMS, 6),
        torch.randn(NUM_USERS, 3),
        torch.randint(0, 4, (NUM_USERS, 1)),
    )


def make_arch(arch: ModelArch):
    model = (
        KGSeq(make_cfg(ModelArch.KG_RNN))
        if arch == ModelArch.KG_RNN
        else SASRecText(make_cfg(ModelArch.SASREC_TEXT))
    )
    model.register_static(*static_inputs())
    return model


def batch_kwargs(batch: int = 3, steps: int = 4):
    return {
        "h_ids": torch.randint(0, NUM_ITEMS, (batch, steps)),
        "h_mask": torch.ones(batch, steps, dtype=torch.bool),
        "user_ids": torch.randint(0, NUM_USERS, (batch,)),
        "h_dense": torch.randn(batch, steps, 2),
        "h_cat": torch.randint(0, 5, (batch, steps, 1)),
        "h_delta": torch.rand(batch, steps) * 100.0,
    }


class StaticFeatureTest(unittest.TestCase):
    def test_register_static_creates_buffers(self):
        model = make_arch(ModelArch.KG_RNN)
        buffers = dict(model.named_buffers())
        for name in ("i_static_feats", "u_static_feats", "u_cat_feats", "edge_index_0"):
            self.assertIn(name, buffers)
        self.assertEqual(model.edge_index[EDGE_TYPE].shape, (2, NUM_ITEMS))

    def test_forward_shapes(self):
        for arch in ARCHES:
            with self.subTest(arch=arch.value):
                model = make_arch(arch)
                model.eval()
                with torch.no_grad():
                    candidates = torch.randint(0, NUM_ITEMS, (3, 3))
                    restricted = model(
                        **batch_kwargs(), candidate_item_ids=candidates
                    )
                    self.assertEqual(restricted.shape, (3, 3))
                    full = model(**batch_kwargs())
                    self.assertEqual(full.shape, (3, NUM_ITEMS))

    def test_item_embeddings_cached_only_in_eval(self):
        for arch in ARCHES:
            with self.subTest(arch=arch.value):
                model = make_arch(arch)
                model.train()
                model.item_embeddings()
                self.assertIsNone(
                    model._item_emb_cache, "training must not populate the cache"
                )

                model.eval()
                with torch.no_grad():
                    first = model.item_embeddings()
                    second = model.item_embeddings()
                self.assertIs(first, second)

                model.train()
                self.assertIsNone(
                    model._item_emb_cache, "train() must invalidate the cache"
                )


class RecSysWiringTest(unittest.TestCase):
    def _knowledge_graph(self) -> HeteroData:
        graph = HeteroData()
        graph["attr"].num_nodes = 4
        graph[EDGE_TYPE].edge_index = static_inputs()[0][EDGE_TYPE]
        return graph

    def test_recsys_forward_uses_registered_static_features(self):
        from edurec.recsys.recsys import RecSys

        _, i_static, u_static, u_cat = static_inputs()
        model = RecSys(
            cfg=make_cfg(ModelArch.KG_RNN),
            knowledge_graph=self._knowledge_graph(),
            i_static_feats=i_static,
            u_static_feats=u_static,
            u_cat_feats=u_cat,
        )
        batch = RecSysQuery(
            query_id=torch.arange(3),
            user_id=torch.tensor([0, 2, 4]),
            history_items=torch.randint(0, NUM_ITEMS, (3, 4)),
            history_valid_mask=torch.ones(3, 4, dtype=torch.bool),
            history_dense_features=torch.randn(3, 4, 2),
            history_cat_features=torch.randint(0, 5, (3, 4, 1)),
            history_timestamps=torch.zeros(3, 4),
            history_delta_times=torch.rand(3, 4) * 100.0,
            target_item_id=torch.randint(0, NUM_ITEMS, (3,)),
            negative_item_ids=torch.randint(0, NUM_ITEMS, (3, 2)),
            sample_weight=torch.ones(3),
        )
        model.eval()
        with torch.no_grad():
            candidates = torch.tensor([[0, 1, 2], [3, 4, 5], [1, 6, 0]])
            self.assertEqual(model(batch, candidate_item_ids=candidates).shape, (3, 3))
            self.assertEqual(model(batch).shape, (3, NUM_ITEMS))


if __name__ == "__main__":
    unittest.main()
