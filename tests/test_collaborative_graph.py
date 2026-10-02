import unittest

import pandas as pd
import torch

from edurec import settings
from edurec.datasets.dataprocessor import DataProcessor
from edurec.datasets.cache import ProcessedData
from edurec.datasets.datamodule import ElearningDataModule
from edurec.datasets.knowledge_graph import build_knowledge_graph
from edurec.datasets.loaders import DatasetName
from edurec.recsys.archs.kgseq import KGSeq
from edurec.recsys.archs.kgtransformer import KGTransformer
from edurec.recsys.archs.modules.graph_user_fusion import GraphUserFusion
from edurec.recsys.configs import ModelConfig


class CollaborativeGraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.processor = DataProcessor(schema={})
        self.processor.user_id_map = {'u0': 0, 'u1': 1, 'u2': 2}
        self.processor.item_id_map = {'i0': 0, 'i1': 1, 'i2': 2}
        self.processor.column_groups = {
            'users': {'categorical': ['topic'], 'list': ['skills']},
            'items': {'categorical': ['topic'], 'list': []},
        }
        self.users = pd.DataFrame({
            settings.USER_COL: ['u2', 'u0', 'u1'],
            'topic': [None, 'math', 'math'],
            'skills': [[], ['algebra', 'logic'], ['logic']],
        })
        self.items = pd.DataFrame({
            settings.ITEM_COL: ['i2', 'i0', 'i1'],
            'topic': ['history', 'math', 'math'],
        })
        self.train = pd.DataFrame({
            settings.USER_COL: [0, 0, 1, 2, -1],
            settings.ITEM_COL: [0, 0, 1, 2, 0],
            settings.RELEVANT_COL: [1, 1, 1, 0, 1],
        })
        self.graph = build_knowledge_graph(
            self.items, torch.zeros(3, 0), self.processor,
            user_frame=self.users, train_interactions=self.train,
        )

    def test_graph_has_separate_attributes_and_only_relevant_train_pairs(self):
        self.assertEqual(self.graph['user'].num_nodes, 3)
        self.assertEqual(self.graph['attr::topic'].num_nodes, 2)
        self.assertEqual(self.graph['user_attr::topic'].num_nodes, 1)
        self.assertEqual(self.graph['user_attr::skills'].num_nodes, 2)
        pairs = self.graph['user', 'interacts', 'item'].edge_index
        torch.testing.assert_close(pairs, torch.tensor([[0, 1], [0, 1]]))
        torch.testing.assert_close(
            self.graph['item', 'rev_interacts', 'user'].edge_index,
            pairs.flip(0),
        )
        # Raw feature frames need ID mapping; their row order is not the node order.
        topic_edges = self.graph['user', 'topic', 'user_attr::topic'].edge_index
        torch.testing.assert_close(topic_edges, torch.tensor([[0, 1], [0, 0]]))

    def test_fusion_falls_back_for_missing_graph_or_history(self):
        fusion = GraphUserFusion(8)
        seq, users = torch.randn(4, 8), torch.randn(3, 8)
        fused, available = fusion(
            seq, users, torch.tensor([0, 1, 2, -1]),
            torch.tensor([False, True, False, True]),
            self.graph.edge_index_dict, True,
        )
        torch.testing.assert_close(available, torch.tensor([True, True, False, False]))
        torch.testing.assert_close(fused[0], users[0])
        torch.testing.assert_close(fused[2:], seq[2:])
        disabled, available = fusion(
            seq, users, torch.tensor([0, 1, 2, -1]),
            torch.ones(4, dtype=torch.bool), self.graph.edge_index_dict, False,
        )
        torch.testing.assert_close(disabled, seq)
        self.assertFalse(available.any())

    def test_datamodule_excludes_validation_and_test_interactions(self):
        dm = ElearningDataModule(
            dataset=next(iter(DatasetName)), batch_size=2,
            test_ratio=0.2, val_ratio=0.1,
        )
        held_out = pd.DataFrame({
            settings.USER_COL: [2], settings.ITEM_COL: [2],
            settings.RELEVANT_COL: [1],
        })
        dm.artifacts = ProcessedData(
            train=self.train, val=held_out, test=held_out,
            user_features=self.users, item_features=self.items,
            i_static_feats=torch.zeros(3, 0), data_processor=self.processor,
        )
        torch.testing.assert_close(
            dm.knowledge_graph['user', 'interacts', 'item'].edge_index,
            torch.tensor([[0, 1], [0, 1]]),
        )

    def test_users_without_attributes_and_empty_training_edges(self):
        self.processor.column_groups = {}
        graph = build_knowledge_graph(
            self.items, torch.zeros(3, 0), self.processor,
            user_frame=self.users, train_interactions=self.train.iloc[:0],
        )
        self.assertEqual(set(graph.node_types), {'user', 'item'})
        self.assertEqual(graph['user', 'interacts', 'item'].edge_index.shape, (2, 0))
        cfg = ModelConfig(
            num_users=3, num_items=3, num_item_dense_feats=0,
            num_item_text_feats=0, emb_dim=8, gru_hidden_dim=8,
            use_user_features=False, seq_cell='lstm',
            kg_edge_types=list(graph.edge_types),
        )
        scores = KGSeq(cfg)(
            h_ids=torch.zeros(3, 1, dtype=torch.long),
            h_mask=torch.zeros(3, 1, dtype=torch.bool),
            edge_index=graph.edge_index_dict,
            i_static_feats=torch.zeros(3, 0),
            u_static_feats=torch.zeros(3, 0),
            u_cat_feats=torch.zeros(3, 0, dtype=torch.long),
            user_ids=torch.arange(3),
        )
        self.assertEqual(scores.shape, (3, 3))
        self.assertTrue(torch.isfinite(scores).all())

    def test_both_architectures_score_and_backpropagate(self):
        nodes = {
            node: self.graph[node].num_nodes for node in self.graph.node_types
            if node not in {'user', 'item'}
        }
        for architecture in (KGSeq, KGTransformer):
            for scorer in ('dot', 'mlp', 'candidate_attention'):
                with self.subTest(architecture=architecture.__name__, scorer=scorer):
                    cfg = ModelConfig(
                        num_users=3, num_items=3, num_item_dense_feats=0,
                        num_item_text_feats=0, emb_dim=8, gru_hidden_dim=8,
                        transformer_hidden_dim=8, transformer_heads=2,
                        hidden_dims=[16], scorer_type=scorer,
                        kg_node_counts=nodes, kg_edge_types=list(self.graph.edge_types),
                        num_user_dense_feats=1, dropout=0,
                    )
                    model = architecture(cfg).eval()
                    args = dict(
                        h_ids=torch.tensor([[1, 0], [0, 0], [0, 0]]),
                        h_mask=torch.tensor([[True, False], [False, False], [False, False]]),
                        edge_index=self.graph.edge_index_dict,
                        i_static_feats=torch.zeros(3, 0),
                        u_static_feats=torch.tensor([[1.], [2.], [3.]]),
                        u_cat_feats=torch.zeros(3, 0, dtype=torch.long),
                        user_ids=torch.arange(3),
                    )
                    full = model(**args)
                    candidates = torch.tensor([[0, 2], [1, 0], [2, 1]])
                    selected = model(**args, candidate_item_ids=candidates)
                    self.assertTrue(torch.isfinite(full).all())
                    torch.testing.assert_close(selected, full.gather(1, candidates))
                    torch.nn.functional.cross_entropy(full, torch.tensor([1, 2, 0])).backward()
                    self.assertGreater(model.kg.user_emb.weight.grad.abs().sum().item(), 0)
                    self.assertGreater(model.kg.attr_embs['user_attr::skills'].weight.grad.abs().sum().item(), 0)
                    self.assertGreater(model.graph_user_fusion.gate.weight.grad.abs().sum().item(), 0)
                    self.assertGreater(model.kg.relation_logits[0].grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
