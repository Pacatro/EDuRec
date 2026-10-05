import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from edurec import settings
from edurec.datasets.atomic_files import save_atomic_files
from edurec.datasets.cache import CACHE_VERSION, ProcessedData, processed_cache_exists
from edurec.datasets.datamodule import ElearningDataModule
from edurec.datasets.dataprocessor import DataProcessor
from edurec.datasets.loaders import DatasetName
from edurec.datasets.preprocessing import preprocess
from edurec.datasets.recsys_dataset import rating_sample_weights
from edurec.recsys.configs import ModelArch, ModelConfig, TrainConfig
from edurec.recsys.recsys import RecSys


def make_datamodule(explicit=True, temporal=True):
    frames = {
        'train': pd.DataFrame({'user_id': [0, 0], 'item_id': [0, 1]}),
        'val': pd.DataFrame({'user_id': [0], 'item_id': [2]}),
        'test': pd.DataFrame({'user_id': [0], 'item_id': [3]}),
    }
    for frame in frames.values():
        if explicit:
            frame['rating'] = [1, 5] if len(frame) == 2 else [3]
        if temporal:
            frame['timestamp'] = frame['item_id'] + 100
    schema = {name: {} for name in ('users', 'items', 'inter')}
    dm = ElearningDataModule(DatasetName.ITM, 2, .2, .1, random_state=42)
    dm.artifacts = preprocess(
        DataProcessor(schema),
        pd.DataFrame({'user_id': [0]}),
        pd.DataFrame({'item_id': range(10)}),
        frames['train'], frames['val'], frames['test'],
    )
    dm.setup('fit')
    dm.setup('test')
    return dm


def make_model(dm, arch=ModelArch.KG_RNN):
    cfg = ModelConfig(
        num_users=dm.num_users, num_items=dm.num_items,
        num_item_dense_feats=0, num_item_text_feats=0, has_history=dm.has_history,
        kg_node_counts=dm.kg_node_counts,
        kg_edge_types=[list(edge) for edge in dm.kg_edge_types],
        emb_dim=8, gru_hidden_dim=8, gru_layers=1,
        transformer_hidden_dim=8, transformer_heads=2, transformer_layers=1,
        hidden_dims=[8], dropout=0, arch=arch,
    )
    model = RecSys(
        cfg, dm.knowledge_graph, dm.i_static_feats, dm.u_static_feats, dm.u_cat_feats,
        train_cfg=TrainConfig(topks=[1]), val_topk=1,
    )
    model.log = Mock()
    model.log_dict = Mock()
    return model


class RatingWeightsTests(unittest.TestCase):
    def test_scale_uses_only_training_and_keeps_low_ratings(self):
        splits = {
            'train': pd.DataFrame({'rating': [1., 3., 5.]}),
            'val': pd.DataFrame({'rating': [-10., 10.]}),
            'test': pd.DataFrame({'rating': []}),
        }
        weights = rating_sample_weights(splits)
        torch.testing.assert_close(weights['train'].mean(), torch.tensor(1.))
        self.assertTrue((weights['train'] > 0).all())
        self.assertAlmostEqual((weights['train'][-1] / weights['train'][0]).item(), 10., places=5)
        torch.testing.assert_close(weights['val'], weights['train'][[0, 2]])
        self.assertEqual(weights['test'].numel(), 0)
        self.assertNotIn('relevant', splits['train'])

    def test_constant_and_implicit_ratings_have_unit_weights(self):
        for df in (pd.DataFrame({'rating': [0., 0.]}), pd.DataFrame({'item_id': [0, 1]})):
            weights = rating_sample_weights({'train': df, 'val': df})
            torch.testing.assert_close(weights['train'], torch.ones(2))

    def test_nonfinite_ratings_are_rejected(self):
        with self.assertRaises(ValueError):
            rating_sample_weights({'train': pd.DataFrame({'rating': [1., np.nan]})})


class RankingPipelineTests(unittest.TestCase):
    def test_all_ratings_enter_histories_graph_and_exports(self):
        dm = make_datamodule()
        self.assertEqual(len(dm.train_ds), 2)
        self.assertEqual(len(dm.val_ds), 1)
        self.assertEqual(len(dm.test_ds), 1)
        self.assertEqual(dm.train_ds.history_items[1, 0].item(), 1)
        self.assertEqual(dm.test_ds.history_valid_mask[0].sum().item(), 3)
        edges = dm.knowledge_graph['user', 'interacts', 'item'].edge_index
        torch.testing.assert_close(edges, torch.tensor([[0, 0], [0, 1]]))
        for frame in dm.artifacts.splits().values():
            self.assertNotIn('relevant', frame)
            self.assertIn('rating', frame)
        with tempfile.TemporaryDirectory() as folder:
            paths = save_atomic_files(dm.artifacts, 'test', Path(folder))
            exported = pd.read_csv(paths['train.inter'], sep='\t')
            self.assertEqual(len(exported), 2)
            self.assertEqual(exported['item_id:token'].tolist(), [0, 1])
            dm.artifacts.save(Path(folder), {'version': CACHE_VERSION})
            self.assertTrue(processed_cache_exists(Path(folder)))
            restored = ProcessedData.load(Path(folder))
            pd.testing.assert_frame_equal(
                restored.splits()["train"], dm.artifacts.splits()["train"]
            )
            (Path(folder) / 'manifest.json').write_text('{"version": 4}')
            self.assertFalse(processed_cache_exists(Path(folder)))

    def test_weighted_loss_and_gradients_including_single_example_batches(self):
        dm = make_datamodule()
        model = make_model(dm)
        batch = next(iter(DataLoader(dm.train_ds, batch_size=2)))
        scores = torch.zeros(2, dm.num_items, requires_grad=True)
        with patch.object(model, 'forward', return_value=scores):
            loss = model.training_step(batch)
        expected = (F.cross_entropy(scores, batch.target_item_id, reduction='none')
                    * batch.sample_weight).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        assert scores.grad is not None
        self.assertAlmostEqual(scores.grad[1].norm().item() / scores.grad[0].norm().item(), 10., places=4)
        losses = []
        for single in DataLoader(dm.train_ds, batch_size=1):
            with patch.object(model, 'forward', return_value=torch.zeros(1, dm.num_items)):
                losses.append(model.training_step(single))
        self.assertAlmostEqual((losses[1] / losses[0]).item(), 10., places=4)

    def test_implicit_negative_sampling_and_loss_are_unchanged(self):
        dm = make_datamodule(explicit=False)
        torch.testing.assert_close(dm.train_ds.sample_weights, torch.ones(2))
        self.assertEqual(dm.train_ds.negative_item_ids.shape, (2, settings.TRAIN_NEGATIVES_PER_POSITIVE))
        self.assertTrue((dm.train_ds.negative_item_ids >= 4).all())
        model = make_model(dm)
        batch = next(iter(DataLoader(dm.train_ds, batch_size=2)))
        scores = torch.randn(2, 1 + settings.TRAIN_NEGATIVES_PER_POSITIVE)
        with patch.object(model, 'forward', return_value=scores):
            loss = model.training_step(batch)
        torch.testing.assert_close(loss, F.cross_entropy(scores, torch.zeros(2, dtype=torch.long)))

    def test_non_temporal_data_keeps_ratings_without_inventing_history(self):
        dm = make_datamodule(temporal=False)
        self.assertEqual(len(dm.train_ds), 2)
        self.assertEqual(dm.train_ds.history_items.shape, (2, 0))
        self.assertLess(dm.train_ds.sample_weights[0], dm.train_ds.sample_weights[1])
        with self.assertRaisesRegex(ValueError, "requires sequential history"):
            make_model(dm)

    def test_real_forward_backward_and_evaluation_for_both_architectures(self):
        for arch in (ModelArch.KG_RNN, ModelArch.KG_TRANSFORMER):
            for explicit, temporal in ((True, True), (False, True)):
                with self.subTest(arch=arch, explicit=explicit, temporal=temporal):
                    dm = make_datamodule(explicit=explicit, temporal=temporal)
                    model = make_model(dm, arch)
                    batch = next(iter(DataLoader(dm.train_ds, batch_size=2)))
                    loss = model.training_step(batch)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(grads)
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                    model.validation_step(next(iter(DataLoader(dm.val_ds))))
                    model.test_step(next(iter(DataLoader(dm.test_ds))))
                    metrics = model.test_ranking_metrics.compute()
                    self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))


if __name__ == '__main__':
    unittest.main()
