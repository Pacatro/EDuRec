"""KGAN equations, query boundaries, and the full baseline runner on CPU."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

from edurec import settings
from edurec.datasets import (
    DataConfig,
    DatasetName,
    ElearningDataModule,
    QueryConfig,
)
from edurec.datasets.preprocessing.build import build_processed
from edurec.datasets.preprocessing.prepare import apply_common_filters
from edurec.evaluation import baseline_train_defaults, get_baseline
from edurec.evaluation.baselines.kgan import (
    KGAN,
    KGANAdapter,
    KGANConfig,
    KGANInput,
    eval_kgan,
)
from edurec.recsys import TrainConfig
from edurec.recsys.configs import resolve_train_config


@pytest.fixture
def data(raw_communities):
    config = DataConfig(
        dataset=DatasetName.COCO,
        min_interactions=3,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=42,
    )
    return build_processed(
        apply_common_filters(raw_communities, config),
        config,
    ).data


@pytest.fixture
def adapter(data):
    return KGANAdapter(
        data,
        QueryConfig(max_history=10, random_state=42),
        KGANConfig(emb_dim=4, memory_size=3, candidate_chunk_size=2),
    )


def make_model(adapter):
    return KGAN(
        adapter.data.num_items,
        adapter.num_entities,
        len(adapter.relation_types),
        adapter.model_config,
        TrainConfig(topks=[1, 2]),
        val_topk=1,
    )


def test_default_baseline_is_available():
    assert get_baseline("KGAN") is eval_kgan
    assert "KGAN" in settings.BASELINES
    spec = importlib.util.spec_from_file_location("eval_cli", "scripts/eval.py")
    assert spec is not None and spec.loader is not None
    # The argparse script imports its adjacent common module.
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        labels = module._selected_baselines(["all"])
        assert "KGAN" in labels and "LightGCN" in labels
        assert labels.count("KGAN") == 1
        assert module._target_models("EDuRec", labels, only_baselines=True) == labels
    finally:
        sys.path.pop(0)


def test_reference_hyperparameters_and_cli_overrides():
    defaults = baseline_train_defaults("KGAN")
    assert defaults is not None
    assert (defaults.epochs, defaults.lr, defaults.batch_size) == (20, 0.001, 1024)
    assert defaults.weight_decay == 0
    assert baseline_train_defaults("KGAN") is not defaults
    assert baseline_train_defaults("LightGCN") is None
    assert KGANConfig().emb_dim == 16
    overridden = resolve_train_config(
        {"epochs": 3, "batch_size": 8, "lr": 0.01}, defaults=defaults
    )
    assert (overridden.epochs, overridden.batch_size, overridden.lr) == (3, 8, 0.01)


def test_aggregate_all_hops_switch(adapter):
    cfg = replace(adapter.model_config, using_all_hops=True)
    model = KGAN(
        adapter.data.num_items,
        adapter.num_entities,
        len(adapter.relation_types),
        cfg,
        TrainConfig(topks=[1]),
        val_topk=1,
    )
    batch = next(iter(DataLoader(adapter.dataset("test"), batch_size=2)))
    # Identical memories and attention parameters in both hops make the sum
    # exactly twice the final-hop representation.
    model.group_attention[1].load_state_dict(model.group_attention[0].state_dict())
    batch = batch._replace(
        heads=batch.heads[:, :1].expand_as(batch.heads),
        tails=batch.tails[:, :1].expand_as(batch.tails),
        valid_mask=batch.valid_mask[:, :1].expand_as(batch.valid_mask),
    )
    all_hops = model(batch)
    model.cfg = replace(cfg, using_all_hops=False)
    torch.testing.assert_close(all_hops, 2 * model(batch))


def test_temporal_memories_use_only_query_history(adapter):
    train = adapter.dataset("train")
    assert not train[0].valid_mask.any()
    for split in ("train", "val", "test"):
        dataset = adapter.dataset(split)
        for index in range(len(dataset)):
            sample = dataset[index]
            query = sample.query
            seeds = set((query.history_items[query.history_valid_mask] - 1).tolist())
            assert set(sample.heads[0][sample.valid_mask[0]].tolist()) <= seeds
            # Every memory is a real directed edge, within its relation group.
            for hop in range(adapter.model_config.n_hops):
                for relation in range(len(adapter.relation_types)):
                    for head, tail, valid in zip(
                        sample.heads[hop, relation],
                        sample.tails[hop, relation],
                        sample.valid_mask[hop, relation],
                    ):
                        if valid:
                            assert (relation, int(tail)) in adapter.adjacency[int(head)]
            if split != "train":
                changed = query._replace(
                    target_item_id=(query.target_item_id + 3) % adapter.data.num_items
                )
                other = adapter.evaluation_input(changed)
                assert torch.equal(sample.heads, other.heads)
                assert torch.equal(sample.tails, other.tails)


def test_unordered_memories_use_train_only_and_exclude_train_target(data):
    data = replace(
        data,
        **{
            name: getattr(data, name).drop(columns=[settings.TIME_COL])
            for name in ("train", "val", "test")
        },
    )
    adapter = KGANAdapter(data, QueryConfig(random_state=42), KGANConfig(memory_size=8))
    for split in ("train", "val", "test"):
        sample = adapter.dataset(split)[0]
        seeds = set(adapter.train_items[int(sample.query.user_id)])
        if split == "train":
            seeds.discard(int(sample.query.target_item_id))
        assert set(sample.heads[0][sample.valid_mask[0]].tolist()) == seeds


def test_sampling_is_reproducible_and_pads_empty_groups(adapter):
    first = adapter.sample_memories([0])
    torch.manual_seed(998)
    second = adapter.sample_memories([0])
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    heads, tails, mask = first
    assert (heads[~mask] == adapter.num_entities).all()
    assert (tails[~mask] == adapter.num_entities).all()
    assert mask[0].any() and not mask[0].all()
    # Reverse edges allow a second hop from an attribute back to courses.
    assert mask[1].any()


def test_batched_evaluation_adapter_matches_dataset(adapter):
    query = next(iter(DataLoader(adapter.evaluation_queries(), batch_size=3)))
    expected = next(iter(DataLoader(adapter.dataset("test"), batch_size=3)))
    actual = adapter.evaluation_input(query)
    assert actual.query is query
    assert torch.equal(actual.heads, expected.heads)
    assert torch.equal(actual.tails, expected.tails)
    assert torch.equal(actual.valid_mask, expected.valid_mask)


def test_full_catalog_candidate_scores_empty_history_and_backward(adapter):
    batch = next(iter(DataLoader(adapter.dataset("train"), batch_size=4)))
    model = make_model(adapter)
    model.log = lambda *args, **kwargs: None
    scores = model(batch)
    assert scores.shape == (4, adapter.data.num_items)
    assert torch.isfinite(scores).all()
    assert torch.equal(scores[0], torch.zeros_like(scores[0]))
    ids = torch.tensor([[3, 0], [2, 7], [0, 1], [1, 5]])
    torch.testing.assert_close(model(batch, ids), scores.gather(1, ids))
    loss = model.training_step(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.entities.weight.grad is not None
    assert model.relations.weight.grad is not None
    assert torch.isfinite(model.entities.weight.grad).all()
    assert model.entities.weight.grad.abs().sum() > 0
    assert (model.entities.weight.grad[model.num_entities] == 0).all()
    # Perturbing masked slots has no effect, even when they contain real IDs.
    modified = batch._replace(
        heads=batch.heads.masked_fill(~batch.valid_mask, 0),
        tails=batch.tails.masked_fill(~batch.valid_mask, 1),
    )
    torch.testing.assert_close(scores, model(modified))
    torch.testing.assert_close(
        model.auxiliary_loss(batch), model.auxiliary_loss(modified)
    )
    assert torch.isfinite(model.half()(batch)).all()


def test_two_level_attention_matches_equations(adapter):
    cfg = KGANConfig(emb_dim=2, n_hops=1, memory_size=2)
    model = KGAN(3, 5, 2, cfg, TrainConfig(topks=[1]), val_topk=1)
    # Force equal inter-group weights while preserving candidate-conditioned
    # intra-group attention; relation 1 is empty and must get zero probability.
    with torch.no_grad():
        model.entities.weight.copy_(
            torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0], [0.0, 0.0]]
            )
        )
        model.relations.weight.fill_(1)
        for parameter in model.group_attention.parameters():
            parameter.zero_()
    query = next(iter(DataLoader(adapter.get_queries("test"), batch_size=1)))
    batch = KGANInput(
        query,
        torch.tensor([[[[0, 1], [5, 5]]]]),
        torch.tensor([[[[3, 4], [5, 5]]]]),
        torch.tensor([[[[True, True], [False, False]]]]),
    )
    scores = model(batch)
    candidates = model.entities.weight[:3]
    intra = candidates.softmax(-1)
    user = intra @ model.entities.weight[3:5]
    expected = (candidates * user).sum(-1)[None]
    torch.testing.assert_close(scores, expected)
    # Two valid triples with score 2 and squared norms 1 + 4 + 2.
    expected_aux = -cfg.kge_weight * torch.tensor(2.0).sigmoid() + cfg.l2_weight * 7
    torch.testing.assert_close(model.auxiliary_loss(batch), expected_aux)


def test_model_view_and_runner_roundtrip(data, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "NUM_WORKERS", 0)
    monkeypatch.setitem(settings.state, "device", "cpu")
    monkeypatch.setitem(settings.state, "random_state", 42)
    monkeypatch.chdir(tmp_path)
    dm = ElearningDataModule(
        DataConfig(dataset=DatasetName.COCO, random_state=42),
        batch_size=4,
        max_history=10,
    )
    dm._data = data
    original = dm.adapter
    view = dm.with_adapter(KGANAdapter, batch_size=2)
    assert view.data is data and view.adapter is not original
    assert dm.batch_size == 4 and view.batch_size == 2
    cfg = KGANConfig(emb_dim=4, memory_size=2, candidate_chunk_size=3)
    result = eval_kgan(
        dm,
        TrainConfig(epochs=1, patience=1, batch_size=4, topks=[1, 2]),
        val_topk=1,
        compile=False,
        cfg=cfg,
    )
    assert result.model.tolist() == ["KGAN"]
    assert {"ndcg@1", "recall@2", "training_time_s", "inference_time_s"} <= set(result)
    assert result.select_dtypes("number").notna().all().all()
    assert dm.adapter is original
    checkpoint = next(tmp_path.rglob("*.ckpt"))
    model = KGAN.load_from_checkpoint(checkpoint, weights_only=False)
    assert isinstance(model, L.LightningModule)
    assert model.cfg == cfg
    view.setup("test")
    batch = next(iter(view.test_dataloader()))
    assert torch.isfinite(model(batch)).all()


def test_kgan_rejects_graph_without_relations(data):
    data = replace(data, schema={"items": {}})
    with pytest.raises(ValueError, match="requires a knowledge graph"):
        KGANAdapter(data).prepare()
