"""Model-specific samples must preserve canonical evaluation over the same processed data."""

from dataclasses import replace

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from edurec import settings
from edurec.datasets import (
    DataAdapter,
    DataConfig,
    DatasetName,
    ElearningDataModule,
    QueryConfig,
    RecSysQuery,
)
from edurec.datasets.preprocessing.build import build_processed
from edurec.datasets.preprocessing.prepare import apply_common_filters
from edurec.evaluation.baselines.recbole import RecBoleAdapter
from edurec.recsys import EdurecDataAdapter


@pytest.fixture
def processed(raw_communities):
    config = DataConfig(
        dataset=DatasetName.COCO,
        min_interactions=3,
        test_ratio=0.2,
        val_ratio=0.2,
        random_state=42,
    )
    return build_processed(apply_common_filters(raw_communities, config), config).data


class PairAdapter(DataAdapter[tuple[torch.Tensor, ...], torch.Tensor]):
    """An unrelated training format with a custom batch collation."""

    def prepare(self, stage=None):
        self.stage = stage

    def dataset(self, split):
        pairs = self.relevant_splits[split][
            [settings.USER_COL, settings.ITEM_COL]
        ].to_numpy(copy=True)
        return TensorDataset(torch.as_tensor(pairs))

    @staticmethod
    def collate_fn(samples):
        return {"pairs": torch.stack([sample[0] for sample in samples])}

    def evaluation_input(self, query):
        return query.user_id


def test_proposed_adapter_builds_and_caches_its_structures(processed):
    config = QueryConfig(max_history=3)
    adapter = EdurecDataAdapter(processed, config)
    assert isinstance(adapter, EdurecDataAdapter)
    assert adapter.dataset("train") is adapter.get_queries("train", config)
    assert "knowledge_graph" not in adapter.__dict__
    assert adapter.knowledge_graph is adapter.knowledge_graph
    assert adapter.data is processed
    assert adapter.metadata.num_items == processed.num_items
    assert adapter.metadata.num_users == processed.num_users
    query = adapter.dataset("test")[0]
    assert adapter.evaluation_input(query) is query


def test_custom_adapter_loaders_do_not_change_evaluation(processed, monkeypatch):
    monkeypatch.setattr(settings, "NUM_WORKERS", 0)
    dm = ElearningDataModule(
        DataConfig(dataset=DatasetName.COCO, random_state=42),
        batch_size=3,
        adapter_factory=PairAdapter,
    )
    dm._data = processed
    dm.setup("fit")
    first_adapter = dm.adapter
    assert isinstance(first_adapter, PairAdapter)
    assert first_adapter.stage == "fit"
    batch = next(iter(dm.train_dataloader()))
    assert batch["pairs"].shape == (3, 2)
    assert not hasattr(dm.adapter, "knowledge_graph")
    assert dm.adapter._queries == {}
    assert "interaction_features" not in dm.adapter.__dict__

    for stage, loader in [
        ("validate", dm.val_dataloader),
        ("test", dm.test_dataloader),
        ("predict", dm.predict_dataloader),
    ]:
        dm.setup(stage)
        assert "pairs" in next(iter(loader()))
    assert dm.adapter is first_adapter
    query = next(iter(dm.evaluation_dataloader()))
    assert isinstance(query, RecSysQuery)
    expected = dm.adapter.get_queries("test", dm.query_config)
    assert query.target_item_id.tolist() == expected.target_item_ids[:3].tolist()


def test_model_adapter_contract_is_abstract(processed):
    with pytest.raises(TypeError):
        DataAdapter(processed, QueryConfig())


def test_sequential_adapter_requires_temporal_data(processed):
    data = replace(
        processed,
        train=processed.train.drop(columns=[settings.TIME_COL]),
        val=processed.val.drop(columns=[settings.TIME_COL]),
        test=processed.test.drop(columns=[settings.TIME_COL]),
    )
    with pytest.raises(ValueError, match="chronological"):
        RecBoleAdapter(data, QueryConfig(), sequential=True)


class PermutedCatalog:
    def __init__(self, data):
        self.users = torch.arange(data.num_users, 0, -1)
        self.items = torch.arange(data.num_items, 0, -1)

    def token2id(self, field, tokens):
        lookup = self.users if field == settings.USER_COL else self.items
        return lookup[[int(token) for token in tokens]].numpy()

    def join(self, interaction):
        return interaction


@pytest.mark.parametrize("sequential", [False, True])
def test_recbole_inputs_and_scores_preserve_canonical_ids(processed, sequential):
    adapter = RecBoleAdapter(
        processed, QueryConfig(max_history=3), sequential=sequential
    )
    query = next(iter(DataLoader(adapter.evaluation_queries(), batch_size=4)))
    with pytest.raises(RuntimeError, match="bind_catalog"):
        adapter.evaluation_input(query)
    catalog = PermutedCatalog(processed)
    adapter.bind_catalog(catalog, torch.device("cpu"))
    inputs = adapter.evaluation_input(query)
    assert torch.equal(inputs[settings.USER_COL], catalog.users[query.user_id])
    assert settings.ITEM_COL not in inputs.interaction  # no target passed as input
    if sequential:
        history = inputs[f"{settings.ITEM_COL}_list"]
        mask = query.history_valid_mask
        assert (history[~mask] == 0).all()
        assert torch.equal(history[mask], catalog.items[query.history_items[mask] - 1])
        assert torch.equal(inputs["item_length"], mask.sum(dim=1))
    scores = torch.arange(processed.num_items + 1).repeat(4, 1)
    assert torch.equal(adapter.canonical_scores(scores), scores[:, catalog.items])


@pytest.mark.parametrize("model,sequential", [("BPR", False), ("SASRec", True)])
def test_recbole_factory_config_and_export(processed, tmp_path, model, sequential):
    adapter = RecBoleAdapter.for_model(model, processed, QueryConfig(max_history=3))
    assert isinstance(adapter, RecBoleAdapter)
    assert adapter.sequential == sequential
    base = {"epochs": 2}
    config = adapter.configure(model, base)
    assert base == {"epochs": 2}
    assert config["epochs"] == 2
    if sequential:
        assert config["MAX_ITEM_LIST_LENGTH"] == 3
    paths = adapter.export(tmp_path / "atomic")
    assert all(path.exists() for path in paths.values())
    for split in ("train", "val", "test"):
        pd.testing.assert_frame_equal(
            adapter.relevant_splits[split], processed.positive_splits()[split]
        )


def test_recbole_export_supports_legacy_cache_without_manifest(processed, tmp_path):
    legacy = replace(processed, manifest=None)
    adapter = RecBoleAdapter(legacy, QueryConfig())
    paths = adapter.export(tmp_path / "atomic", dataset_name="legacy")
    assert all(path.exists() for path in paths.values())
    assert (tmp_path / "atomic" / "legacy.inter").exists()


def test_recbole_runner_uses_custom_adapter_factory(processed, monkeypatch):
    from edurec.evaluation.baselines import recbole as runner

    class CustomBPRAdapter(RecBoleAdapter):
        def configure(self, model, base):
            return {**super().configure(model, base), "embedding_size": 4}

    monkeypatch.setitem(settings.state, "device", "cpu")
    monkeypatch.setattr(settings, "NUM_WORKERS", 0)
    dm = ElearningDataModule(DataConfig(dataset=DatasetName.COCO), batch_size=4)
    dm._data = processed
    instances = []

    def make_adapter(model: str) -> CustomBPRAdapter:
        assert model == "BPR"
        return CustomBPRAdapter(processed, dm.query_config)

    def run_model(**kwargs):
        adapter = kwargs["adapter"]
        assert isinstance(adapter, CustomBPRAdapter)
        assert kwargs["config_dict"]["embedding_size"] == 4
        assert (dm.atomic_folder / "coco.inter").exists()
        assert adapter.data is dm.data
        instances.append(adapter)
        return {"ndcg@1": 0.5}, 0.1, 0.01

    monkeypatch.setattr(runner, "_run_model", run_model)
    result = runner.eval_recbole_baselines(
        ["BPR", "BPR"], dm, topks=[1], adapter_factory=make_adapter
    )
    assert result["ndcg@1"].tolist() == [0.5, 0.5]
    assert instances[0] is not instances[1]
    assert dm.require_adapter(EdurecDataAdapter) is dm.adapter
    with pytest.raises(TypeError, match="requires PairAdapter"):
        dm.require_adapter(PairAdapter)
