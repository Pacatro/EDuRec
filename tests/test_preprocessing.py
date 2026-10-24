from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from edurec import settings
from edurec.datasets import DataAdapter, DataConfig, DatasetName
from edurec.datasets.adapters.queries import QueryConfig
from edurec.datasets.cache import ProcessedData
from edurec.datasets.features import text
from edurec.datasets.pipeline.build import build_processed
from edurec.recsys import RecSys
from edurec.recsys.configs import ModelArch, TrainConfig
from scripts.common import build_config


@pytest.fixture
def fake_embedding_model(monkeypatch):
    class FakeModel:
        max_seq_length = 0

        def __init__(self):
            self.documents = []

        def encode(self, docs, **kwargs):
            self.documents.extend(docs)
            return np.tile([3.0, 4.0], (len(docs), settings.TEXT_EMBEDDING_DIM // 2))

    model = FakeModel()
    monkeypatch.setattr(text, "_get_sentence_embedding_model", lambda name: model)
    return model


def test_text_encoding_uses_one_policy_for_empty_docs_and_prefixes(
    fake_embedding_model,
):
    vectors = text.encode_documents(["name: python", "", "name: graphs"])

    assert vectors.dtype == np.float32
    assert not vectors[1].any()
    np.testing.assert_allclose(np.linalg.norm(vectors[[0, 2]], axis=1), 1.0)
    assert fake_embedding_model.documents == [
        "query: name: python",
        "query: name: graphs",
    ]
    assert fake_embedding_model.max_seq_length == settings.TEXT_MAX_TOKENS


def test_empty_text_does_not_load_a_model(monkeypatch):
    def unexpected_call(name):
        raise AssertionError("Empty documents must not load pretrained weights")

    monkeypatch.setattr(text, "_get_sentence_embedding_model", unexpected_call)
    assert not text.encode_documents(["", ""]).any()
    assert text.encode_documents([]).shape == (0, settings.TEXT_EMBEDDING_DIM)


@pytest.mark.parametrize("arch", [ModelArch.KG_RNN, ModelArch.SASREC_TEXT])
def test_raw_features_reach_graph_torch_queries_and_models(
    make_raw, model_config, fake_embedding_model, tmp_path, arch
):
    users = ["user-z", "user-a", "user-m", "user-b"]
    items = [900, 100, 700, 300]
    raw = make_raw(
        {user: items for user in users},
        {item: ["python", None, "python"] for item in items},
    )
    raw.user_features["age"] = [20.0, 30.0, 40.0, 50.0]
    raw.user_features["group"] = ["a", "b", None, "a"]
    raw.item_features["duration"] = [10.0, 20.0, 30.0, 40.0]
    raw.item_features["title"] = ["<b>Python</b>", None, "Graphs", "Ranking"]
    raw.item_features["level"] = ["beginner", "advanced", "beginner", "advanced"]
    raw.interactions["progress"] = raw.interactions[settings.TIME_COL] * 10
    raw.interactions["device"] = ["train", "train", "new-val", "new-test"] * 4
    raw.schema["users"] = {"num": ["age"], "cat": ["group"]}
    raw.schema["items"].update(
        num=["duration"],
        cat=["level"],
        text=["title"],
        graph_text={"concept": ["name"]},
    )
    raw.schema["inter"] = {"num": ["progress"], "cat": ["device"]}
    config = DataConfig(
        dataset=DatasetName.COCO,
        min_interactions=3,
        random_state=42,
    )
    result = build_processed(raw, config)
    data = result.data
    assert data.id_maps["items"] == dict(zip(items, range(4)))
    assert data.user_features[settings.USER_COL].tolist() == list(range(4))
    assert data.i_static_feats.shape == (4, 1 + settings.TEXT_EMBEDDING_DIM)
    assert data.u_static_feats[:, 0].tolist() == pytest.approx([0, 1 / 3, 2 / 3, 1])
    assert not data.i_static_feats[1, 1:].any()
    assert data.val["num__progress"].tolist() == [2.0] * 4
    assert data.test["cat__device"].tolist() == [-1] * 4
    assert data.item_features["concept"].tolist() == [["python"]] * 4
    assert result.data.reports["split_report"]["test"]["targets_unseen_in_train"] == 4

    data.save(tmp_path / "cache")
    loaded = ProcessedData.load(tmp_path / "cache")
    pd.testing.assert_frame_equal(loaded.test, data.test)
    assert loaded.id_maps == data.id_maps
    assert loaded.schema == data.schema
    inter_transformer = loaded.transforms["inter"]
    assert inter_transformer is not None
    assert inter_transformer.named_transformers_["num"].named_steps[
        "scaler"
    ].data_max_.tolist() == [10.0]

    adapter = DataAdapter(loaded)
    graph = adapter.knowledge_graph
    assert graph["attr::concept"].node_ids == ["python"]
    assert graph["attr::concept"].x.shape == (1, settings.TEXT_EMBEDDING_DIM)
    pd.testing.assert_frame_equal(
        loaded.graph_node_features["concept"], data.graph_node_features["concept"]
    )
    assert graph["item", "concept", "attr::concept"].edge_index.tolist() == [
        list(range(4)),
        [0] * 4,
    ]

    dataset = adapter.get_queries("test", QueryConfig(max_history=4, random_state=42))
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    assert batch.history_items[0].tolist() == [1, 2, 3, 0]
    assert batch.history_dense_features[0, :3, 1].tolist() == [0.0, 1.0, 2.0]
    assert batch.history_cat_features[0, :3, 0].tolist() == [0, 0, -1]
    assert batch.target_item_id.tolist() == [3] * 4

    # Exercise the same dimension/cardinality inference used by the CLI.
    from edurec.datasets import ElearningDataModule

    dm = ElearningDataModule(config=config, batch_size=4)
    dm._data = loaded
    dm.train_ds = adapter.get_queries("train")
    cfg = build_config(dm, base=replace(model_config, arch=arch))
    module = RecSys(
        cfg=cfg,
        knowledge_graph=graph,
        i_static_feats=loaded.i_static_feats,
        u_static_feats=loaded.u_static_feats,
        u_cat_feats=loaded.u_cat_feats,
        train_cfg=TrainConfig(topks=[1]),
        val_topk=1,
    )
    scores = module(batch)
    assert scores.shape == (4, 4)
    assert torch.isfinite(scores).all()
    scores.square().mean().backward()
    assert module.model.item_embeddings().requires_grad

    paths = adapter.export_recbole(
        tmp_path / "recbole", dataset_name="fixture", mode="sequential"
    )
    exported = pd.read_csv(paths["item"], sep="\t")
    assert exported[f"{settings.ITEM_COL}:token"].tolist() == list(range(4))
    assert len(pd.read_csv(paths["test_seq"], sep="\t")) == 4
