import json

import numpy as np
import pandas as pd
import pytest
import torch

from edurec import settings
from edurec.datasets import DataAdapter, DataConfig, DatasetName
from edurec.datasets.adapters import graphs, queries
from edurec.datasets.adapters.queries import QueryConfig
from edurec.datasets.cache import ProcessedData
from edurec.datasets.pipeline.build import build_processed
from edurec.datasets.registry import RawData

DATA_DEFAULTS = {
    "test_ratio": 0.2,
    "val_ratio": 0.2,
    "min_interactions": 3,
    "remove_sparse": True,
    "deduplicate": False,
    "random_state": 42,
}


def make_config(**overrides) -> DataConfig:
    return DataConfig(
        dataset=DatasetName.COCO,
        **{**DATA_DEFAULTS, **overrides},
    )


@pytest.fixture
def processed(raw_communities):
    return build_processed(raw_communities, make_config()).data


@pytest.fixture
def adapter(processed):
    return DataAdapter(processed)


@pytest.fixture
def implicit_adapter():
    """Implicit feedback where every user keeps unseen items for negatives."""
    histories = {
        user: [(user + 2 * step) % 12 for step in range(6)] for user in range(6)
    }
    events = pd.DataFrame(
        [
            (user, item, time)
            for user, items in histories.items()
            for time, item in enumerate(items)
        ],
        columns=[settings.USER_COL, settings.ITEM_COL, settings.TIME_COL],
    )
    raw = RawData(
        interactions=events,
        user_features=pd.DataFrame({settings.USER_COL: list(histories)}),
        item_features=pd.DataFrame(
            {settings.ITEM_COL: list(range(12)), "concept": ["a"] * 12}
        ),
        schema={"items": {"graph_list": ["concept"]}},
    )
    return DataAdapter(build_processed(raw, make_config()).data)


def test_processed_data_requires_every_data_field():
    with pytest.raises(TypeError):
        ProcessedData()  # pyright: ignore[reportCallIssue]


def test_bipartite_graph_uses_train_pairs_only(adapter):
    edge_index = adapter.bipartite_graph
    train = adapter.relevant_splits["train"]

    expected = train[[settings.USER_COL, settings.ITEM_COL]].to_numpy(dtype=np.int64).T
    observed = set(map(tuple, edge_index.t().tolist()))

    assert edge_index.shape[0] == 2
    assert np.array_equal(edge_index.numpy(), expected)
    # Every user of the fixture keeps one val and one test event out of TRAIN.
    assert (0, 2) not in observed
    assert (0, 3) not in observed


def test_bipartite_graph_keeps_repeated_events():
    frame = pd.DataFrame(
        {
            settings.USER_COL: [0, 0, 1],
            settings.ITEM_COL: [1, 1, 0],
        }
    )

    assert graphs.build_bipartite_graph(frame).tolist() == [[0, 0, 1], [1, 1, 0]]


def test_query_histories_never_include_target_or_future(adapter):
    config = QueryConfig(max_history=10, random_state=42)
    positives = adapter.relevant_splits
    global_items = [
        int(item)
        for split in ("train", "val", "test")
        for item in positives[split][settings.ITEM_COL]
    ]
    dataset = adapter.get_queries("test", config=config)
    offset = len(positives["train"]) + len(positives["val"])

    for row in range(len(dataset)):
        context = dataset.history_context_index[row]
        valid = dataset.history_valid_mask[row]
        target_index = offset + row
        past = context[valid].tolist()

        assert all(index < target_index for index in past)
        assert target_index not in past
        assert dataset.history_items[row][valid].tolist() == [
            global_items[index] + 1 for index in past
        ]


def test_negative_policy_is_unchanged(implicit_adapter):
    config = QueryConfig(max_history=3, negatives_per_positive=2, random_state=42)
    dataset = implicit_adapter.get_queries("train", config=config)
    positives = implicit_adapter.relevant_splits
    expected = queries.generate_negative_samples(
        interactions=positives["train"],
        item_ids=list(range(implicit_adapter.data.num_items)),
        num_negatives=2,
        random_state=42,
        observed_interactions=pd.concat(positives.values(), ignore_index=True),
    )

    assert torch.equal(
        dataset.negative_item_ids, torch.as_tensor(expected, dtype=torch.long)
    )
    observed_by_user = (
        pd.concat(positives.values(), ignore_index=True)
        .groupby(settings.USER_COL)[settings.ITEM_COL]
        .agg(set)
    )
    for row, user_id in enumerate(dataset.user_ids):
        negatives = dataset.negative_item_ids[row].tolist()
        assert len(set(negatives)) == len(negatives)
        assert not set(negatives).intersection(observed_by_user[int(user_id)])


def test_explicit_feedback_has_no_negatives(adapter):
    dataset = adapter.get_queries("train")

    assert dataset.negative_item_ids.numel() == 0


def test_views_are_memoised_by_content(adapter):
    assert adapter.get_queries("train") is adapter.get_queries("train")
    assert adapter.knowledge_graph is adapter.knowledge_graph
    assert adapter.bipartite_graph is adapter.bipartite_graph
    assert adapter.get_queries(
        "train", config=QueryConfig(max_history=2)
    ) is not adapter.get_queries("train")


def test_unknown_split_is_rejected(adapter):
    with pytest.raises(ValueError, match="split must be one of"):
        adapter.get_queries("nope")


def test_manifest_roundtrip_and_fingerprint(processed, tmp_path):
    folder = tmp_path / "cache"
    processed.save(folder)
    loaded = ProcessedData.load(folder)

    assert loaded.manifest == processed.manifest
    payload = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert payload["dataset"] == "coco"
    assert payload["config"] == processed.manifest.config
    assert payload["split_strategy"] == "temporal"


def test_manifest_rejects_other_data_options(processed):
    expected = make_config(remove_sparse=False).manifest("temporal")

    assert processed.manifest.incompatibility(expected) is not None


def test_manifest_ignores_training_seed_for_temporal_splits(processed):
    expected = make_config(random_state=7).manifest("temporal")

    assert processed.manifest.incompatibility(expected) is None


def test_manifest_checks_seed_for_random_splits():
    first = make_config(random_state=1).manifest("random")
    second = make_config(random_state=2).manifest("random")

    assert first.incompatibility(second) is not None


def test_recbole_general_export_matches_processed_splits(adapter, tmp_path):
    paths = adapter.export_recbole(tmp_path, dataset_name="coco", mode="general")
    train = adapter.relevant_splits["train"]
    rows = pd.read_csv(paths["train.inter"], sep="\t")

    assert paths["inter"].exists()
    assert paths["user"].exists()
    assert paths["item"].exists()
    assert list(rows.columns) == [
        f"{settings.USER_COL}:token",
        f"{settings.ITEM_COL}:token",
    ]
    assert len(rows) == len(train)


def test_recbole_sequential_export_matches_query_histories(adapter, tmp_path):
    config = QueryConfig(max_history=3, random_state=42)
    paths = adapter.export_recbole(
        tmp_path, dataset_name="coco", mode="sequential", config=config
    )
    frame = pd.read_csv(paths["test_seq"], sep="\t", keep_default_na=False)
    dataset = adapter.get_queries("test", config=config)

    assert paths["valid_seq"].exists()
    for row, (items, mask) in enumerate(
        zip(dataset.history_items, dataset.history_valid_mask, strict=True)
    ):
        expected = " ".join(str(int(item) - 1) for item in items[mask].tolist())
        assert frame[f"{settings.ITEM_COL}_list:token_seq"][row] == expected


def test_sequential_export_requires_temporal_history(implicit_adapter, tmp_path):
    raw = ProcessedData(
        train=implicit_adapter.data.train.drop(columns=[settings.TIME_COL]),
        val=implicit_adapter.data.val.drop(columns=[settings.TIME_COL]),
        test=implicit_adapter.data.test.drop(columns=[settings.TIME_COL]),
        u_static_feats=implicit_adapter.data.u_static_feats,
        u_cat_feats=implicit_adapter.data.u_cat_feats,
        i_static_feats=implicit_adapter.data.i_static_feats,
        user_features=implicit_adapter.data.user_features,
        item_features=implicit_adapter.data.item_features,
        schema=implicit_adapter.data.schema,
        id_maps=implicit_adapter.data.id_maps,
        transforms=implicit_adapter.data.transforms,
        graph_node_features=implicit_adapter.data.graph_node_features,
    )

    with pytest.raises(ValueError, match="chronological"):
        DataAdapter(raw).export_recbole(
            tmp_path, dataset_name="coco", mode="sequential"
        )
