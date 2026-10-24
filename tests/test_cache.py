import pandas as pd
import pytest
import torch

from edurec import settings
from edurec.datasets import (
    CommunityConfig,
    DataConfig,
    DatasetName,
    ElearningDataModule,
    datamodule,
)
from edurec.datasets import cache as cache_module
from edurec.datasets.cache import ProcessedData

COMMUNITY_KEYS = (
    "max_users",
    "max_interactions",
    "max_items",
    "seed",
    "community_fields",
    "min_history_retention",
)


def _unexpected_call(*args, **kwargs):
    raise AssertionError("raw data should not be read or downloaded")


@pytest.fixture
def cache_dirs(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    atomic = tmp_path / "atomicfiles"
    monkeypatch.setattr(settings, "PROCESSED_FOLDER", str(processed))
    monkeypatch.setattr(settings, "ATOMICFILES_FOLDER", str(atomic))
    return processed / "coco", atomic / "coco"


@pytest.fixture
def make_datamodule(raw_communities, monkeypatch, cache_dirs):
    def build(**overrides):
        monkeypatch.setattr(
            datamodule, "load_raw_data", lambda dataset: raw_communities
        )
        use_processed_data = overrides.pop("use_processed_data", False)
        community = CommunityConfig(
            **{key: overrides.pop(key) for key in COMMUNITY_KEYS if key in overrides}
        )
        params = {
            "test_ratio": 0.2,
            "val_ratio": 0.2,
            "random_state": 42,
        }
        config = DataConfig(
            dataset=DatasetName.COCO,
            community=community,
            **{**params, **overrides},
        )
        return ElearningDataModule(
            config=config,
            batch_size=8,
            use_processed_data=use_processed_data,
        )

    return build


def test_setup_writes_processed_cache(make_datamodule, cache_dirs):
    processed, _ = cache_dirs
    make_datamodule().setup()

    assert (processed / "train.feather").exists()
    assert (processed / "static_feats.safetensors").exists()
    assert (processed / "report.json").exists()
    assert (processed / "manifest.json").exists()


def test_loaded_cache_matches_built_artifacts(make_datamodule, cache_dirs):
    processed, _ = cache_dirs
    built = make_datamodule()
    built.setup()
    loaded = ProcessedData.load(processed)

    for split, frame in built.data.splits().items():
        pd.testing.assert_frame_equal(loaded.splits()[split], frame)
    pd.testing.assert_frame_equal(loaded.user_features, built.data.user_features)
    pd.testing.assert_frame_equal(loaded.item_features, built.data.item_features)
    for name in ("u_static_feats", "u_cat_feats", "i_static_feats"):
        assert torch.equal(getattr(loaded, name), getattr(built.data, name))
    assert loaded.reports == built.data.reports


def test_use_processed_loads_cache_without_reading_raw_data(
    make_datamodule, monkeypatch
):
    built = make_datamodule()
    built.setup()
    monkeypatch.setattr(datamodule, "load_raw_data", _unexpected_call)

    cached = make_datamodule(use_processed_data=True)
    cached.setup()

    assert cached.data.num_interactions == built.data.num_interactions
    assert cached.data.num_items == built.data.num_items
    assert cached.data.reports == built.data.reports


def test_load_accepts_legacy_cache_without_report(make_datamodule, cache_dirs):
    processed, _ = cache_dirs
    built = make_datamodule()
    built.setup()
    (processed / "report.json").unlink()

    loaded = ProcessedData.load(processed)

    assert loaded.reports == {}
    pd.testing.assert_frame_equal(loaded.train, built.data.train)


def test_use_processed_accepts_legacy_cache_without_report(
    make_datamodule, cache_dirs, monkeypatch
):
    processed, _ = cache_dirs
    built = make_datamodule()
    built.setup()
    (processed / "report.json").unlink()
    monkeypatch.setattr(datamodule, "load_raw_data", _unexpected_call)

    dm = make_datamodule(use_processed_data=True)
    dm.setup()

    assert dm.data.num_interactions == built.data.num_interactions
    assert dm.data.reports == {}


def test_incompatible_cache_raises_actionable_error(
    make_datamodule, cache_dirs, monkeypatch
):
    processed, _ = cache_dirs
    make_datamodule().setup()
    (processed / "processor.joblib").write_bytes(b"not a pickle")
    monkeypatch.setattr(datamodule, "load_raw_data", _unexpected_call)

    dm = make_datamodule(use_processed_data=True)
    with pytest.raises(RuntimeError, match="incompatible"):
        dm.setup()


def test_use_processed_builds_when_cache_is_missing(make_datamodule, cache_dirs):
    processed, _ = cache_dirs
    dm = make_datamodule(use_processed_data=True)
    dm.setup()

    assert dm.data.num_interactions > 0
    assert (processed / "train.feather").exists()


def test_prepare_data_downloads_only_when_cache_is_missing(
    make_datamodule, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        datamodule, "download_raw_data", lambda dataset: calls.append(dataset)
    )
    make_datamodule(use_processed_data=True).prepare_data()
    assert calls == [DatasetName.COCO]

    make_datamodule().setup()
    calls.clear()
    make_datamodule(use_processed_data=True).prepare_data()
    assert calls == []


def test_subset_build_replaces_previous_cache(make_datamodule, cache_dirs):
    processed, _ = cache_dirs
    make_datamodule().setup()
    make_datamodule(max_items=3).setup()

    assert ProcessedData.load(processed).i_static_feats.shape[0] == 3


def test_save_replaces_folder_without_leftovers(make_datamodule, tmp_path):
    dm = make_datamodule()
    dm.setup()
    folder = tmp_path / "saved"
    folder.mkdir()
    (folder / "stale.feather").write_text("old")

    dm.data.save(folder)

    assert not (folder / "stale.feather").exists()
    assert not (tmp_path / "saved.tmp").exists()
    assert ProcessedData.load(folder).train is not None


def test_save_removes_leftover_temporary_folder(make_datamodule, tmp_path):
    dm = make_datamodule()
    dm.setup()
    leftover = tmp_path / "saved.tmp"
    leftover.mkdir()
    (leftover / "junk").write_text("x")

    dm.data.save(tmp_path / "saved")

    assert not leftover.exists()


def test_interrupted_save_keeps_previous_cache(make_datamodule, tmp_path, monkeypatch):
    full = make_datamodule()
    full.setup()
    folder = tmp_path / "saved"
    full.data.save(folder)
    previous = (folder / "train.feather").read_bytes()

    subset = make_datamodule(max_items=3)
    subset.setup()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cache_module, "save_file", fail)
    with pytest.raises(OSError):
        subset.data.save(folder)

    assert (folder / "train.feather").read_bytes() == previous
    assert ProcessedData.load(folder).i_static_feats.shape[0] == full.data.num_items
    assert not (tmp_path / "saved.tmp").exists()


def test_atomic_files_are_replaced_when_written(make_datamodule, cache_dirs):
    _, atomic = cache_dirs
    atomic.mkdir(parents=True)
    stale = atomic / "coco.old.inter"
    stale.write_text("x")

    dm = make_datamodule()
    dm.setup()
    dm.export_recbole()

    assert not stale.exists()
    assert (atomic / "coco.inter").exists()


def test_use_processed_rejects_incompatible_options(
    make_datamodule, cache_dirs, monkeypatch
):
    make_datamodule().setup()
    monkeypatch.setattr(datamodule, "load_raw_data", _unexpected_call)

    dm = make_datamodule(use_processed_data=True, max_items=3)
    with pytest.raises(RuntimeError, match="different data options"):
        dm.setup()


def test_use_processed_warns_on_cache_without_manifest(
    make_datamodule, cache_dirs, monkeypatch
):
    processed, _ = cache_dirs
    built = make_datamodule()
    built.setup()
    (processed / "manifest.json").unlink()
    monkeypatch.setattr(datamodule, "load_raw_data", _unexpected_call)
    dm = make_datamodule(use_processed_data=True)

    with pytest.warns(UserWarning, match="manifest"):
        dm.setup()

    assert dm.data.num_interactions == built.data.num_interactions


def test_setup_test_only_creates_test_queries(make_datamodule):
    dm = make_datamodule()
    dm.setup("test")

    assert hasattr(dm, "test_ds")
    assert not hasattr(dm, "train_ds")


def test_repeated_setup_reuses_views(make_datamodule):
    dm = make_datamodule()
    dm.setup("fit")
    train_ds = dm.train_ds
    dm.setup("test")

    assert dm.train_ds is train_ds
    assert dm.test_ds is dm.adapter.get_queries("test", config=dm.query_config)


def test_dataloader_batches_preserve_query_contract(make_datamodule, monkeypatch):
    monkeypatch.setattr(settings, "NUM_WORKERS", 0)
    dm = make_datamodule()
    dm.setup("fit")

    batch = next(iter(dm.train_dataloader()))

    assert batch.history_items.shape == batch.history_valid_mask.shape
    assert batch.history_items.shape[0] == batch.target_item_id.numel()
    assert batch.history_dense_features.shape[:2] == batch.history_items.shape
    assert batch.target_item_id.max().item() < dm.data.num_items
