import json
from pathlib import Path

import pytest

from edurec.recsys.io import load_model
from edurec.recsys.model import GhostConfig


def test_load_model_returns_latest_saved_artifacts(tmp_path: Path):
    models_root = tmp_path / "models" / "mars"
    older_dir = models_root / "Ghost_20240101_000000"
    newer_dir = models_root / "Ghost_20240102_000000"
    older_dir.mkdir(parents=True)
    newer_dir.mkdir(parents=True)

    cfg = GhostConfig(num_users=2, num_items=3, num_ctx_feats=4)
    older_file = older_dir / f"{older_dir.name}.pt"
    newer_file = newer_dir / f"{newer_dir.name}.pt"
    older_cfg = older_dir / f"{older_dir.name}.json"
    newer_cfg = newer_dir / f"{newer_dir.name}.json"

    older_file.write_bytes(b"old")
    newer_file.write_bytes(b"new")
    older_cfg.write_text(json.dumps(cfg.__dict__), encoding="utf-8")
    newer_cfg.write_text(json.dumps(cfg.__dict__), encoding="utf-8")

    model_file, loaded_cfg = load_model(tmp_path / "models", "mars")

    assert model_file == newer_file
    assert loaded_cfg == cfg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
