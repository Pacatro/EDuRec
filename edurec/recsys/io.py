import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .ghost import GhostConfig
from .two_tower import RetrievalConfig
from ..datasets import Phase
from .. import settings

type RecsysConfig = GhostConfig | RetrievalConfig


def save_model(
    model_config: RecsysConfig,
    dataset_name: str,
    best_model_path: str | Path,
    models_folder: str | Path,
    metrics: dict[str, float] | None = None,
) -> tuple[Path, Path, Path | None]:
    """Save the best model and its config to the expected folder structure."""
    if not is_dataclass(model_config):
        raise TypeError("model_config must be a dataclass instance.")

    if isinstance(model_config, GhostConfig):
        phase = Phase.RANKING
    elif isinstance(model_config, RetrievalConfig):
        phase = Phase.RETRIEVAL
    else:
        raise TypeError(f"Unsupported model config type: {type(model_config)!r}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_root = Path(models_folder)
    models_root.mkdir(parents=True, exist_ok=True)
    model_folder = models_root / dataset_name / phase.value / timestamp
    model_folder.mkdir(parents=True, exist_ok=True)

    model_file_path = model_folder / settings.MODEL_FILENAME
    model_config_path = model_folder / settings.MODEL_METADATA_FILENAME

    Path(best_model_path).rename(model_file_path)
    model_config_path.write_text(
        json.dumps(
            {"phase": phase.value, "config": asdict(model_config)},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    metrics_path = save_metrics(metrics, model_folder) if metrics is not None else None
    return model_file_path, model_config_path, metrics_path


def save_metrics(metrics: dict[str, float], saving_models_folder: str | Path) -> Path:
    file_path = Path(saving_models_folder) / settings.METRICS_FILENAME
    pd.DataFrame.from_dict(metrics, orient="index").to_csv(file_path, index=True)
    return file_path


def load_model(
    models_folder: str | Path,
    dataset_name: str,
    phase: Phase | str | None = None,
) -> tuple[Path, RecsysConfig]:
    """Load the most recent saved model and rebuild its config."""
    root = Path(models_folder) / dataset_name

    if not root.exists():
        raise FileNotFoundError(f"Models folder {root} does not exist")

    if not root.is_dir():
        raise NotADirectoryError(f"Models folder is not a directory: {root}")

    requested_type = Phase(phase) if phase is not None else None
    search_roots = (
        [root / requested_type.value]
        if requested_type is not None
        else [root / phase.value for phase in Phase]
    )
    model_dirs = [
        path
        for phase_root in search_roots
        if phase_root.exists()
        for path in phase_root.iterdir()
        if path.is_dir()
        and (path / settings.MODEL_FILENAME).exists()
        and (path / settings.MODEL_METADATA_FILENAME).exists()
    ]

    if not model_dirs:
        if requested_type is None:
            raise FileNotFoundError(f"No models found in {root}")
        raise FileNotFoundError(f"No models of type {requested_type!r} found in {root}")

    latest_dir = max(model_dirs, key=lambda path: path.stat().st_mtime)
    model_file = latest_dir / settings.MODEL_FILENAME
    config_file = latest_dir / settings.MODEL_METADATA_FILENAME

    if not model_file.exists():
        raise FileNotFoundError(f"Model file {model_file} does not exist")

    if model_file.suffix != ".pt":
        raise ValueError(f"Model file {model_file} is not a pytorch model")

    if not config_file.exists():
        raise FileNotFoundError(f"Model config {config_file} does not exist")

    config_payload = json.loads(config_file.read_text(encoding="utf-8"))
    config_data = config_payload.get("config", config_payload)
    saved_type = Phase(config_payload.get("phase", latest_dir.parent.name))

    if saved_type == Phase.RANKING:
        return model_file, GhostConfig(**config_data)
    if saved_type == Phase.RETRIEVAL:
        return model_file, RetrievalConfig(**config_data)

    raise ValueError(f"Unsupported model type: {saved_type}")
