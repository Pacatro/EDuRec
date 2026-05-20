import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Mapping

import pandas as pd

from .. import settings
from .architecture import GhostConfig


def save_model(
    model_config: GhostConfig,
    dataset_name: str,
    best_model_path: str | Path,
    models_folder: str | Path,
    metrics: Mapping[str, float],
) -> tuple[Path, Path, Path | None]:
    """Save the best model and its config to the expected folder structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_root = Path(models_folder)
    models_root.mkdir(parents=True, exist_ok=True)

    model_folder = models_root / dataset_name / timestamp
    model_folder.mkdir(parents=True, exist_ok=True)

    model_file_path = model_folder / settings.MODEL_FILENAME
    model_config_path = model_folder / settings.MODEL_METADATA_FILENAME

    Path(best_model_path).rename(model_file_path)
    model_config_path.write_text(
        json.dumps(asdict(model_config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics_path = save_metrics(metrics, model_folder)
    return model_file_path, model_config_path, metrics_path


def save_metrics(
    metrics: Mapping[str, float],
    saving_models_folder: str | Path,
) -> Path:
    file_path = Path(saving_models_folder) / settings.METRICS_FILENAME
    pd.DataFrame.from_dict(dict(metrics), orient="index").to_csv(file_path, index=True)
    return file_path


def load_model(
    models_folder: str | Path,
    dataset_name: str,
) -> tuple[Path, GhostConfig]:
    """Load the most recent saved model and rebuild its config."""
    root = Path(models_folder) / dataset_name

    if not root.exists():
        raise FileNotFoundError(f"Models folder {root} does not exist")

    if not root.is_dir():
        raise NotADirectoryError(f"Models folder is not a directory: {root}")

    model_dirs = sorted(
        {
            config_file.parent
            for config_file in root.rglob(settings.MODEL_METADATA_FILENAME)
            if (config_file.parent / settings.MODEL_FILENAME).exists()
        },
        key=lambda path: path.stat().st_mtime,
    )

    if not model_dirs:
        raise FileNotFoundError(f"No models found in {root}")

    latest_dir = model_dirs[-1]
    model_file = latest_dir / settings.MODEL_FILENAME
    config_file = latest_dir / settings.MODEL_METADATA_FILENAME

    if model_file.suffix != ".pt":
        raise ValueError(f"Model file {model_file} is not a pytorch model")

    config_payload = json.loads(config_file.read_text(encoding="utf-8"))
    config_data = config_payload.get("config", config_payload)

    return model_file, GhostConfig(**config_data)
