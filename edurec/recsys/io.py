import json
import shutil
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .. import settings
from .configs import ModelConfig


def save_model(
    model_config: ModelConfig,
    dataset_name: str,
    best_model_path: str | Path,
    models_folder: str | Path,
) -> tuple[Path, Path]:
    """Save the best model and its config to the expected folder structure."""
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    models_root = Path(models_folder)
    models_root.mkdir(parents=True, exist_ok=True)

    model_folder = models_root / dataset_name / timestamp
    model_folder.mkdir(parents=True, exist_ok=True)

    model_file_path = model_folder / settings.MODEL_FILENAME
    model_config_path = model_folder / settings.MODEL_METADATA_FILENAME

    shutil.move(best_model_path, model_file_path)
    model_config_path.write_text(
        json.dumps(asdict(model_config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return model_file_path, model_config_path


def save_metrics(
    metrics: Mapping[str, float],
    dataset_name: str,
    saving_models_folder: str | Path,
) -> Path:
    file_path = (
        Path(saving_models_folder) / f"{settings.METRICS_FILENAME}_{dataset_name}.csv"
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(dict(metrics), orient="index").to_csv(file_path, index=True)
    return file_path


def load_model(
    models_folder: str | Path,
    dataset_name: str,
) -> tuple[Path, ModelConfig]:
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
    fields = ModelConfig.__dataclass_fields__
    config_data = {k: v for k, v in config_data.items() if k in fields}

    return model_file, ModelConfig(**config_data)
