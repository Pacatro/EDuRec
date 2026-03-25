import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from .model import GhostConfig


def save_model(
    model_name: str,
    model_config: GhostConfig,
    best_model_path: str,
    models_folder: str,
    dataset_name: str,
    metrics: dict[str, float] | None = None,
) -> tuple[Path, Path, Path | None]:
    """Save the best model to the specified folder."""
    out_model = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    saving_models_folder = Path(models_folder) / dataset_name / out_model
    saving_models_folder.mkdir(parents=True, exist_ok=True)

    model_file_path = saving_models_folder / f"{out_model}.pt"
    model_config_path = saving_models_folder / f"{out_model}.json"
    Path(best_model_path).rename(model_file_path)

    model_config_path.write_text(
        json.dumps(asdict(model_config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics_path = (
        save_metrics(metrics, saving_models_folder) if metrics is not None else None
    )

    return model_file_path, model_config_path, metrics_path


def save_metrics(metrics: dict[str, float], saving_models_folder: str | Path) -> Path:
    file_path = Path(saving_models_folder) / "metrics.csv"
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.to_csv(file_path, index=True)
    return file_path


def load_model(
    models_folder: str | Path, dataset_name: str
) -> tuple[Path, GhostConfig]:
    """Get the last saved model and config."""
    root = Path(models_folder) / dataset_name

    if not root.exists():
        raise FileNotFoundError(f"Models folder {root} does not exist")

    if not root.is_dir():
        raise NotADirectoryError(f"Models folder is not a directory: {root}")

    models_dirs = [p for p in root.iterdir() if p.is_dir()]

    if not models_dirs:
        raise FileNotFoundError(f"No models found in {root}")

    latest_dir = root / max(models_dirs, key=lambda p: p.stat().st_mtime).name

    if not latest_dir.exists():
        raise FileNotFoundError(f"Model {latest_dir} does not exist")

    model_file = latest_dir / f"{latest_dir.name}.pt"
    config_file = latest_dir / f"{latest_dir.name}.json"

    if not model_file.exists():
        raise FileNotFoundError(f"Model file {model_file} does not exist")

    if model_file.suffix != ".pt":
        raise ValueError(f"Model file {model_file} is not a pytorch model")

    if not config_file.exists():
        raise FileNotFoundError(f"Model config {config_file} does not exist")

    config_data = json.loads(config_file.read_text(encoding="utf-8"))
    model_config = GhostConfig(**config_data)

    return model_file, model_config
