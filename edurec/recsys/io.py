import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .ghost import GhostConfig
from .two_tower import RetrievalConfig
from ..datasets import Phase

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
        model_type = Phase.RANKING
    elif isinstance(model_config, RetrievalConfig):
        model_type = Phase.RETRIEVAL
    else:
        raise TypeError(f"Unsupported model config type: {type(model_config)!r}")

    model_name = f"{model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_folder = Path(models_folder) / dataset_name / model_type.value / model_name
    model_folder.mkdir(parents=True, exist_ok=True)

    model_file_path = model_folder / f"{model_name}.pt"
    model_config_path = model_folder / f"{model_name}.json"

    Path(best_model_path).rename(model_file_path)
    model_config_path.write_text(
        json.dumps(
            {"model_type": model_type, "config": asdict(model_config)},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    metrics_path = save_metrics(metrics, model_folder) if metrics is not None else None
    return model_file_path, model_config_path, metrics_path


def save_metrics(metrics: dict[str, float], saving_models_folder: str | Path) -> Path:
    file_path = Path(saving_models_folder) / "metrics.csv"
    pd.DataFrame.from_dict(metrics, orient="index").to_csv(file_path, index=True)
    return file_path


def load_model(
    models_folder: str | Path,
    dataset_name: str,
    model_type: Phase | str | None = None,
) -> tuple[Path, RecsysConfig]:
    """Load the most recent saved model and rebuild its config."""
    root = Path(models_folder) / dataset_name

    if not root.exists():
        raise FileNotFoundError(f"Models folder {root} does not exist")

    if not root.is_dir():
        raise NotADirectoryError(f"Models folder is not a directory: {root}")

    requested_type = Phase(model_type) if model_type is not None else None
    model_dirs = [
        path
        for path in root.rglob("*")
        if path.is_dir()
        and (path / f"{path.name}.pt").exists()
        and (path / f"{path.name}.json").exists()
    ]

    if requested_type is not None:
        filtered_dirs: list[Path] = []
        for path in model_dirs:
            config_payload = json.loads(
                (path / f"{path.name}.json").read_text(encoding="utf-8")
            )
            config_data = config_payload.get("config", config_payload)
            saved_type = config_payload.get("model_type")
            saved_type = (
                Phase(saved_type)
                if saved_type is not None
                else Phase.RANKING
                if "edge_dropout" in config_data or "gnn_layers" in config_data
                else Phase.RETRIEVAL
            )
            if saved_type == requested_type:
                filtered_dirs.append(path)
        model_dirs = filtered_dirs

    if not model_dirs:
        if requested_type is None:
            raise FileNotFoundError(f"No models found in {root}")
        raise FileNotFoundError(f"No models of type {requested_type!r} found in {root}")

    latest_dir = max(model_dirs, key=lambda path: path.stat().st_mtime)
    model_file = latest_dir / f"{latest_dir.name}.pt"
    config_file = latest_dir / f"{latest_dir.name}.json"

    if not model_file.exists():
        raise FileNotFoundError(f"Model file {model_file} does not exist")

    if model_file.suffix != ".pt":
        raise ValueError(f"Model file {model_file} is not a pytorch model")

    if not config_file.exists():
        raise FileNotFoundError(f"Model config {config_file} does not exist")

    config_payload = json.loads(config_file.read_text(encoding="utf-8"))
    config_data = config_payload.get("config", config_payload)
    saved_type = config_payload.get("model_type")

    saved_type = (
        Phase(saved_type)
        if saved_type is not None
        else Phase.RANKING
        if "edge_dropout" in config_data or "gnn_layers" in config_data
        else Phase.RETRIEVAL
    )

    if saved_type == Phase.RANKING:
        return model_file, GhostConfig(**config_data)
    if saved_type == Phase.RETRIEVAL:
        return model_file, RetrievalConfig(**config_data)

    raise ValueError(f"Unsupported model type: {saved_type}")
