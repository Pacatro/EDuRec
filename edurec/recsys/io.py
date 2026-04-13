import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .model import GnnRankerConfig
from .two_tower import RetrievalConfig

type RecsysConfig = GnnRankerConfig | RetrievalConfig

_RERANKER_TYPE = "reranker"
_RETRIEVAL_TYPE = "retrieval"


def save_model(
    model_name: str,
    model_config: RecsysConfig,
    dataset_name: str,
    best_model_path: str | Path,
    models_folder: str | Path,
    metrics: dict[str, float] | None = None,
) -> tuple[Path, Path, Path | None]:
    """Save the best model and its config to the specified folder."""
    if not is_dataclass(model_config):
        raise TypeError("model_config must be a dataclass instance.")

    out_model = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    saving_models_folder = Path(models_folder) / dataset_name / out_model
    saving_models_folder.mkdir(parents=True, exist_ok=True)

    model_file_path = saving_models_folder / f"{out_model}.pt"
    model_config_path = saving_models_folder / f"{out_model}.json"
    Path(best_model_path).rename(model_file_path)

    model_config_path.write_text(
        json.dumps(
            {
                "model_type": _get_model_type(model_config),
                "config": asdict(model_config),
            },
            indent=2,
            ensure_ascii=False,
        ),
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
    models_folder: str | Path,
    dataset_name: str,
    model_type: str | None = None,
) -> tuple[Path, RecsysConfig]:
    """Get the last saved model and reconstruct its config."""
    root = Path(models_folder) / dataset_name

    if not root.exists():
        raise FileNotFoundError(f"Models folder {root} does not exist")

    if not root.is_dir():
        raise NotADirectoryError(f"Models folder is not a directory: {root}")

    models_dirs = [p for p in root.iterdir() if p.is_dir()]

    if not models_dirs:
        raise FileNotFoundError(f"No models found in {root}")

    if model_type is not None:
        models_dirs = [
            p for p in models_dirs if _load_model_type(p / f"{p.name}.json") == model_type
        ]

        if not models_dirs:
            raise FileNotFoundError(
                f"No models of type {model_type!r} found in {root}"
            )

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

    config_payload = _load_config_payload(config_file)
    model_config = _build_model_config(config_payload)

    return model_file, model_config


def _get_model_type(model_config: RecsysConfig) -> str:
    if isinstance(model_config, GnnRankerConfig):
        return _RERANKER_TYPE
    if isinstance(model_config, RetrievalConfig):
        return _RETRIEVAL_TYPE
    raise TypeError(f"Unsupported model config type: {type(model_config)!r}")


def _build_model_config(config_payload: dict) -> RecsysConfig:
    if "model_type" in config_payload and "config" in config_payload:
        model_type = config_payload["model_type"]
        config_data = config_payload["config"]
    else:
        model_type = _infer_model_type(config_payload)
        config_data = config_payload

    if model_type == _RERANKER_TYPE:
        return GnnRankerConfig(**config_data)
    if model_type == _RETRIEVAL_TYPE:
        return RetrievalConfig(**config_data)

    raise ValueError(f"Unsupported model type: {model_type}")


def _infer_model_type(config_data: dict) -> str:
    if "edge_dropout" in config_data or "gnn_layers" in config_data:
        return _RERANKER_TYPE
    return _RETRIEVAL_TYPE


def _load_config_payload(config_file: str | Path) -> dict:
    return json.loads(Path(config_file).read_text(encoding="utf-8"))


def _load_model_type(config_file: str | Path) -> str:
    config_payload = _load_config_payload(config_file)
    if "model_type" in config_payload:
        return config_payload["model_type"]
    return _infer_model_type(config_payload)
