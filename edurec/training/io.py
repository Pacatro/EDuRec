import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from edurec.training.model import EDuRecConfig


def save_model(
    model_name: str,
    model_config: EDuRecConfig,
    best_model_path: str,
    models_folder: str,
) -> None:
    """Save the best model to the specified folder."""
    out_model = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    saving_models_folder = Path(models_folder) / f"{out_model}"
    saving_models_folder.mkdir(parents=True, exist_ok=True)

    model_file_path = saving_models_folder / f"{out_model}.pt"
    model_config_path = saving_models_folder / f"{out_model}.json"
    Path(best_model_path).rename(model_file_path)

    model_config_path.write_text(
        json.dumps(asdict(model_config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Model saved in: {model_file_path}")


def load_model(models_folder: str | Path) -> tuple[Path, EDuRecConfig]:
    """Get the last saved model and config."""
    root = Path(models_folder)

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

    if latest_dir.suffix != ".pt":
        raise ValueError(f"Model {latest_dir} is not a pytorch model")

    config_data = json.loads(config_file.read_text(encoding="utf-8"))
    model_config = EDuRecConfig(**config_data)

    return model_file, model_config
