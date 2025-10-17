from pathlib import Path
from datetime import datetime


def save_best_model(model_name: str, best_model_path: str, models_folder: str) -> None:
    """Save the best model to the specified folder."""
    out_model = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    saving_models_folder = Path(models_folder)
    saving_models_folder.mkdir(parents=True, exist_ok=True)

    file_path = saving_models_folder / f"{out_model}.pt"
    Path(best_model_path).rename(file_path)
    print(f"Model saved in: {file_path}")


def get_last_model(models_folder: str) -> str:
    """Get the last model in the specified folder."""
    saving_models_folder = Path(models_folder)

    if not saving_models_folder.exists():
        raise FileNotFoundError(f"Models folder {saving_models_folder} does not exist")

    models = [f for f in saving_models_folder.iterdir() if f.is_file()]

    if len(models) == 0:
        raise FileNotFoundError(f"No models found in {saving_models_folder}")

    model_path = saving_models_folder / max(models).name

    if model_path.suffix != ".pt":
        raise ValueError(f"Model {model_path} is not a pytorch model")

    return str(model_path)
