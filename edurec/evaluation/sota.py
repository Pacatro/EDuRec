from pathlib import Path
from typing import Any

import pandas as pd
from recbole.quick_start import run_recbole
from recbole.utils import ModelType, InputType, get_model

from .. import settings
from ..datasets import ElearningDataModule

BENCHMARK_SPLITS = ("train", "valid", "test")


def eval_sota_models(
    models: list[str],
    dm: ElearningDataModule,
    cfg_path: Path | None = None,
    epochs: int = settings.EPOCHS,
    lr: float = settings.LR,
    batch_size: int = settings.BATCH_SIZE,
    patience: int = settings.PATIENCE,
    topks: list[int] | None = None,
    results_path: Path | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    dataset_name = dm.data_variant
    atomic_dataset_dir = dm.atomic_folder

    base_config = _build_config_dict(
        data_root=atomic_dataset_dir.parent,
        atomic_dataset_dir=atomic_dataset_dir,
        dataset_name=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        topks=topks or settings.TOP_KS,
        show_progress=show_progress,
    )

    results = []

    for model in models:
        print(f"[EVAL] Evaluating {model}...")
        metrics = _run_model(
            model=model,
            dataset_name=dataset_name,
            cfg_path=cfg_path,
            config_dict=_config_for_model(model, base_config),
        )
        result = {
            "model": model,
            "seed": settings.state["random_state"],
            **metrics,
        }

        if results_path is not None:
            model_root = results_path / model / f"seed_{result['seed']}"
            model_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([result]).to_csv(
                model_root / settings.METRICS_FILENAME,
                index=False,
            )

        results.append(result)

    return pd.DataFrame(results)


def _run_model(
    model: str,
    dataset_name: str,
    cfg_path: Path | None,
    config_dict: dict[str, object],
) -> dict[str, Any]:
    result = run_recbole(
        model=model,
        dataset=dataset_name,
        config_file_list=[str(cfg_path)] if cfg_path else None,
        config_dict=config_dict,
        saved=False,
    )

    return result["test_result"]


def _config_for_model(model: str, base_config: dict[str, object]) -> dict[str, object]:
    config = dict(base_config)
    model_class = get_model(model)
    model_type = getattr(model_class, "type", None)
    input_type = getattr(model_class, "input_type", None)

    if model_type == ModelType.SEQUENTIAL:
        config.pop("benchmark_filename", None)
        config["load_col"] = {
            "inter": [
                settings.USER_COL,
                settings.ITEM_COL,
                settings.TIME_COL,
            ]
        }
        config["ITEM_LIST_LENGTH_FIELD"] = "item_length"
        config["LIST_SUFFIX"] = "_list"
        config["MAX_ITEM_LIST_LENGTH"] = settings.MAX_HISTORY_LEN
        config["train_neg_sample_args"] = None
        return config

    config["train_neg_sample_args"] = (
        {
            "distribution": "uniform",
            "sample_num": 1,
            "alpha": 1.0,
            "dynamic": False,
            "candidate_num": 0,
        }
        if input_type in {InputType.PAIRWISE, InputType.POINTWISE}
        else None
    )

    return config


def _build_config_dict(
    data_root: Path,
    atomic_dataset_dir: Path,
    dataset_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    topks: list[int],
    show_progress: bool = False,
) -> dict[str, object]:
    load_col = {"inter": [settings.USER_COL, settings.ITEM_COL]}

    load_col["user"] = _field_names(atomic_dataset_dir / f"{dataset_name}.user")
    load_col["item"] = _field_names(atomic_dataset_dir / f"{dataset_name}.item")

    return {
        "data_path": str(data_root),
        "benchmark_filename": list(BENCHMARK_SPLITS),
        "USER_ID_FIELD": settings.USER_COL,
        "ITEM_ID_FIELD": settings.ITEM_COL,
        "load_col": load_col,
        "seed": settings.state["random_state"],
        "reproducibility": True,
        "gpu_id": 0,
        "use_gpu": settings.state["device"] != "cpu",
        "epochs": epochs,
        "train_batch_size": batch_size,
        "eval_batch_size": batch_size,
        "learner": "adam",
        "learning_rate": lr,
        "stopping_step": patience,
        "eval_step": 1,
        "save_dataset": False,
        "save_dataloaders": False,
        "checkpoint_dir": "saved/recbole",
        "eval_args": {
            "group_by": "user",
            "order": "TO",
            "mode": "full",
        },
        "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision", "MAP"],
        "topk": topks,
        "valid_metric": f"NDCG@{max(topks)}",
        "metric_decimal_place": 4,
        "embedding_size": settings.EMB_DIM,
        "show_progress": show_progress,
        "state": "info" if show_progress else "error",
        "log_wandb": False,
    }


def _field_names(path: Path) -> list[str]:
    columns = pd.read_csv(path, sep="\t", nrows=0).columns
    return [column.split(":", maxsplit=1)[0] for column in columns]
