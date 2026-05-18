from pathlib import Path
from typing import Any

import pandas as pd
from recbole.quick_start import run_recbole

from .. import settings
from ..datasets import ElearningDataModule

BENCHMARK_SPLITS = ("train", "valid", "test")


def eval_sota_models(
    models: list[str],
    dm: ElearningDataModule,
    cfg_path: Path | None = None,
    epochs: int = settings.EPOCHS,
    lr: float = settings.RANKER_LR,
    batch_size: int = settings.RANKER_BATCH_SIZE,
    patience: int = settings.RANKER_PATIENCE,
    top_ks: list[int] | None = None,
    load_side_features: bool = False,
    show_progress: bool = False,
) -> pd.DataFrame:
    dataset_name = dm.dataset_name.value
    atomic_dataset_dir = dm.atomic_folder

    config_dict = _build_config_dict(
        data_root=atomic_dataset_dir.parent,
        atomic_dataset_dir=atomic_dataset_dir,
        dataset_name=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        top_ks=top_ks or settings.TOP_KS,
        load_side_features=load_side_features,
        show_progress=show_progress,
    )

    results = [
        _run_model(
            model=model,
            dataset_name=dataset_name,
            cfg_path=cfg_path,
            config_dict=config_dict,
        )
        for model in models
    ]

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

    return {"model": model, **result["test_result"]}


def _build_config_dict(
    data_root: Path,
    atomic_dataset_dir: Path,
    dataset_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    top_ks: list[int],
    load_side_features: bool = False,
    show_progress: bool = False,
) -> dict[str, object]:
    load_col = {"inter": [settings.USER_COL, settings.ITEM_COL]}

    if load_side_features:
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
        "use_gpu": True,
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
        "train_neg_sample_args": {
            "distribution": "uniform",
            "sample_num": 1,
            "alpha": 1.0,
            "dynamic": False,
            "candidate_num": 0,
        },
        "eval_args": {
            "group_by": "user",
            "order": "TO",
            "mode": "full",
        },
        "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision", "MAP"],
        "topk": top_ks,
        "valid_metric": f"NDCG@{max(top_ks)}",
        "metric_decimal_place": 4,
        "embedding_size": settings.EMB_DIM,
        "show_progress": show_progress,
    }


def _field_names(path: Path) -> list[str]:
    columns = pd.read_csv(path, sep="\t", nrows=0).columns
    return [column.split(":", maxsplit=1)[0] for column in columns]
