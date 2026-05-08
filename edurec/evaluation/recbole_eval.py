from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from recbole.quick_start.quick_start import run_recbole
from recbole.utils import get_model

from .. import settings
from ..datasets import DatasetName, ElearningDataModule
from .cross_validation import (
    CVStrategy,
    FoldSplit,
    InteractionMode,
    build_temporal_series,
    coerce_model_list,
    generate_cv_folds,
    infer_cv_strategy,
    infer_interaction_mode,
    sort_temporal_interactions,
)

DEFAULT_METRICS: tuple[str, ...] = ("Recall", "NDCG", "MRR", "Hit", "Precision")
POINTWISE_SEQUENCE_MODELS: set[str] = {"BERT4Rec", "GRU4Rec", "NARM", "SASRec"}
ITEM_LIST_FIELD = f"{settings.ITEM_COL}_list"


def evaluate_recbole_models(
    dataset: DatasetName,
    *,
    models: Sequence[str] | None = None,
    n_splits: int = 5,
    val_size: float = 0.1,
    batch_size: int = 256,
    epochs: int = 50,
    patience: int = 5,
    top_ks: Sequence[int] = (5, 10, 20),
    results_folder: str = settings.RESULTS_FOLDER,
    remove_sparse: bool = settings.REMOVE_SPARSE,
    min_interactions: int = settings.MIN_INTERACTIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions = _load_interactions(
        dataset=dataset,
        remove_sparse=remove_sparse,
        min_interactions=min_interactions,
    )
    cv_strategy = infer_cv_strategy(interactions)
    interaction_mode = infer_interaction_mode(interactions)
    selected_models = coerce_model_list(models, interactions)
    folds = generate_cv_folds(
        interactions=interactions,
        n_splits=n_splits,
        val_size=val_size,
        random_state=settings.state["random_state"],
        strategy=cv_strategy,
    )

    output_dir = Path(results_folder) / "recbole" / dataset.value
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for fold in folds:
        for model_name in selected_models:
            sequential_model = _is_sequential_model(model_name)
            model_dir = output_dir / f"fold_{fold.fold}" / model_name.lower()
            dataset_token = f"{dataset.value}_fold_{fold.fold}_{model_name.lower()}"

            _write_fold_dataset(
                fold=fold,
                fold_dir=model_dir,
                dataset_token=dataset_token,
                temporal=cv_strategy == CVStrategy.TEMPORAL,
                sequential=sequential_model,
            )

            if settings.state["verbose"]:
                print(
                    f"[SOTA] Evaluating {model_name} on {dataset.value} "
                    f"(fold {fold.fold}/{len(folds)})"
                )

            try:
                result = run_recbole(
                    model=model_name,
                    dataset=dataset_token,
                    config_dict=_build_recbole_config(
                        fold_dir=model_dir,
                        model_name=model_name,
                        temporal=cv_strategy == CVStrategy.TEMPORAL,
                        sequential=sequential_model,
                        batch_size=batch_size,
                        epochs=epochs,
                        patience=patience,
                        top_ks=top_ks,
                    ),
                    saved=False,
                )
            except Exception as exc:
                rows.append(
                    {
                        "dataset": dataset.value,
                        "model": model_name,
                        "fold": fold.fold,
                        "cv_strategy": cv_strategy.value,
                        "interaction_mode": interaction_mode.value,
                        "error": str(exc),
                    }
                )
                continue

            metric_row: dict[str, object] = {
                "dataset": dataset.value,
                "model": model_name,
                "fold": fold.fold,
                "cv_strategy": cv_strategy.value,
                "interaction_mode": interaction_mode.value,
                "error": None,
            }
            metric_row.update(
                {metric: float(value) for metric, value in result["test_result"].items()}
            )
            rows.append(metric_row)

    fold_results = pd.DataFrame(rows)
    if fold_results.empty:
        raise RuntimeError("RecBole evaluation did not produce any result rows.")

    summary = _summarize_results(
        fold_results=fold_results,
        dataset=dataset,
        cv_strategy=cv_strategy,
        interaction_mode=interaction_mode,
    )

    fold_results.to_csv(output_dir / "fold_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    return summary, fold_results


def _load_interactions(
    dataset: DatasetName,
    remove_sparse: bool,
    min_interactions: int,
) -> pd.DataFrame:
    dm = ElearningDataModule(
        dataset=dataset,
        batch_size=1,
        test_ratio=0.2,
        val_ratio=0.1,
        remove_sparse=remove_sparse,
        min_interactions=min_interactions,
        use_processed_data=False,
        random_state=settings.state["random_state"],
    )
    return dm.interactions.copy()


def _write_fold_dataset(
    fold: FoldSplit,
    fold_dir: Path,
    dataset_token: str,
    temporal: bool,
    sequential: bool,
) -> None:
    dataset_dir = fold_dir / dataset_token
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if sequential:
        _write_sequential_fold_dataset(
            fold=fold,
            dataset_dir=dataset_dir,
            dataset_token=dataset_token,
        )
        return

    _write_interactions_file(
        interactions=fold.train,
        output_path=dataset_dir / f"{dataset_token}.train.inter",
        temporal=temporal,
    )
    _write_interactions_file(
        interactions=fold.valid,
        output_path=dataset_dir / f"{dataset_token}.valid.inter",
        temporal=temporal,
    )
    _write_interactions_file(
        interactions=fold.test,
        output_path=dataset_dir / f"{dataset_token}.test.inter",
        temporal=temporal,
    )


def _write_sequential_fold_dataset(
    fold: FoldSplit,
    dataset_dir: Path,
    dataset_token: str,
) -> None:
    history_state: dict[object, list[object]] = {}
    for split_name, split_df in (
        ("train", fold.train),
        ("valid", fold.valid),
        ("test", fold.test),
    ):
        export_df, history_state = _build_sequential_rows(split_df, history_state)
        export_df.rename(
            columns={
                settings.USER_COL: f"{settings.USER_COL}:token",
                settings.ITEM_COL: f"{settings.ITEM_COL}:token",
                ITEM_LIST_FIELD: f"{ITEM_LIST_FIELD}:token_seq",
            }
        ).to_csv(
            dataset_dir / f"{dataset_token}.{split_name}.inter",
            sep="\t",
            index=False,
        )


def _build_sequential_rows(
    interactions: pd.DataFrame,
    initial_state: dict[object, list[object]],
) -> tuple[pd.DataFrame, dict[object, list[object]]]:
    ordered_df = sort_temporal_interactions(interactions)
    history_state = {user_id: list(items) for user_id, items in initial_state.items()}
    rows: list[dict[str, object]] = []

    for row in ordered_df.itertuples(index=False):
        user_id = getattr(row, settings.USER_COL)
        item_id = getattr(row, settings.ITEM_COL)
        history = history_state.get(user_id, [])
        if history:
            rows.append(
                {
                    settings.USER_COL: user_id,
                    settings.ITEM_COL: item_id,
                    ITEM_LIST_FIELD: " ".join(map(str, history[-50:])),
                }
            )
        history_state.setdefault(user_id, []).append(item_id)

    return (
        pd.DataFrame(
            rows,
            columns=[settings.USER_COL, settings.ITEM_COL, ITEM_LIST_FIELD],
        ),
        history_state,
    )


def _write_interactions_file(
    interactions: pd.DataFrame,
    output_path: Path,
    temporal: bool,
) -> None:
    export_df = interactions[[settings.USER_COL, settings.ITEM_COL]].copy()

    if temporal:
        ordered_df = sort_temporal_interactions(interactions)
        export_df = ordered_df[[settings.USER_COL, settings.ITEM_COL]].copy()
        export_df[settings.TIME_COL] = build_temporal_series(ordered_df)

    renamed_columns = {
        settings.USER_COL: f"{settings.USER_COL}:token",
        settings.ITEM_COL: f"{settings.ITEM_COL}:token",
    }
    if temporal:
        renamed_columns[settings.TIME_COL] = f"{settings.TIME_COL}:float"

    export_df.rename(columns=renamed_columns).to_csv(
        output_path,
        sep="\t",
        index=False,
    )


def _build_recbole_config(
    *,
    fold_dir: Path,
    model_name: str,
    temporal: bool,
    sequential: bool,
    batch_size: int,
    epochs: int,
    patience: int,
    top_ks: Sequence[int],
) -> dict[str, object]:
    checkpoint_dir = fold_dir / "checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if sequential:
        load_columns = [settings.USER_COL, settings.ITEM_COL, ITEM_LIST_FIELD]
    else:
        load_columns = [settings.USER_COL, settings.ITEM_COL]
        if temporal:
            load_columns.append(settings.TIME_COL)

    return {
        "data_path": str(fold_dir),
        "benchmark_filename": ["train", "valid", "test"],
        "USER_ID_FIELD": settings.USER_COL,
        "ITEM_ID_FIELD": settings.ITEM_COL,
        "TIME_FIELD": None if sequential else settings.TIME_COL if temporal else None,
        "load_col": {"inter": load_columns},
        "field_separator": "\t",
        "seq_separator": " ",
        "epochs": epochs,
        "stopping_step": patience,
        "train_batch_size": batch_size,
        "eval_batch_size": batch_size,
        "metrics": list(DEFAULT_METRICS),
        "topk": list(top_ks),
        "valid_metric": f"Recall@{max(top_ks)}",
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "order": "TO" if sequential or temporal else "RO",
            "group_by": "user" if not temporal else "none",
            "mode": {"valid": "full", "test": "full"},
        },
        "train_neg_sample_args": (
            None
            if model_name in POINTWISE_SEQUENCE_MODELS
            else {"distribution": "uniform", "sample_num": 1}
        ),
        "checkpoint_dir": str(checkpoint_dir),
        "save_dataset": False,
        "save_dataloaders": False,
        "dataset_save_path": None,
        "dataloaders_save_path": None,
        "repeatable": True,
        "reproducibility": True,
        "seed": settings.state["random_state"],
        "device": _resolve_device(),
        "show_progress": settings.state["verbose"],
        "state": "INFO" if settings.state["verbose"] else "ERROR",
    }


def _resolve_device() -> str:
    configured_device = settings.state["device"]
    if configured_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return configured_device


def _is_sequential_model(model_name: str) -> bool:
    return "sequential_recommender" in get_model(model_name).__module__


def _summarize_results(
    *,
    fold_results: pd.DataFrame,
    dataset: DatasetName,
    cv_strategy: CVStrategy,
    interaction_mode: InteractionMode,
) -> pd.DataFrame:
    successful_rows = fold_results[fold_results["error"].isna()].copy()
    if successful_rows.empty:
        raise RuntimeError("All RecBole evaluations failed.")

    metadata_cols = {
        "dataset",
        "model",
        "fold",
        "cv_strategy",
        "interaction_mode",
        "error",
    }
    metric_cols = [
        col for col in successful_rows.columns if col not in metadata_cols
    ]

    summary = (
        successful_rows.groupby("model", as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values("model")
        .reset_index(drop=True)
    )
    summary.insert(0, "dataset", dataset.value)
    summary.insert(1, "cv_strategy", cv_strategy.value)
    summary.insert(2, "interaction_mode", interaction_mode.value)
    summary["n_folds"] = successful_rows.groupby("model")["fold"].count().values
    return summary


__all__ = ["evaluate_recbole_models"]
