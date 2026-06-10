from datetime import datetime
from typing import Annotated
from pathlib import Path

import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule, dataset_loaders
from ..recsys import EDuRecConfig, optimize_model

app = typer.Typer(no_args_is_help=True)


@app.command(name="optim", help="Run a hyperparameter optimization for the model.")
def optimize(
    dataset: Annotated[DatasetName | None, typer.Option("--dataset", "-d")] = None,
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            min=1,
            help="Number of training epochs used by all evaluated models.",
        ),
    ] = settings.EPOCHS,
    patience: Annotated[
        int,
        typer.Option(
            "--patience",
            "-p",
            min=1,
            help="Early stopping patience used by all evaluated models.",
        ),
    ] = settings.PATIENCE,
    n_trials: Annotated[
        int,
        typer.Option(
            "--trials",
            "-n",
            min=1,
            help="Number of optimization trials to perform.",
        ),
    ] = settings.OPTIM_N_TRIALS,
    batch_size: Annotated[
        int, typer.Option("--batch_size", "-b", min=1)
    ] = settings.BATCH_SIZE,
    val_ratio: Annotated[float, typer.Option("--val_size", "-v")] = settings.VAL_RATIO,
    test_ratio: Annotated[
        float, typer.Option("--test_size", "-t")
    ] = settings.TEST_RATIO,
    remove_sparse: Annotated[
        bool, typer.Option("--remove_sparse", "-R")
    ] = settings.REMOVE_SPARSE,
    min_interactions: Annotated[
        int, typer.Option("--min_interactions", "-i")
    ] = settings.MIN_INTERACTIONS,
    use_processed_data: Annotated[
        bool, typer.Option("--use_processed", "-P")
    ] = settings.SAVE_DATA,
):
    datasets = [dataset] if dataset is not None else dataset_loaders.keys()

    results_root = (
        Path(settings.RESULTS_FOLDER)
        / "optimization"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    results_root.mkdir(parents=True, exist_ok=True)

    print("\n[OPTIM] Hyperparameter optimization run")
    print(f"[OPTIM] Datasets: {', '.join(ds.value for ds in datasets)}")
    print(f"[OPTIM] Results folder: {results_root}")

    for dataset in datasets:
        if settings.state["verbose"]:
            print(
                f"[OPTIM] Config: epochs={epochs}, batch_size={batch_size}, trials={n_trials}, patience={patience}"
            )
            print(
                "[OPTIM] Data config: "
                f"use_processed={use_processed_data}, remove_sparse={remove_sparse}, "
                f"min_interactions={min_interactions}, "
                f"val_ratio={val_ratio}, test_ratio={test_ratio}"
            )

        dm = ElearningDataModule(
            dataset=dataset,
            batch_size=batch_size,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            use_processed_data=use_processed_data,
            random_state=settings.state["random_state"],
            min_interactions=min_interactions,
            remove_sparse=remove_sparse,
            save_atomic_files=False,
        )

        dm.setup()

        split_sizes = {
            split: len(getattr(dm.artifacts, split))
            for split in ("train", "val", "test")
            if getattr(dm.artifacts, split) is not None
        }

        print(
            "[OPTIM] Data ready: "
            f"users={dm.num_users:,}, items={dm.num_items:,}, "
            f"interactions={dm.num_interactions:,}, "
            f"sparsity={dm.sparsity:.4f}"
        )
        print(
            "[OPTIM] Splits: "
            + ", ".join(f"{split}={size:,}" for split, size in split_sizes.items())
        )
        if settings.state["verbose"]:
            print(
                "[OPTIM] Features: "
                f"context={dm.train_ds.num_ctx_feats}, "
                f"user_dense={dm.num_user_dense_feats}, "
                f"item_dense={dm.num_item_dense_feats}, "
                f"user_text={dm.num_user_text_feats}, "
                f"item_text={dm.num_item_text_feats}, "
                f"user_cat={len(dm.user_cat_cardinalities)}, "
                f"item_cat={len(dm.item_cat_cardinalities)}"
            )

        cfg = EDuRecConfig(
            num_users=dm.num_users,
            num_items=dm.num_items,
            num_ctx_feats=dm.train_ds.num_ctx_feats,
            num_user_dense_feats=dm.num_user_dense_feats,
            num_item_dense_feats=dm.num_item_dense_feats,
            num_user_text_feats=dm.num_user_text_feats,
            num_item_text_feats=dm.num_item_text_feats,
            user_cat_cardinalities=dm.user_cat_cardinalities,
            item_cat_cardinalities=dm.item_cat_cardinalities,
        )

        dataset_results_path = results_root / dataset.value

        study = optimize_model(
            base_config=cfg,
            dm=dm,
            n_trials=n_trials,
            epochs=epochs,
            patience=patience,
            verbose=settings.state["verbose"],
            results_path=dataset_results_path,
        )
        cfg = EDuRecConfig(**study.best_trial.user_attrs["config"])

        print(
            f"[OPTIM] Best NDCG {study.best_value} in trial: {study.best_trial.number}"
        )
        print("[OPTIM] Params:", study.best_params)

        cfg_path = results_root / f"config-{dataset.value}.yaml"
        cfg.save(cfg_path)
        print("[OPTIM] Saved config:", cfg_path)
        print("[OPTIM] Trials log:", dataset_results_path / "trials.csv")
        print("[OPTIM] Study storage:", dataset_results_path / "study.db")
