from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule
from ..recsys import EDuRecConfig, optimize_model
from .utils import build_config, dataset_run_name, datasets_to_run, print_data_summary

app = typer.Typer(no_args_is_help=True)


@app.command(name="optim", help="Run a hyperparameter optimization for the model.")
def optimize(
    dataset: Annotated[DatasetName | None, typer.Option("--dataset", "-d")] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "-l",
            min=1,
            help="Maximum number of interactions to use before splitting.",
        ),
    ] = None,
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
) -> None:
    datasets = datasets_to_run(dataset)
    verbose = settings.state["verbose"]

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
        run_name = dataset_run_name(dataset, limit)
        if verbose:
            print(
                f"[OPTIM] Config: epochs={epochs}, batch_size={batch_size}, trials={n_trials}, patience={patience}"
            )
            print(
                "[OPTIM] Data config: "
                f"use_processed={use_processed_data}, remove_sparse={remove_sparse}, "
                f"min_interactions={min_interactions}, "
                f"val_ratio={val_ratio}, test_ratio={test_ratio}, limit={limit}"
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
            limit=limit,
        )

        dm.setup()

        print_data_summary("OPTIM", dm)

        dataset_results_path = results_root / run_name

        study = optimize_model(
            base_config=build_config(dm),
            dm=dm,
            n_trials=n_trials,
            epochs=epochs,
            patience=patience,
            verbose=verbose,
            results_path=dataset_results_path,
        )
        best_cfg = EDuRecConfig(**study.best_trial.user_attrs["config"])

        print(
            f"[OPTIM] Best NDCG {study.best_value} in trial: {study.best_trial.number}"
        )
        print("[OPTIM] Params:", study.best_params)

        cfg_path = results_root / f"config-{run_name}.yaml"
        best_cfg.save(cfg_path)
        print("[OPTIM] Saved config:", cfg_path)
        print("[OPTIM] Trials log:", dataset_results_path / "trials.csv")
        print("[OPTIM] Study storage:", dataset_results_path / "study.db")
