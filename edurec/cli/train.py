import datetime
from pathlib import Path
from typing import Annotated

import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule
from ..recsys import EDuRecConfig, RecSys, train_model
from ..recsys.io import save_metrics, save_model
from .utils import (
    build_config,
    dataset_run_name,
    datasets_to_run,
    print_data_summary,
    print_model_modules,
)

app = typer.Typer(no_args_is_help=True)


@app.command(name="train", help="Train the model.")
def train(
    dataset: Annotated[DatasetName | None, typer.Option("--dataset", "-d")] = None,
    epochs: Annotated[int, typer.Option("--epochs", "-e")] = settings.EPOCHS,
    lr: Annotated[float, typer.Option("--lr", "-l")] = settings.LR,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Maximum number of interactions to use before splitting.",
        ),
    ] = None,
    batch_size: Annotated[
        int, typer.Option("--batch_size", "-b")
    ] = settings.BATCH_SIZE,
    patience: Annotated[int, typer.Option("--patience", "-p")] = settings.PATIENCE,
    val_size: Annotated[float, typer.Option("--val_size", "-v")] = settings.VAL_RATIO,
    test_size: Annotated[
        float, typer.Option("--test_size", "-t")
    ] = settings.TEST_RATIO,
    top_k: Annotated[int, typer.Option("--top_k", "-k")] = settings.TOP_K,
    remove_sparse: Annotated[
        bool, typer.Option("--remove_sparse", "-R")
    ] = settings.REMOVE_SPARSE,
    min_interactions: Annotated[
        int, typer.Option("--min_interactions", "-i")
    ] = settings.MIN_INTERACTIONS,
    adaptive_k: Annotated[
        bool, typer.Option("--adaptive_k", "-a")
    ] = settings.ADAPTIVE_K,
    debug: Annotated[bool, typer.Option("--debug", "-D")] = False,
    save: Annotated[bool, typer.Option("--save_model", "-S")] = False,
    use_processed_data: Annotated[
        bool, typer.Option("--use_processed", "-P")
    ] = settings.SAVE_DATA,
    models_folder: Annotated[
        Path, typer.Option("--models-folder", "-M")
    ] = Path(settings.MODELS_FOLDER),
    configs_folder: Annotated[
        Path, typer.Option("--configs-folder", "-C")
    ] = Path(settings.CONFIGS_FOLDER),
    experiment_name: Annotated[
        str | None, typer.Option("--experiment-name", "-E")
    ] = None,
) -> None:
    started_at = datetime.datetime.now(datetime.UTC)
    verbose = settings.state["verbose"]
    monitor_metric = f"val/ndcg@{top_k}"
    training_root = Path(settings.RESULTS_FOLDER) / "training"

    datasets = datasets_to_run(dataset)

    for dataset_idx, dataset_name in enumerate(datasets, start=1):
        run_name = dataset_run_name(dataset_name, limit)
        dataset_experiment_name = (
            f"{experiment_name}_{run_name}" if experiment_name else None
        )
        print("\n[TRAIN] Training run")
        print(f"[TRAIN] Dataset {dataset_idx}/{len(datasets)}: {run_name}")
        print("[TRAIN] Model: EDuRec")
        print(f"[TRAIN] Monitor: {monitor_metric}")
        print(f"[TRAIN] Save model: {save}")
        print("[TRAIN] Preparing data...")

        if verbose:
            print(
                "[TRAIN] Config: "
                f"epochs={epochs}, lr={lr}, batch_size={batch_size}, "
                f"patience={patience}, adaptive_k={adaptive_k}, debug={debug}"
            )

            if dataset_experiment_name:
                print(
                    f"[TRAIN] Logger: WandB, experiment_name={dataset_experiment_name}"
                )

            print(
                "[TRAIN] Data config: "
                f"use_processed={use_processed_data}, remove_sparse={remove_sparse}, "
                f"min_interactions={min_interactions}, "
                f"val_ratio={val_size}, test_ratio={test_size}, limit={limit}"
            )

        dm = ElearningDataModule(
            dataset=dataset_name,
            batch_size=batch_size,
            test_ratio=test_size,
            val_ratio=val_size,
            use_processed_data=use_processed_data,
            random_state=settings.state["random_state"],
            min_interactions=min_interactions,
            remove_sparse=remove_sparse,
            save_atomic_files=True,
            limit=limit,
        )

        dm.prepare_data()
        dm.setup()

        print_data_summary("TRAIN", dm)

        config_path = configs_folder / f"config-{run_name}.yaml"

        if config_path.exists():
            cfg = build_config(
                dm,
                base=EDuRecConfig.load(config_path),
                adaptive_k=adaptive_k,
                topks=settings.TOP_KS,
            )
            print(f"[TRAIN] Using saved configuration: {config_path}")
        else:
            cfg = build_config(
                dm,
                lr=lr,
                adaptive_k=adaptive_k,
                topks=settings.TOP_KS,
            )
            print("[TRAIN] No saved configuration found; using the full model.")

        print_model_modules("TRAIN", cfg)

        recsys = RecSys(
            cfg=cfg,
            inter_graph=dm.build_inter_graph(),
            u_static_feats=dm.u_static_feats,
            i_static_feats=dm.i_static_feats,
        )

        print("[TRAIN] Training EDuRec...")

        trainer, best_model_path, timer = train_model(
            model=recsys,
            dm=dm,
            debug=debug,
            epochs=epochs,
            patience=patience,
            experiment_name=dataset_experiment_name,
            monitor=monitor_metric,
            verbose=verbose,
        )

        if debug:
            elapsed = str(
                datetime.datetime.now(datetime.UTC) - started_at
            ).split(".", maxsplit=1)[0]
            print("[TRAIN] Debug mode: skipping evaluation")
            print(f"[TRAIN] Finished {dataset_name.value} in {elapsed}\n")
            return

        metrics = dict(
            trainer.test(ckpt_path="best", datamodule=dm, weights_only=False)[0]
        )
        metrics["training_time_s"] = timer.time_elapsed("train")
        metrics["inference_time_s"] = timer.time_elapsed("test")

        print(f"[TRAIN] Training time: {metrics['training_time_s']}")
        print(f"[TRAIN] Inference time: {metrics['inference_time_s']}")
        print(f"[TRAIN] Best checkpoint: {best_model_path}")

        metrics_path = save_metrics(metrics, dataset_name.value, training_root)
        print(f"[TRAIN] Metrics saved: {metrics_path}")

        if save and trainer.is_global_zero:
            model_file_path, model_config_path = save_model(
                model_config=cfg,
                dataset_name=run_name,
                best_model_path=best_model_path,
                models_folder=models_folder,
            )
            print(f"[TRAIN] Model weights saved: {model_file_path}")
            print(f"[TRAIN] Model config saved: {model_config_path}")

        now = datetime.datetime.now(datetime.UTC)
        elapsed = str(now - started_at).split(".", maxsplit=1)[0]
        print(f"[TRAIN] Finished {dataset_name.value} in {elapsed}\n")
