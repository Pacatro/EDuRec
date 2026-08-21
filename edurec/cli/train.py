import datetime
from pathlib import Path
from typing import Annotated

import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule
from ..recsys import ModelConfig, RecSys, train_model
from ..recsys.configs import monitor_topk, resolve_train_config
from ..recsys.io import save_metrics, save_model
from .utils import (
    build_config,
    config_paths,
    datasets_to_run,
    dataset_train_defaults,
    print_data_summary,
    print_model_modules,
)

app = typer.Typer(no_args_is_help=True)


@app.command(name="train", help="Train the model.")
def train(
    dataset: Annotated[DatasetName | None, typer.Option("--dataset", "-d")] = None,
    epochs: Annotated[
        int | None,
        typer.Option(
            "--epochs",
            "-e",
            min=1,
            help="Number of training epochs. Uses the saved training config if omitted.",
        ),
    ] = None,
    lr: Annotated[
        float | None,
        typer.Option(
            "--lr",
            "-l",
            min=0.0,
            help="Learning rate. Uses the saved training config if omitted.",
        ),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch_size",
            "-b",
            min=1,
            help="Batch size. Uses the saved training config if omitted.",
        ),
    ] = None,
    patience: Annotated[
        int | None,
        typer.Option(
            "--patience",
            "-p",
            min=1,
            help="Early stopping patience. Uses the saved training config if omitted.",
        ),
    ] = None,
    val_size: Annotated[float, typer.Option("--val_size", "-v")] = settings.VAL_RATIO,
    test_size: Annotated[
        float, typer.Option("--test_size", "-t")
    ] = settings.TEST_RATIO,
    top_k: Annotated[
        int | None,
        typer.Option(
            "--top_k",
            "-k",
            min=1,
            help="Validation cutoff for early stopping. "
            "Defaults to the maximum saved top-k.",
        ),
    ] = None,
    remove_sparse: Annotated[
        bool, typer.Option("--remove_sparse", "-R")
    ] = settings.REMOVE_SPARSE,
    min_interactions: Annotated[
        int, typer.Option("--min_interactions", "-i")
    ] = settings.MIN_INTERACTIONS,
    adaptive_k: Annotated[
        bool | None,
        typer.Option(
            "--adaptive_k",
            "-a",
            help="Use adaptive k. Uses the saved training config if omitted.",
        ),
    ] = None,
    compile: Annotated[
        bool,
        typer.Option("--compile", help="Compile the model before training."),
    ] = settings.COMPILE_MODEL,
    debug: Annotated[bool, typer.Option("--debug", "-D")] = False,
    save: Annotated[bool, typer.Option("--save_model", "-S")] = False,
    use_processed_data: Annotated[
        bool, typer.Option("--use_processed", "-P")
    ] = settings.SAVE_DATA,
    models_folder: Annotated[Path, typer.Option("--models-folder", "-M")] = Path(
        settings.MODELS_FOLDER
    ),
    configs_folder: Annotated[Path, typer.Option("--configs-folder", "-C")] = Path(
        settings.CONFIGS_FOLDER
    ),
    experiment_name: Annotated[
        str | None, typer.Option("--experiment-name", "-E")
    ] = None,
) -> None:
    started_at = datetime.datetime.now(datetime.UTC)
    verbose = settings.state["verbose"]
    training_root = Path(settings.RESULTS_FOLDER) / "training"

    datasets = datasets_to_run(dataset)

    for dataset_idx, dataset_name in enumerate(datasets, start=1):
        run_name = dataset_name.value
        dataset_experiment_name = (
            f"{experiment_name}_{run_name}" if experiment_name else None
        )
        model_config_path, train_config_path = config_paths(configs_folder, run_name)
        train_cfg = resolve_train_config(
            cli={
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "patience": patience,
                "adaptive_k": adaptive_k,
            },
            saved_path=train_config_path,
            defaults=dataset_train_defaults(dataset_name),
        )
        val_topk = monitor_topk(top_k, train_cfg)

        print("\n[TRAIN] Training run")
        print(f"[TRAIN] Dataset {dataset_idx}/{len(datasets)}: {run_name}")
        print("[TRAIN] Model: EDuRec")
        print(f"[TRAIN] Monitor: val/ndcg@{val_topk}")
        print(f"[TRAIN] Save model: {save}")
        print("[TRAIN] Preparing data...")

        if verbose:
            print(
                "[TRAIN] Config: "
                f"epochs={train_cfg.epochs}, lr={train_cfg.lr}, "
                f"batch_size={train_cfg.batch_size}, patience={train_cfg.patience}, "
                f"adaptive_k={train_cfg.adaptive_k}, debug={debug}"
            )

            if dataset_experiment_name:
                print(
                    f"[TRAIN] Logger: WandB, experiment_name={dataset_experiment_name}"
                )

            print(
                "[TRAIN] Data config: "
                f"use_processed={use_processed_data}, remove_sparse={remove_sparse}, "
                f"min_interactions={min_interactions}, "
                f"val_ratio={val_size}, test_ratio={test_size}"
            )

        dm = ElearningDataModule(
            dataset=dataset_name,
            batch_size=train_cfg.batch_size,
            test_ratio=test_size,
            val_ratio=val_size,
            use_processed_data=use_processed_data,
            random_state=settings.state["random_state"],
            min_interactions=min_interactions,
            remove_sparse=remove_sparse,
            save_atomic_files=True,
        )

        dm.prepare_data()
        dm.setup()

        print_data_summary("TRAIN", dm)

        if model_config_path.exists():
            cfg = build_config(dm, base=ModelConfig.load(model_config_path))
            print(f"[TRAIN] Using saved model config: {model_config_path}")
        else:
            cfg = build_config(dm)
            print("[TRAIN] No saved model config found; using the full model.")

        if train_config_path.exists():
            print(f"[TRAIN] Using saved training config: {train_config_path}")

        print_model_modules("TRAIN", cfg)

        recsys = RecSys(
            cfg=cfg,
            inter_graph=dm.build_inter_graph(),
            u_static_feats=dm.u_static_feats,
            i_static_feats=dm.i_static_feats,
            train_cfg=train_cfg,
            val_topk=val_topk,
        )

        print("[TRAIN] Training EDuRec...")

        trainer, best_model_path, timer = train_model(
            model=recsys,
            dm=dm,
            debug=debug,
            epochs=train_cfg.epochs,
            patience=train_cfg.patience,
            experiment_name=dataset_experiment_name,
            monitor=recsys.monitor,
            compile=compile,
            verbose=verbose,
        )

        if debug:
            elapsed = str(datetime.datetime.now(datetime.UTC) - started_at).split(
                ".", maxsplit=1
            )[0]
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
