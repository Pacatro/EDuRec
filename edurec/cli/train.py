from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule
from ..recsys import EDuRecConfig, RecSys, optimize_model, train_model
from ..recsys.io import save_model
from .utils import (
    build_config,
    dataset_config_path,
    dataset_run_name,
    datasets_to_run,
    print_data_summary,
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
    optimize: Annotated[
        bool,
        typer.Option(
            "--optimize",
            "-o",
            help="Optimize hyperparameters before the final training run.",
        ),
    ] = False,
    n_trials: Annotated[
        int,
        typer.Option(
            "--trials",
            "-n",
            min=1,
            help="Number of hyperparameter optimization trials.",
        ),
    ] = settings.OPTIM_N_TRIALS,
    use_processed_data: Annotated[
        bool, typer.Option("--use_processed", "-P")
    ] = settings.SAVE_DATA,
    models_folder: Annotated[
        str, typer.Option("--models-folder", "-M")
    ] = settings.MODELS_FOLDER,
    configs_folder: Annotated[
        str, typer.Option("--configs-folder", "-C")
    ] = settings.CONFIGS_FOLDER,
    experiment_name: Annotated[
        str | None, typer.Option("--experiment-name", "-E")
    ] = None,
) -> None:
    started_at = datetime.now()
    verbose = settings.state["verbose"]
    monitor_metric = f"val/ndcg@{top_k}"
    optimization_root = (
        Path(settings.RESULTS_FOLDER)
        / "optimization"
        / started_at.strftime("%Y%m%d_%H%M%S")
    )

    datasets = datasets_to_run(dataset)

    for dataset_idx, dataset in enumerate(datasets, start=1):
        run_name = dataset_run_name(dataset, limit)
        dataset_experiment_name = (
            f"{experiment_name}_{run_name}" if experiment_name else None
        )
        print("\n[TRAIN] Training run")
        print(f"[TRAIN] Dataset {dataset_idx}/{len(datasets)}: {run_name}")
        print("[TRAIN] Model: EDuRec")
        print(f"[TRAIN] Monitor: {monitor_metric}")
        print(f"[TRAIN] Save model: {save}")
        print(f"[TRAIN] Optimize hyperparameters: {optimize}")
        print("[TRAIN] Preparing data...")

        if verbose:
            print(
                "[TRAIN] Config: "
                f"epochs={epochs}, lr={lr}, batch_size={batch_size}, "
                f"patience={patience}, adaptive_k={adaptive_k}, debug={debug}, "
                f"optimize={optimize}, trials={n_trials}"
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
            dataset=dataset,
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

        if verbose:
            if use_processed_data and dm.is_processed:
                print(
                    f"[TRAIN] Using saved processed data from {settings.PROCESSED_FOLDER}"
                )
            else:
                print(
                    f"[TRAIN] Processing raw data from {settings.DATA_FOLDER}/raw/{dataset.value}"
                )

        dm.setup()

        print_data_summary("TRAIN", dm)

        config_path = dataset_config_path(configs_folder, dataset, limit)

        base_cfg = build_config(
            dm,
            lr=lr,
            adaptive_k=adaptive_k,
            topks=settings.TOP_KS,
        )

        if optimize:
            dataset_results_path = optimization_root / run_name
            print(f"[TRAIN] Optimizing {n_trials} trials...")
            study = optimize_model(
                base_config=base_cfg,
                dm=dm,
                n_trials=n_trials,
                epochs=epochs,
                patience=patience,
                val_topk=top_k,
                verbose=verbose,
                results_path=dataset_results_path,
            )
            cfg = EDuRecConfig(**study.best_trial.user_attrs["config"])
            config_path.parent.mkdir(parents=True, exist_ok=True)
            cfg.save(config_path)
            print(
                f"[TRAIN] Best optimization score: {study.best_value} "
                f"(trial {study.best_trial.number})"
            )
            print(f"[TRAIN] Best parameters: {study.best_params}")
            print(f"[TRAIN] Configuration saved: {config_path}")
        elif config_path.exists():
            cfg = replace(
                EDuRecConfig.load(config_path),
                num_users=dm.num_users,
                num_items=dm.num_items,
                num_ctx_feats=dm.train_ds.num_ctx_feats,
                num_user_dense_feats=dm.num_user_dense_feats,
                num_item_dense_feats=dm.num_item_dense_feats,
                num_user_text_feats=dm.num_user_text_feats,
                num_item_text_feats=dm.num_item_text_feats,
                user_cat_cardinalities=dm.user_cat_cardinalities,
                item_cat_cardinalities=dm.item_cat_cardinalities,
                adaptive_k=adaptive_k,
                topks=settings.TOP_KS,
            )
            print(f"[TRAIN] Using saved configuration: {config_path}")
        else:
            cfg = base_cfg
            print("[TRAIN] No saved configuration found; using the full model.")

        recsys = RecSys(
            cfg=cfg,
            inter_graph=dm.build_inter_graph(),
            u_static_feats=dm.u_static_feats,
            i_static_feats=dm.i_static_feats,
            user_stats=dm.user_stats,
            item_stats=dm.item_stats,
        )

        print("[TRAIN] Training EDuRec...")

        trainer, best_model_path = train_model(
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
            elapsed = str(datetime.now() - started_at).split(".", maxsplit=1)[0]
            print("[TRAIN] Debug mode: skipping evaluation")
            print(f"[TRAIN] Finished {dataset.value} in {elapsed}\n")
            return

        metrics = trainer.test(ckpt_path="best", datamodule=dm, weights_only=False)[0]
        print(f"[TRAIN] Best checkpoint: {best_model_path}")

        if save and trainer.is_global_zero:
            model_file_path, model_config_path, metrics_path = save_model(
                model_config=cfg,
                dataset_name=run_name,
                best_model_path=best_model_path,
                models_folder=models_folder,
                metrics=metrics,
            )

            print(f"[TRAIN] Model weights saved: {model_file_path}")
            print(f"[TRAIN] Model config saved: {model_config_path}")
            print(f"[TRAIN] Metrics saved: {metrics_path}")

        elapsed = str(datetime.now() - started_at).split(".", maxsplit=1)[0]
        print(f"[TRAIN] Finished {dataset.value} in {elapsed}\n")
