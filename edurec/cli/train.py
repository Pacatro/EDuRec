from datetime import datetime
from typing import Annotated

import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule, dataset_loaders
from ..recsys import train_model, EDuRecConfig, RecSys
from ..recsys.io import save_model

app = typer.Typer(no_args_is_help=True)


@app.command(name="train", help="Train the reranker model.")
def train(
    dataset: Annotated[DatasetName | None, typer.Option("--dataset", "-d")] = None,
    epochs: Annotated[int, typer.Option("--epochs", "-e")] = settings.EPOCHS,
    lr: Annotated[float, typer.Option("--lr", "-l")] = settings.LR,
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
    use_logger: Annotated[bool, typer.Option("--use_logger", "-L")] = False,
    debug: Annotated[bool, typer.Option("--debug", "-D")] = False,
    save: Annotated[bool, typer.Option("--save_model", "-S")] = False,
    use_processed_data: Annotated[
        bool, typer.Option("--use_processed", "-P")
    ] = settings.SAVE_DATA,
    models_folder: Annotated[
        str, typer.Option("--models-folder", "-M")
    ] = settings.MODELS_FOLDER,
) -> None:
    started_at = datetime.now()
    monitor_metric = f"val/ndcg@{top_k}"

    datasets = [dataset] if dataset is not None else dataset_loaders.keys()

    for dataset_idx, dataset in enumerate(datasets, start=1):
        print("\n[TRAIN] Training run")
        print(f"[TRAIN] Dataset {dataset_idx}/{len(datasets)}: {dataset.value}")
        print("[TRAIN] Model: EDuRec")
        print(f"[TRAIN] Monitor: {monitor_metric}")
        print(f"[TRAIN] Save model: {save}")
        print("[TRAIN] Preparing data...")

        if settings.state["verbose"]:
            print(
                "[TRAIN] Config: "
                f"epochs={epochs}, lr={lr}, batch_size={batch_size}, "
                f"patience={patience}, adaptive_k={adaptive_k}, debug={debug}, "
                f"use_logger={use_logger}"
            )
            print(
                "[TRAIN] Data config: "
                f"use_processed={use_processed_data}, remove_sparse={remove_sparse}, "
                f"min_interactions={min_interactions}, "
                f"val_ratio={val_size}, test_ratio={test_size}"
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
        )

        if settings.state["verbose"]:
            if use_processed_data and dm.is_processed:
                print(
                    f"[TRAIN] Using saved processed data from {settings.PROCESSED_FOLDER}"
                )
            else:
                print(
                    f"[TRAIN] Processing raw data from {settings.DATA_FOLDER}/raw/{dataset.value}"
                )

        dm.setup()

        split_sizes = {
            split: len(getattr(dm.artifacts, split))
            for split in ("train", "val", "test")
            if getattr(dm.artifacts, split) is not None
        }

        print(
            "[TRAIN] Data ready: "
            f"users={dm.num_users}, items={dm.num_items}, "
            f"interactions={dm.num_interactions}, sparsity={dm.sparsity:.4f}"
        )
        print(
            "[TRAIN] Splits: "
            + ", ".join(f"{split}={size}" for split, size in split_sizes.items())
        )

        if settings.state["verbose"]:
            print(
                "[TRAIN] Features: "
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
            lr=lr,
            adaptive_k=adaptive_k,
            topks=settings.TOP_KS,
        )

        ranker = RecSys(
            cfg=cfg,
            inter_graph=dm.build_inter_graph(),
            u_static_feats=dm.u_static_feats,
            i_static_feats=dm.i_static_feats,
        )

        print("[TRAIN] Training EDuRec...")
        trainer, best_model_path = train_model(
            model=ranker,
            dm=dm,
            debug=debug,
            use_logger=use_logger,
            epochs=epochs,
            patience=patience,
            monitor=monitor_metric,
            verbose=settings.state["verbose"],
        )

        if debug:
            elapsed = str(datetime.now() - started_at).split(".", maxsplit=1)[0]
            print("[TRAIN] Debug mode: skipping evaluation")
            print(f"[TRAIN] Finished {dataset.value} in {elapsed}\n")
            return

        print("[TRAIN] Testing best checkpoint...")
        metrics = trainer.test(ckpt_path="best", datamodule=dm, weights_only=False)[0]
        elapsed = str(datetime.now() - started_at).split(".", maxsplit=1)[0]

        print("[TRAIN] Test metrics:")
        for name, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"[TRAIN]   {name}: {value:.4f}")
            else:
                print(f"[TRAIN]   {name}: {value}")
        print(f"[TRAIN] Best checkpoint: {best_model_path}")

        if save and trainer.is_global_zero:
            model_file_path, model_config_path, metrics_path = save_model(
                model_config=cfg,
                dataset_name=dataset.value,
                best_model_path=best_model_path,
                models_folder=models_folder,
                metrics=metrics,
            )

            print(f"[TRAIN] Model weights saved: {model_file_path}")
            print(f"[TRAIN] Model config saved: {model_config_path}")
            print(f"[TRAIN] Metrics saved: {metrics_path}")

        print(f"[TRAIN] Finished {dataset.value} in {elapsed}\n")
