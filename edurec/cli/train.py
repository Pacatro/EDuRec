from typing import Annotated

import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule
from ..recsys import train_model
from ..recsys.architecture.ghost import GhostConfig
from ..recsys.io import save_model
from ..recsys.ranker import Ranker

app = typer.Typer(no_args_is_help=True)


@app.command(name="train", help="Train the reranker model.")
def train_ranker(
    dataset: Annotated[DatasetName, typer.Option("--dataset", "-d")] = DatasetName.MARS,
    epochs: Annotated[int, typer.Option("--epochs", "-e")] = settings.EPOCHS,
    lr: Annotated[float, typer.Option("--lr", "-l")] = settings.RANKER_LR,
    batch_size: Annotated[
        int, typer.Option("--batch_size", "-b")
    ] = settings.RANKER_BATCH_SIZE,
    patience: Annotated[
        int, typer.Option("--patience", "-p")
    ] = settings.RANKER_PATIENCE,
    val_size: Annotated[float, typer.Option("--val_size", "-v")] = settings.VAL_RATIO,
    test_size: Annotated[
        float, typer.Option("--test_size", "-t")
    ] = settings.TEST_RATIO,
    top_k: Annotated[int, typer.Option("--top_k", "-k")] = settings.RANKER_TOP_K,
    remove_sparse: Annotated[
        bool, typer.Option("--remove_sparse", "-R")
    ] = settings.REMOVE_SPARSE,
    min_interactions: Annotated[
        int, typer.Option("--min_interactions", "-I")
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
    if settings.state["verbose"]:
        print(f"[TRAIN] Using dataset {dataset.value}")
        print(f"[TRAIN] Removing sparse users: {remove_sparse}")
        print(f"[TRAIN] Minimum interactions per user: {min_interactions}")

        if use_processed_data:
            print(
                f"[TRAIN] Using saved processed data from {settings.PROCESSED_FOLDER}"
            )
        else:
            print(
                f"[TRAIN] Processing raw data from {settings.DATA_FOLDER}/{dataset.value}"
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

    dm.setup()

    if settings.state["verbose"]:
        print(f"[TRAIN] Dataset sparsity before preprocessing: {dm.sparsity}")
        print(f"[TRAIN] Number of users before preprocessing: {dm.num_users}")
        print(f"[TRAIN] Number of items before preprocessing: {dm.num_items}")
        print(f"[TRAIN] Number of interactions: {dm.num_interactions}")
        print(f"[TRAIN] Number of user features: {dm.num_user_feats}")
        print(f"[TRAIN] Number of item features: {dm.num_item_feats}")
        print(f"[TRAIN] Number of interactions context features: {dm.num_ctx_feats}")
        print(f"[TRAIN] Epochs: {epochs}")
        print(f"[TRAIN] Learning rate: {lr}")
        print(f"[TRAIN] Batch size: {batch_size}")
        print(f"[TRAIN] Patience: {patience}")
        print(f"[TRAIN] Use processed cache: {use_processed_data}")
        print(f"[TRAIN] Remove sparse users/items: {remove_sparse}")
        print(f"[TRAIN] Min interactions: {min_interactions}")
        print(f"[TRAIN] Validation ratio: {val_size}")
        print(f"[TRAIN] Test ratio: {test_size}")

    cfg = GhostConfig(
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

    train_graph = dm.build_inter_graph()

    ranker = Ranker(
        cfg=cfg,
        inter_graph=train_graph,
        u_static_feats=dm.u_static_feats,
        i_static_feats=dm.i_static_feats,
        lr=lr,
        val_topk=top_k,
        adaptive_k=adaptive_k,
    )

    trainer, best_model_path = train_model(
        model=ranker,
        dm=dm,
        top_k=top_k,
        debug=debug,
        use_logger=use_logger,
        epochs=epochs,
        patience=patience,
        monitor=f"val/ndcg@{top_k}",
    )

    if debug:
        print("[TRAIN] Debug mode: Skipping evaluation")
        return

    metrics = trainer.test(ckpt_path="best", datamodule=dm, weights_only=False)[0]

    if save and trainer.is_global_zero:
        model_file_path, model_config_path, metrics_path = save_model(
            model_config=cfg,
            dataset_name=dataset.value,
            best_model_path=best_model_path,
            models_folder=models_folder,
            metrics=metrics,
        )

        if settings.state["verbose"]:
            print(f"Model weights saved in: {model_file_path}")
            print(f"Model config saved in: {model_config_path}")
            print(f"Metrics saved in: {metrics_path}")
