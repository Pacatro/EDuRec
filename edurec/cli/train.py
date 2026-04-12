from typing import Annotated, cast

import typer

from .. import config
from ..datasets import DatasetName, ElearningDataModule
from ..recsys import GhostConfig, RecSys, train_recsys
from ..recsys.io import save_model

app = typer.Typer(no_args_is_help=True)


@app.command(help="Train the recommendation model.")
def train(
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ] = DatasetName.MARS,
    epochs: Annotated[
        int, typer.Option("--epochs", "-e", help="Number of epochs")
    ] = config.EPOCHS,
    lr: Annotated[float, typer.Option("--lr", "-l", help="Learning rate")] = config.LR,
    batch_size: Annotated[
        int, typer.Option("--batch_size", "-b", help="Batch size")
    ] = config.BATCH_SIZE,
    patience: Annotated[
        int, typer.Option("--patience", "-p", help="Patience")
    ] = config.PATIENCE,
    val_size: Annotated[
        float, typer.Option("--val_size", "-v", help="Validation size")
    ] = config.VAL_RATIO,
    test_size: Annotated[
        float, typer.Option("--test_size", "-t", help="Test size")
    ] = config.TEST_RATIO,
    top_k: Annotated[
        int, typer.Option("--top_k", "-k", help="Top-k value")
    ] = config.TOP_K,
    n_neg_train: Annotated[
        int,
        typer.Option(
            "--n_neg_train", help="Number of negatives to sample for training"
        ),
    ] = config.N_NEG_TRAIN,
    n_neg_val: Annotated[
        int,
        typer.Option(
            "--n_neg_val", help="Number of negatives to sample for validation"
        ),
    ] = config.N_NEG_VAL,
    n_neg_test: Annotated[
        int,
        typer.Option("--n_neg_test", help="Number of negatives to sample for testing"),
    ] = config.N_NEG_TEST,
    remove_sparse_users: Annotated[
        bool,
        typer.Option(
            "--remove_sparse_users",
            "-R",
            help="Remove users with less than MIN_INTERACTIONS interactions",
        ),
    ] = config.REMOVE_SPARSE_USERS,
    min_interactions: Annotated[
        int,
        typer.Option(
            "--min_interactions",
            "-I",
            help="Minimum number of interactions per user",
        ),
    ] = config.MIN_INTERACTIONS,
    monitor: Annotated[
        str, typer.Option("--monitor", "-m", help="Monitor metric")
    ] = config.MONITOR,
    adaptive_k: Annotated[
        bool,
        typer.Option(
            "--adaptive_k", "-a", help="Use adaptive k to compute some metrics"
        ),
    ] = config.ADAPTIVE_K,
    use_logger: Annotated[
        bool, typer.Option("--use_logger", "-L", help="Use MLFlow logger")
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", "-D", help="Debug mode")] = False,
    save: Annotated[
        bool, typer.Option("--save_model", "-S", help="Save model")
    ] = False,
    use_procesed_data: Annotated[
        bool, typer.Option("--use_processed", "-P", help="Use saved processed data")
    ] = config.SAVE_DATA,
    models_folder: Annotated[
        str,
        typer.Option(
            "--models-folder", "-M", help="Folder where save the trained model."
        ),
    ] = config.MODELS_FOLDER,
):
    dm = ElearningDataModule(
        dataset=dataset,
        batch_size=batch_size,
        test_ratio=test_size,
        val_ratio=val_size,
        use_processed_data=use_procesed_data,
        random_state=config.state["random_state"],
        n_neg_train=n_neg_train,
        n_neg_val=n_neg_val,
        n_neg_test=n_neg_test,
        min_interactions=min_interactions,
        remove_sparse_users=remove_sparse_users,
    )

    if config.state["verbose"]:
        print(f"[TRAIN] Using dataset {dataset.value}")
        print(f"[TRAIN] Using adaptive k: {adaptive_k}")
        print(f"[TRAIN] Removing sparse users: {remove_sparse_users}")
        print(f"[TRAIN] Minimum interactions per user: {min_interactions}")

        if use_procesed_data:
            print(f"[TRAIN] Using saved processed data from {config.PROCESSED_FOLDER}")
        else:
            print(
                f"[TRAIN] Processing raw data from {config.DATA_FOLDER}/{dataset.value}"
            )

    dm.setup()

    if config.state["verbose"]:
        print(f"[TRAIN] Dataset sparsity: {dm.sparsity}")
        print(f"[TRAIN] Number of users: {dm.num_users}")
        print(f"[TRAIN] Number of items: {dm.num_items}")
        print(f"[TRAIN] Number of interactions: {dm.num_interactions}")
        print(f"[TRAIN] Number of user features: {dm.num_user_feats}")
        print(f"[TRAIN] Number of item features: {dm.num_item_feats}")
        print(f"[TRAIN] Number of interactions context features: {dm.num_ctx_feats}")
        print(f"[TRAIN] Negatives samples for training: {dm.n_neg_train}")
        print(f"[TRAIN] Negatives samples for validation: {dm.n_neg_val}")
        print(f"[TRAIN] Negatives samples for testing: {dm.n_neg_test}")

    cfg = GhostConfig(
        num_users=dm.num_users,
        num_items=dm.num_items,
        num_ctx_feats=dm.train_ds.num_ctx_feats,
        num_user_dense_feats=dm.num_user_dense_feats,
        num_item_dense_feats=dm.num_item_dense_feats,
        user_cat_cardinalities=dm.user_cat_cardinalities,
        item_cat_cardinalities=dm.item_cat_cardinalities,
    )

    assert dm.u_static_feats is not None and dm.i_static_feats is not None

    recsys = RecSys(
        cfg=cfg,
        inter_graph=dm.create_inter_graph(),
        u_static_feats=dm.u_static_feats,
        i_static_feats=dm.i_static_feats,
        lr=lr,
        top_k=top_k,
        adaptive_k=adaptive_k,
    )

    trainer, best_model_path = train_recsys(
        recsys=recsys,
        dm=dm,
        top_k=top_k,
        debug=debug,
        use_logger=use_logger,
        epochs=epochs,
        patience=patience,
        monitor=monitor,
    )

    if debug:
        print("[TRAIN] Debug mode: Skipping evaluation")
        return

    test_results = trainer.test(
        ckpt_path="best",
        datamodule=dm,
        weights_only=False,
    )[0]

    if save:
        model_file_path, model_config_path, metrics_path = save_model(
            model_name=recsys.model_name,
            model_config=cfg,
            dataset_name=dataset.value,
            best_model_path=best_model_path,
            models_folder=models_folder,
            metrics=cast(dict[str, float], test_results),
        )

        if config.state["verbose"]:
            print(f"Model weights saved in: {model_file_path}")
            print(f"Model config saved in: {model_config_path}")
            print(f"Metrics saved in: {metrics_path}")
