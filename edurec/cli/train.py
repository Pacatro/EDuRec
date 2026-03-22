from typing import Annotated, cast

import lightning as L
import typer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .. import config
from ..datasets import DatasetName, ElearningDataModule
from ..recsys.arquitecture import GhostConfig
from ..recsys.engine import RecSys
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
    monitor: Annotated[
        str, typer.Option("--monitor", "-m", help="Monitor metric")
    ] = config.MONITOR,
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
    )

    if config.state["verbose"]:
        print(f"[TRAIN] Dataset {dataset.value} sparsity: {dm.sparsity}")
        print(f"[TRAIN] Number of users: {dm.num_users}")
        print(f"[TRAIN] Number of items: {dm.num_items}")

    cfg = GhostConfig(
        user_dim=dm.num_user_feats, item_dim=dm.num_item_feats, hidden_dim=64
    )
    assert dm.u_static is not None and dm.i_static is not None

    recsys = RecSys(
        cfg=cfg,
        inter_graph=dm.create_inter_graph(),
        u_static=dm.u_static,
        i_static=dm.i_static,
        lr=lr,
        top_k=top_k,
    )

    model_name = recsys.model_name

    # recsys = torch.compile(recsys) if not debug else recsys

    # Callbacks y Loggers
    early_stop_model = EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode="max",
        min_delta=config.DELTA,
        verbose=True,
    )
    checkpoint_model = ModelCheckpoint(
        monitor=monitor, mode="max", save_top_k=1, filename=f"best_{model_name}"
    )

    train_logger = (
        WandbLogger(project=config.EXPERIMENT_NAME, name=f"train_{model_name}")
        if use_logger and not debug
        else None
    )

    trainer = L.Trainer(
        logger=train_logger,
        max_epochs=epochs,
        accelerator=config.state["device"],
        devices="auto",
        log_every_n_steps=10,
        callbacks=[early_stop_model, checkpoint_model],
        fast_dev_run=debug,
    )

    trainer.fit(recsys, datamodule=dm)

    if debug:
        print("[TRAIN] Debug mode: Skipping evaluation")
        return

    test_results = trainer.test(model=recsys, datamodule=dm)

    if save:
        save_model(
            model_name,
            cfg,
            checkpoint_model.best_model_path,
            models_folder,
            dataset.value,
            cast(dict[str, float], test_results[0]),
        )
