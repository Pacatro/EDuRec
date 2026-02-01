from typing import Annotated

import lightning as L
import torch
import typer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .. import config
from ..datasets import DatasetName
from ..training.datamodule import ElearningDataModule
from ..training.engine import RecSys
from ..training.io import save_model
from ..training.model import EDuRecConfig, EDuRecMTL

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
    alpha: Annotated[
        float, typer.Option("--alpha", "-a", help="Alpha value")
    ] = config.ALPHA,
    val_size: Annotated[
        float, typer.Option("--val_size", "-v", help="Validation size")
    ] = config.VAL_SIZE,
    test_size: Annotated[
        float, typer.Option("--test_size", "-t", help="Test size")
    ] = config.TEST_SIZE,
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
    save_data: Annotated[
        bool, typer.Option("--save_data", "-P", help="Save data")
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
        test_size=test_size,
        val_size=val_size,
        save_data=save_data,
        random_state=config.state["random_state"],
    )

    model_config = EDuRecConfig(
        n_users=dm.num_users,
        n_items=dm.num_items,
        numeric_features=dm.numeric_features,
        cat_cardinalities=dm.cat_cardinalities,
    )

    model = EDuRecMTL(model_config)

    if config.state["verbose"]:
        print(f"[TRAIN] Dataset {dataset.value} sparsity: {dm.sparsity}")
        print(f"[TRAIN] Training model: {model.__class__.__name__}")
        print(f"[TRAIN] Using logger: {use_logger}")
        print(f"[TRAIN] Min rating: {dm.min_rating}")
        print(f"[TRAIN] Max rating: {dm.max_rating}")
        print(f"[TRAIN] Monitoring: {monitor}")

    # SmoothL1Loss parece mas interasante que MSE
    recsys = RecSys(
        model=model,
        top_k=top_k,
        lr=lr,
        monitor=monitor,
        alpha=alpha,
        rating_loss_fn=torch.nn.SmoothL1Loss(),
    )

    # Compile model for better performance
    torch.compile(recsys)

    early_stop_model = EarlyStopping(
        monitor=monitor,
        patience=config.PATIENCE,
        mode="min",
        min_delta=config.DELTA,
        verbose=True,
    )

    checkpoint_model = ModelCheckpoint(
        monitor=monitor,
        mode="min",
        save_top_k=1,
        filename="best_model",
    )

    train_logger = (
        WandbLogger(
            project=config.EXPERIMENT_NAME, name=f"train_{model.__class__.__name__}"
        )
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
        enable_model_summary=config.state["verbose"],
    )

    trainer.fit(recsys, datamodule=dm)

    if debug:
        print("Debug mode enabled. Skipping evaluation.")
        return

    trainer.test(model=recsys, datamodule=dm)

    # Save best model path
    if save:
        save_model(
            model.__class__.__name__,
            model_config,
            checkpoint_model.best_model_path,
            models_folder,
            dataset.value,
        )
