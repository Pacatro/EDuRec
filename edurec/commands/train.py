from typing import Annotated
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import typer

from ..core import config
from ..core.datamodule import ELearningDataModule
from ..core.engine import RecSys
from ..core.model import MF, EDuRec
from ..core.datasets import DatasetName, load_data
from ..core.model_io import save_best_model

app = typer.Typer(no_args_is_help=True)


@app.command(help="Train the recommendation model.")
def train(
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ] = DatasetName.mars,
    target: Annotated[
        str, typer.Option("--target", "-t", help="Target column")
    ] = config.TARGET_COL,
    epochs: Annotated[
        int, typer.Option("--epochs", "-e", help="Number of epochs")
    ] = config.EPOCHS,
    lr: Annotated[float, typer.Option("--lr", "-l", help="Learning rate")] = config.LR,
    batch_size: Annotated[
        int, typer.Option("--batch_size", "-b", help="Batch size")
    ] = config.BATCH_SIZE,
    top_k: Annotated[
        int, typer.Option("--top_k", "-k", help="Top-k value")
    ] = config.TOP_K,
    # balance: Annotated[
    #     bool, typer.Option("--balance", "-B", help="Balance dataset")
    # ] = config.BALANCE,
    use_logger: Annotated[
        bool, typer.Option("--use_logger", "-L", help="Use MLFlow logger")
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", "-D", help="Debug mode")] = False,
    save_model: Annotated[
        bool, typer.Option("--save_model", "-S", help="Save model")
    ] = False,
    models_folder: Annotated[
        str,
        typer.Option(
            "--models-folder", "-M", help="Folder where save the trained model."
        ),
    ] = config.MODELS_FOLDER,
):
    df = load_data(dataset)

    droped_interactions = df[df["user_id"] == 292680].sample(n=10, random_state=1)
    df = df.drop(droped_interactions.index.to_list())
    droped_interactions.to_csv("./data/predict_df.csv")

    dm = ELearningDataModule(
        df,
        target=target,
        batch_size=batch_size,
        # balance=balance,
    )

    models = [
        EDuRec(
            n_users=dm.num_users,
            n_items=dm.num_items,
            cont_features=dm.cont_features,
            cat_cardinalities=dm.cat_cardinalities,
        ),
        MF(n_users=dm.num_users, n_items=dm.num_items),
    ]

    if config.state["verbose"]:
        print(f"[TRAIN] Dataset {dataset} sparsity: {dm.sparsity}")
        print(f"[TRAIN] Dataset {dataset} threshold: {dm.threshold}")
        print(f"[TRAIN] Dataset {dataset} lenght: {len(dm.df)}")

    for model in models:
        if config.state["verbose"]:
            print(f"[TRAIN] Training model: {model.__class__.__name__}")
            print(f"[TRAIN] Using logger: {use_logger}")

        # Crear callbacks únicos para cada modelo
        early_stop_model = EarlyStopping(
            monitor="val/MSE",
            patience=config.PATIENCE,
            mode="min",
            min_delta=config.DELTA,
            verbose=True,
        )

        checkpoint_model = ModelCheckpoint(
            monitor="val/MSE",
            # dirpath=f"./checkpoints/{model.__class__.__name__}",
            mode="min",
            save_top_k=1,
            filename="best_model",
        )

        train_logger = (
            MLFlowLogger(
                experiment_name="edurec",
                run_name=f"{model.__class__.__name__}",
                tracking_uri="file:./mlruns",
                # log_model=True,
                # prefix=f"{model.__class__.__name__}-",
                # artifact_location="checkpoints",
            )
            if use_logger and not debug
            else None
        )

        recsys = RecSys(
            model=model,
            top_k=top_k,
            threshold=dm.threshold,
            lr=lr,
        )

        trainer = L.Trainer(
            logger=train_logger,
            max_epochs=epochs,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10,
            callbacks=[early_stop_model, checkpoint_model],
            fast_dev_run=debug,
            enable_model_summary=config.state["verbose"],
        )

        trainer.fit(recsys, datamodule=dm)

        if debug:
            print("Debug mode enabled. Skipping evaluation.")
            continue  # Usar continue en lugar de return

        dm.setup("test")
        trainer.test(model=recsys, datamodule=dm)

        # Save best model path
        if save_model:
            save_best_model(
                model.__class__.__name__,
                checkpoint_model.best_model_path,  # Usar el checkpoint específico
                models_folder,
            )

        # Finalizar el logger si existe
        if train_logger is not None:
            train_logger.finalize("success")
