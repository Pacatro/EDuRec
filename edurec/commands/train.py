from pathlib import Path
from typing import Annotated
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import typer

from ..core import config
from ..core.datamodule import ELearningDataModule
from ..core.engine import UcoRecSys
from ..core.model import NeuralHybrid
from ..core.datasets import DatasetName, load_data

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
    output_model: Annotated[
        str, typer.Option("--output_model", "-o", help="Output model path")
    ] = config.OUTPUT_MODEL_PATH,
    balance: Annotated[
        bool, typer.Option("--balance", "-B", help="Balance dataset")
    ] = config.BALANCE,
    gpu: Annotated[bool, typer.Option("--gpu", "-g", help="Use GPU")] = False,
):
    df = load_data(dataset)

    droped_interactions = df[df["user_id"] == 292680].sample(n=5)
    df = df.drop(droped_interactions.index.to_list())
    droped_interactions.to_csv("./data/predict_df.csv")
    dm = ELearningDataModule(
        df,
        target=target,
        batch_size=batch_size,
        balance=balance,
    )

    dm.setup("fit")

    model = NeuralHybrid(
        n_users=dm.num_users,
        n_items=dm.num_items,
        cont_features=dm.cont_features,
        cat_cardinalities=dm.cat_cardinalities,
    )

    if config.state["verbose"]:
        # print(f"[TRAIN] Dataset {dataset}:\n{dm.df}\n")
        print(f"[TRAIN] Dataset {dataset} sparsity: {dm.sparsity}")
        print(f"[TRAIN] Dataset {dataset} threshold: {dm.threshold}")
        print(f"[TRAIN] Dataset {dataset} lenght: {len(dm.df)}")
        # print(f"[TRAIN] Train dataset:\n{dm.train_dataset.df}\n")
        print(f"[TRAIN] Model:\n{model}\n")

    recsys = UcoRecSys(
        model=model,
        top_k=top_k,
        threshold=dm.threshold,
        lr=lr,
    )

    early_stop = EarlyStopping(
        monitor="val/MSE",
        patience=config.PATIENCE,
        mode="min",
        min_delta=config.DELTA,
        verbose=True,
    )

    checkpoint = ModelCheckpoint(
        monitor="val/MSE", mode="min", save_top_k=1, filename="best-model"
    )

    device = "auto" if gpu else "cpu"

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=device,
        devices="auto",
        callbacks=[early_stop, checkpoint],
        log_every_n_steps=10,
    )

    trainer.fit(recsys, datamodule=dm)

    dm.setup("test")
    trainer.test(model=recsys, datamodule=dm)

    # Guardar ruta del mejor modelo
    best_path = checkpoint.best_model_path

    Path(best_path).rename(output_model)
    print(f"Modelo entrenado guardado en: {output_model}")
