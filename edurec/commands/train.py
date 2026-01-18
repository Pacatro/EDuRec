from typing import Annotated
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import typer

from ..core import config
from ..core.data import ElearningDataModule

# from ..core.engine import RecSys
# from ..core.model import EDuRecV1
from ..core.datasets import DatasetName, load_data
from ..core.model_io import save_best_model

app = typer.Typer(no_args_is_help=True)


@app.command(help="Train the recommendation model.")
def train(
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ] = DatasetName.MARS,
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
    dm = ElearningDataModule(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        test_size=0.2,
        val_size=0.2,
        threshold=0.5,
        random_state=42,
    )

    dm.setup()

    print(dm.train_df)
