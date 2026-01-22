from pathlib import Path
from typing import Annotated
import typer
from enum import Enum

from . import config
from .cli.train import app as train_app
from .cli.eval import app as eval_app
# from .commands.predict import app as predict_app


app = typer.Typer(
    rich_markup_mode=None,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(train_app)
app.add_typer(eval_app)
# app.add_typer(predict_app)


class Device(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


@app.callback()
def main(
    device: Annotated[
        Device,
        typer.Option("--device", "-d", help="Device to use"),
    ] = Device.AUTO,
    random_state: Annotated[
        int | None,
        typer.Option("--random-state", "-r", help="Random state"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose mode"),
    ] = False,
):
    config.state["verbose"] = verbose
    config.state["random_state"] = random_state
    config.state["device"] = device.value

    if verbose:
        print(f"[CONFIG] Device: {device}")
        print(f"[CONFIG] Random state: {random_state}")

    results_folder = Path(config.RESULTS_FOLDER)

    folders = [results_folder, results_folder / "stats"]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    app()
