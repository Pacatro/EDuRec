import warnings
from enum import StrEnum
from typing import Annotated

import typer

from . import settings
from .cli import dataset_app, eval_app, test_app, train_app, optim_app

# Ignore pandas future warnings
warnings.filterwarnings("ignore", category=FutureWarning)


app = typer.Typer(
    rich_markup_mode=None,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(train_app)
app.add_typer(test_app)
app.add_typer(dataset_app)
app.add_typer(eval_app)
app.add_typer(optim_app)


class Device(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


@app.callback()
def main(
    device: Annotated[
        Device,
        typer.Option("--device", "-d", help="Device to use"),
    ] = Device(settings.state["device"]),
    random_state: Annotated[
        int | None,
        typer.Option("--random-state", "-r", help="Random state"),
    ] = settings.state["random_state"],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose mode"),
    ] = settings.state["verbose"],
):
    settings.state["verbose"] = verbose
    settings.seed_everything(random_state)
    settings.state["device"] = device.value

    if verbose:
        print(f"[CONFIG] Device: {device.value}")
        print(f"[CONFIG] Random state: {random_state}")


if __name__ == "__main__":
    app()
