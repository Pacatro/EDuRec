import warnings
from enum import Enum
from typing import Annotated

import typer
import torch

from . import config
from .cli import eval_app, train_app, train_comp_app

# Ignore pandas future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set a default seed for reproducibility
torch.manual_seed(config.state["random_state"])

app = typer.Typer(
    rich_markup_mode=None,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(train_app)
app.add_typer(eval_app)
app.add_typer(train_comp_app)
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
    ] = Device(config.state["device"]),
    random_state: Annotated[
        int | None,
        typer.Option("--random-state", "-r", help="Random state"),
    ] = config.state["random_state"],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose mode"),
    ] = config.state["verbose"],
):
    config.state["verbose"] = verbose
    config.state["random_state"] = random_state
    config.state["device"] = device.value

    if verbose:
        print(f"[CONFIG] Device: {device.value}")
        print(f"[CONFIG] Random state: {random_state}")


if __name__ == "__main__":
    app()
