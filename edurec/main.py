import warnings
from enum import StrEnum
from typing import Annotated

import typer

from edurec import settings
from edurec.cli import (
    ablation_app,
    dataset_app,
    eval_app,
    optim_app,
    train_app,
)

# Ignore pandas future warnings
warnings.filterwarnings("ignore", category=FutureWarning)


app = typer.Typer(
    rich_markup_mode=None,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(train_app)
app.add_typer(dataset_app)
app.add_typer(eval_app)
app.add_typer(optim_app)
app.add_typer(ablation_app)


class Device(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


DEFAULT_DEVICE = Device(settings.state["device"])


@app.callback()
def main(
    device: Annotated[
        Device,
        typer.Option("--device", "-d", help="Device to use"),
    ] = DEFAULT_DEVICE,
    random_state: Annotated[
        int | None,
        typer.Option("--random-state", "-r", help="Random state"),
    ] = settings.state["random_state"],
    max_users: Annotated[
        int | None,
        typer.Option(
            "--max-users",
            min=1,
            help="Sample this many complete user histories; omit for the full dataset.",
        ),
    ] = None,
    selection_seed: Annotated[
        int,
        typer.Option(
            "--selection-seed", help="User selection seed, independent of training."
        ),
    ] = 42,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose mode"),
    ] = settings.state["verbose"],
):
    settings.SELECTION_MAX_USERS = max_users
    settings.SELECTION_SEED = selection_seed
    settings.state["verbose"] = verbose
    settings.seed_everything(random_state)
    settings.state["device"] = device.value

    if verbose:
        print(f"[CONFIG] Device: {device.value}")
        print(f"[CONFIG] Random state: {random_state}")


if __name__ == "__main__":
    app()
