from pathlib import Path
from typing import Annotated
import typer

from .core import config
from .cli.train import app as train_app
# from .commands.eval import app as eval_app
# from .commands.predict import app as predict_app


app = typer.Typer(
    rich_markup_mode=None,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(train_app)
# app.add_typer(eval_app)
# app.add_typer(predict_app)


@app.callback()
def main(
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

    results_folder = Path(config.RESULTS_FOLDER)

    folders = [results_folder, results_folder / "stats"]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    app()
