from pathlib import Path
from typing import Annotated

import typer
from .core import config

app = typer.Typer(
    rich_markup_mode=None,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# app.add_typer(train_app)
# app.add_typer(eval_app)
# app.add_typer(predict_app)


@app.callback()
def main(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose mode"),
    ] = False,
):
    config.state["verbose"] = verbose

    results_folder = Path(config.RESULTS_FOLDER)

    folders = [results_folder, results_folder / "stats"]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    app()
