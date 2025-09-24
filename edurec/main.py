from pathlib import Path
from typing import Annotated
import typer

from edurec.core import config
from .commands.train import app as train_app
from .commands.eval import app as eval_app
from .commands.predict import app as predict_app


app = typer.Typer(
    no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]}
)
app.add_typer(train_app)
app.add_typer(eval_app)
app.add_typer(predict_app)


@app.callback()
def main(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose mode"),
    ] = False,
):
    if verbose:
        config.state["verbose"] = True

    folders = [Path(config.RESULTS_FOLDER), Path(config.RESULTS_FOLDER) / "stats"]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    # # Modo SURPRISE
    # if args.surprise:
    #     surprise_eval(
    #         df=df,
    #         dataset=args.dataset,
    #         cv_type=args.cvtype,
    #         n_splits=args.k_splits,
    #         min_rating=df[config.TARGET_COL].min(),
    #         max_rating=df[config.TARGET_COL].max(),
    #         k=args.top_k,
    #         results_folder=config.RESULTS_FOLDER,
    #         target=config.TARGET_COL,
    #         seeds=args.seeds,
    #     )

    # # Modo Test estadísticos
    # if args.stats_test:
    #     get_stats_tests(top_k=args.top_k, verbose=args.verbose)


if __name__ == "__main__":
    app()
