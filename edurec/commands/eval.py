from typing import Annotated
import pandas as pd
import typer


from ..core import config
from ..core.datasets import DatasetName, load_data
from ..core.evaluation import cross_validate, CVType
from ..core.model import EDuRec
from ..core.stats import friedman_test


app = typer.Typer(no_args_is_help=True)


@app.command(
    help="Evaluates the performance of the proposed model in the top-k recommendations"
)
def eval(
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ] = DatasetName.MARS,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Batch size for training.")
    ] = config.BATCH_SIZE,
    top_k: Annotated[
        int, typer.Option("--top_k", "-k", help="Top-k value")
    ] = config.TOP_K,
    epochs: Annotated[
        int, typer.Option("--epochs", "-e", help="Number of epochs for training.")
    ] = config.EPOCHS,
    n_splits: Annotated[
        int,
        typer.Option("--n-splits", "-n", help="Number of splits for cross-validation."),
    ] = config.K,
    patience: Annotated[
        int, typer.Option("--patience", "-p", help="Patience for early stopping.")
    ] = config.PATIENCE,
    delta: Annotated[
        float,
        typer.Option(
            "--delta",
            help="Minimum change to qualify as an improvement for early stopping.",
        ),
    ] = config.DELTA,
    cv_type: Annotated[
        CVType,
        typer.Option("--cv-type", help="Cross-validation type."),
    ] = CVType.kfold,
    results_folder: Annotated[
        str, typer.Option("--results-folder", help="Folder to save results.")
    ] = config.RESULTS_FOLDER,
    use_logger: Annotated[
        bool, typer.Option("--use_logger", "-L", help="Use MLFlow logger")
    ] = False,
):
    df = load_data(dataset)

    if config.state["verbose"]:
        print(f"[EVAL] Dataset: {dataset.name} | top-{top_k} | CV type: {cv_type.name}")

    avg_metrics = cross_validate(
        df=df,
        model_class=EDuRec,
        n_splits=n_splits,
        random_state=42,
        epochs=epochs,
        cv_type=cv_type,
        batch_size=batch_size,
        top_k=top_k,
        patience=patience,
        delta=delta,
        # use_logger=use_logger,
        verbose=config.state["verbose"],
    )

    results_path = f"{results_folder}/metrics_{cv_type}_k={n_splits}_{dataset.value}_top-{top_k}.csv"
    avg_metrics.to_csv(results_path)

    if config.state["verbose"]:
        print(f"Resultados guardados en {results_path}")


@app.command("stats", help="Performs statistical tests to compare model performances.")
def stats(
    top_k: Annotated[
        int, typer.Option("--top_k", "-k", help="Top-k value")
    ] = config.TOP_K,
):
    datasets = ["mars", "itm"]
    files = [
        f"./results/metrics_kfold_k=5_mars_top-{top_k}.csv",
        f"./results/metrics_kfold_k=5_itm_top-{top_k}.csv",
    ]

    models = [
        "Modelo Propuesto",
    ]

    stats_results = {dataset: {"p_value": 0, "stat": 0} for dataset in datasets}

    for dataset in datasets:
        stat, p = friedman_test(
            files, models, dataset, top_k, verbose=config.state["verbose"]
        )
        print(f"Results for dataset {dataset} in top-{top_k} are:")
        print(f"Stat: {stat}, p: {p}")
        stats_results[dataset]["p_value"] = p
        stats_results[dataset]["stat"] = stat

    stastics_path = f"{config.RESULTS_FOLDER}/stats/stats_{top_k}.csv"
    pd.DataFrame.from_dict(stats_results, orient="index").to_csv(stastics_path)
