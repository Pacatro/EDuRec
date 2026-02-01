from typing import Annotated
from pathlib import Path

import pandas as pd
import typer

from .. import config
from ..datasets import DatasetName, load_data
from ..evaluation import CVType, cross_validate, sota_cross_validate
from ..evaluation.stats import friedman_test
from ..training.model import EDuRecV1

app = typer.Typer(no_args_is_help=True)


@app.command(
    help="Evaluates the performance of the proposed model in the top-k recommendations"
)
def eval(
    eval_sota: Annotated[
        bool,
        typer.Option(
            "--eval-sota",
            "-S",
            help="Also evaluate the performance of the SOTA models.",
        ),
    ] = False,
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ] = DatasetName.MARS,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Batch size for training.")
    ] = config.BATCH_SIZE,
    top_k: Annotated[
        int, typer.Option("--top-k", "-k", help="Top-k value")
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
    monitor: Annotated[
        str,
        typer.Option("--monitor", "-m", help="Metric to monitor for early stopping."),
    ] = config.MONITOR,
    cv_type: Annotated[
        CVType,
        typer.Option("--cv-type", help="Cross-validation type."),
    ] = CVType.KFOLD,
    results_folder: Annotated[
        str, typer.Option("--results-folder", help="Folder to save results.")
    ] = config.RESULTS_FOLDER,
):
    models_classes = (
        [
            "BPR",
            # "DeepFM",
            # EDuRecV1,
            # "NeuMF",
            # "xDeepFM",
        ]
        if eval_sota
        else [EDuRecV1]
    )
    df = load_data(dataset)

    results_path = Path(results_folder)
    results_path.mkdir(parents=True, exist_ok=True)

    for model_class in models_classes:
        model_name = (
            model_class.__name__ if not isinstance(model_class, str) else model_class
        )
        if config.state["verbose"]:
            print(
                f"[EVAL] Model: {model_name} | Dataset: {dataset.value} | top-{top_k} | CV type: {cv_type.value}"
            )

        model_run_name = (
            f"{model_name}_{cv_type.value}_k={n_splits}_{dataset.value}_top-{top_k}"
        )

        params = {
            "df": df,
            "model_class": model_class,
            "n_splits": n_splits,
            "epochs": epochs,
            "cv_type": cv_type,
            "batch_size": batch_size,
            "top_k": top_k,
            "patience": patience,
            "delta": delta,
            "monitor": monitor,
            "verbose": config.state["verbose"],
        }

        avg_metrics = (
            cross_validate(**params)
            if not isinstance(model_class, str)
            else sota_cross_validate(**params)
        )

        model_results_path = results_path / f"{model_run_name}.csv"
        avg_metrics.to_csv(model_results_path)

        if config.state["verbose"]:
            print(f"Resultados guardados en {model_results_path}")


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
