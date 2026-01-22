from typing import Annotated

import mlflow
import pandas as pd
import typer

from .. import config
from ..data.datasets import DatasetName
from ..evaluation.cross_validation import CVType, cross_validate
from ..evaluation.stats import friedman_test
from ..training.model import MF, EDuRecV1, NeuralMF

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
    min_topk: Annotated[
        int, typer.Option("--min-topk", "-k", help="Minimum top-k value.")
    ] = 10,
    max_topk: Annotated[
        int, typer.Option("--max-topk", "-K", help="Maximum top-k value.")
    ] = 20,
    topk_step: Annotated[
        int, typer.Option("--topk-step", "-s", help="Top-k step.")
    ] = 5,
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
):
    models_classes = [
        EDuRecV1,
        MF,
        NeuralMF,
    ]

    mlflow.set_experiment(f"{config.EXPERIMENT_NAME}_CV")

    for top_k in range(min_topk, max_topk + 1, topk_step):
        for model_class in models_classes:
            if config.state["verbose"]:
                print(
                    f"[EVAL] Model: {model_class.__name__} | Dataset: {dataset.name} | top-{top_k} | CV type: {cv_type.name}"
                )

            model_run_name = f"{model_class.__name__}_{cv_type.name}_k={n_splits}_{dataset.name}_top-{top_k}"

            with mlflow.start_run(run_name=model_run_name):
                mlflow.log_params(
                    {
                        "model": model_class.__name__,
                        "top-k": top_k,
                    }
                )

                avg_metrics = cross_validate(
                    dataset_name=dataset,
                    model_class=model_class,
                    n_splits=n_splits,
                    epochs=epochs,
                    cv_type=cv_type,
                    batch_size=batch_size,
                    top_k=top_k,
                    patience=patience,
                    delta=delta,
                    verbose=config.state["verbose"],
                )

                metrics_dict = {}
                for metric_name, row in avg_metrics.iterrows():
                    metrics_dict[f"{metric_name}"] = float(row["mean"])
                    # metrics_dict[f"{metric_name}_std"] = float(row["std"])

                metrics_dict["top_k"] = top_k
                mlflow.log_metrics(metrics_dict)

                results_path = f"{results_folder}/{model_run_name}.csv"
                avg_metrics.to_csv(results_path)

                mlflow.log_artifact(results_path)

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
