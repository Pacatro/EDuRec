from typing import Annotated
import pandas as pd
import typer
from surprise import (
    NMF,
    SVD,
    BaselineOnly,
    CoClustering,
    Dataset,
    KNNBaseline,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    NormalPredictor,
    Reader,
    SlopeOne,
    SVDpp,
)

from ..core import config
from ..core.datasets import DatasetName, load_data
from ..core.evaluation import cross_validate, CVType
from ..core.model import NeuralHybrid
from ..core.stats import friedman_test
from ..core.surprise_eval import cross_validation, preprocess_ratings


app = typer.Typer(no_args_is_help=True)


@app.command(
    help="Evaluates the performance of the proposed model and state-of-the-art models in the top-k recommendations"
)
def eval_model(
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ] = DatasetName.mars,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Batch size for training.")
    ] = 32,
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
):
    df = load_data(dataset)
    avg_metrics = cross_validate(
        df=df,
        model_class=NeuralHybrid,
        n_splits=n_splits,
        random_state=42,
        epochs=epochs,
        cv_type=cv_type,
        batch_size=batch_size,
        top_k=top_k,
        patience=patience,
        delta=delta,
        verbose=config.state["verbose"],
    )

    results_path = f"{results_folder}/metrics_{cv_type}_k={n_splits}_{dataset.value}_top-{top_k}.csv"
    avg_metrics.to_csv(results_path)

    if config.state["verbose"]:
        print(f"Resultados guardados en {results_path}")


def surprise_eval(
    df: pd.DataFrame,
    dataset: str,
    k: int,
    n_splits: int = 5,
    target: str = "rating",
    min_rating: int = 1,
    max_rating: int = 10,
    cv_type: CVType = CVType.kfold,
    results_folder: str = "results",
    seeds: list[int] = [42],
):
    reader = Reader(rating_scale=(min_rating, max_rating))
    df = preprocess_ratings(df)
    data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

    deterministic_algos = [
        NormalPredictor,
        BaselineOnly,
        KNNBasic,
        KNNWithMeans,
        KNNWithZScore,
        KNNBaseline,
        SlopeOne,
    ]

    stochastic_algos = [SVD, SVDpp, NMF, CoClustering]

    detailed_results = {}
    threshold = df[target].mean()

    for algo in deterministic_algos:
        print(f"Running {algo.__name__} {cv_type} cross validation")
        results = cross_validation(
            algo_class=algo,
            data=data,
            n_splits=n_splits,
            k=k,
            cv_type=cv_type,
            threshold=float(threshold),
        )
        detailed_results[algo.__name__] = results

    for algo in stochastic_algos:
        for random_state in seeds:
            print(
                f"Running {algo.__name__} (SEED: {random_state}) {cv_type} cross validation"
            )
            results = cross_validation(
                algo_class=algo,
                data=data,
                n_splits=n_splits,
                k=k,
                cv_type=cv_type,
                threshold=float(threshold),
                random_state=random_state,
            )
            detailed_results[f"{algo.__name__} (Seed: {random_state})"] = results

    combined_df = pd.DataFrame(
        {algo: results["Mean+-Std"] for algo, results in detailed_results.items()}
    )

    results_path = (
        f"{results_folder}/surprise_{cv_type}_k={n_splits}_{dataset}_top-{k}.csv"
    )

    combined_df.to_csv(results_path)

    print(f"Resultados guardados en {results_path}")


def get_stats_tests(top_k: int, verbose: bool = False):
    datasets = ["mars", "itm"]
    files = [
        f"./results/metrics_kfold_k=5_mars_top-{top_k}.csv",
        f"./results/metrics_kfold_k=5_itm_top-{top_k}.csv",
        f"./results/surprise_kfold_k=5_mars_top-{top_k}.csv",
        f"./results/surprise_kfold_k=5_itm_top-{top_k}.csv",
    ]

    models = [
        "NormalPredictor",
        "KNNBasic",
        "KNNWithMeans",
        "KNNWithZScore",
        "KNNBaseline",
        "SlopeOne",
        "SVD (Seed: 0)",
        "SVDpp (Seed: 0)",
        "NMF (Seed: 0)",
        "CoClustering (Seed: 0)",
        "SVD (Seed: 1)",
        "SVDpp (Seed: 1)",
        "NMF (Seed: 1)",
        "CoClustering (Seed: 1)",
        "SVD (Seed: 42)",
        "SVDpp (Seed: 42)",
        "NMF (Seed: 42)",
        "CoClustering (Seed: 42)",
        "Modelo Propuesto",
    ]

    stats_results = {dataset: {"p_value": 0, "stat": 0} for dataset in datasets}

    for dataset in datasets:
        stat, p = friedman_test(files, models, dataset, top_k, verbose=verbose)
        print(f"Results for dataset {dataset} in top-{top_k} are:")
        print(f"Stat: {stat}, p: {p}")
        stats_results[dataset]["p_value"] = p
        stats_results[dataset]["stat"] = stat

    stastics_path = f"{config.RESULTS_FOLDER}/stats/stats_{top_k}.csv"
    pd.DataFrame.from_dict(stats_results, orient="index").to_csv(stastics_path)
