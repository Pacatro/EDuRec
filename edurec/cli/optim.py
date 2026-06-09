from datetime import datetime
from typing import Annotated
from pathlib import Path

import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule
from ..recsys import EDuRecConfig, optimize_model, get_best_config

app = typer.Typer(no_args_is_help=True)


@app.command(name="optim", help="Run a hyperparameter optimization for the model.")
def optimize(
    dataset: Annotated[
        DatasetName | None,
        typer.Option("--dataset", "-d", help="Dataset to use."),
    ] = None,
    n_trials: Annotated[
        int,
        typer.Option(
            "--trials",
            "-n",
            min=1,
            help="Number of optimization trials to perform.",
        ),
    ] = settings.OPTIM_N_TRIALS,
    batch_size: Annotated[
        int, typer.Option("--batch_size", "-b", min=1)
    ] = settings.BATCH_SIZE,
    val_ratio: Annotated[float, typer.Option("--val_size", "-v")] = settings.VAL_RATIO,
    test_ratio: Annotated[
        float, typer.Option("--test_size", "-t")
    ] = settings.TEST_RATIO,
    remove_sparse: Annotated[
        bool, typer.Option("--remove_sparse", "-R")
    ] = settings.REMOVE_SPARSE,
    min_interactions: Annotated[
        int, typer.Option("--min_interactions", "-i")
    ] = settings.MIN_INTERACTIONS,
    use_processed_data: Annotated[
        bool, typer.Option("--use_processed", "-P")
    ] = settings.SAVE_DATA,
):
    datasets = (
        [dataset]
        if dataset is not None
        else [
            DatasetName.MARS,
            DatasetName.ITM,
            DatasetName.DORIS,
        ]
    )

    results_root = (
        Path(settings.RESULTS_FOLDER)
        / "optimization"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    results_root.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        dm = ElearningDataModule(
            dataset=dataset,
            batch_size=batch_size,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            use_processed_data=use_processed_data,
            random_state=settings.state["random_state"],
            min_interactions=min_interactions,
            remove_sparse=remove_sparse,
            save_atomic_files=False,
        )

        dm.setup()

        cfg = EDuRecConfig(
            num_users=dm.num_users,
            num_items=dm.num_items,
            num_ctx_feats=dm.train_ds.num_ctx_feats,
            num_user_dense_feats=dm.num_user_dense_feats,
            num_item_dense_feats=dm.num_item_dense_feats,
            num_user_text_feats=dm.num_user_text_feats,
            num_item_text_feats=dm.num_item_text_feats,
            user_cat_cardinalities=dm.user_cat_cardinalities,
            item_cat_cardinalities=dm.item_cat_cardinalities,
        )

        study = optimize_model(
            base_config=cfg,
            dm=dm,
            n_trials=n_trials,
            verbose=settings.state["verbose"],
        )
        cfg = get_best_config(study)

        print("Best NDCG:", study.best_value)
        print("Params:", study.best_params)

        cfg_path = results_root / f"config-{dataset.value}.yaml"
        cfg.save(cfg_path)
        print("Saved config:", cfg_path)
