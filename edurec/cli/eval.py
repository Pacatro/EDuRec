from pathlib import Path
from typing import Annotated

import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule
from ..evaluation import eval_sota_models

app = typer.Typer(no_args_is_help=True)


@app.command(name="sota", help="Evaluate RecBole models on EDuRec atomic splits.")
def eval_sota_command(
    cfg_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Optional RecBole YAML file with model-specific overrides.",
        ),
    ] = None,
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use."),
    ] = DatasetName.MARS,
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            min=1,
            help="Number of RecBole training epochs.",
        ),
    ] = settings.EPOCHS,
    lr: Annotated[
        float,
        typer.Option(
            "--lr",
            "-l",
            min=0.0,
            help="RecBole learning rate.",
        ),
    ] = settings.RANKER_LR,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            min=1,
            help="Batch size used by EDuRec preprocessing and RecBole.",
        ),
    ] = settings.RANKER_BATCH_SIZE,
    patience: Annotated[
        int,
        typer.Option(
            "--patience",
            "-p",
            min=1,
            help="Early stopping patience for RecBole.",
        ),
    ] = settings.RANKER_PATIENCE,
    val_size: Annotated[
        float,
        typer.Option(
            "--val-size",
            "-v",
            min=0.0,
            max=1.0,
            help="Validation split ratio used by the EDuRec preprocessing.",
        ),
    ] = settings.VAL_RATIO,
    test_size: Annotated[
        float,
        typer.Option(
            "--test-size",
            "-t",
            min=0.0,
            max=1.0,
            help="Test split ratio used by the EDuRec preprocessing.",
        ),
    ] = settings.TEST_RATIO,
    top_ks: Annotated[
        list[int],
        typer.Option(
            "--top-k",
            "-k",
            min=1,
            help="Top-k values to evaluate. Repeat this option for multiple values.",
        ),
    ] = settings.TOP_KS,
    remove_sparse: Annotated[
        bool,
        typer.Option(
            "--remove-sparse/--keep-sparse",
            "-R/-K",
            help="Remove sparse users/items before preprocessing.",
        ),
    ] = settings.REMOVE_SPARSE,
    min_interactions: Annotated[
        int,
        typer.Option(
            "--min-interactions",
            "-I",
            min=1,
            help="Minimum interactions per user/item after sparse filtering.",
        ),
    ] = settings.MIN_INTERACTIONS,
    use_processed_data: Annotated[
        bool,
        typer.Option(
            "--use-processed/--no-use-processed",
            "-P/-N",
            help="Reuse cached processed data when available.",
        ),
    ] = settings.SAVE_DATA,
    load_side_features: Annotated[
        bool,
        typer.Option(
            "--side-features/--no-side-features",
            "-S/-X",
            help="Load RecBole .user and .item atomic files in addition to .inter files.",
        ),
    ] = False,
) -> None:
    if settings.state["verbose"]:
        print(f"[EVAL] Dataset: {dataset.value}")
        print(f"[EVAL] Models: {', '.join(settings.SOTA_MODELS)}")
        if cfg_path is not None:
            print(f"[EVAL] Extra RecBole config: {cfg_path}")
        print(f"[EVAL] Epochs: {epochs}")
        print(f"[EVAL] Learning rate: {lr}")
        print(f"[EVAL] Batch size: {batch_size}")
        print(f"[EVAL] Patience: {patience}")
        print(f"[EVAL] Top-k values: {top_ks}")
        print(f"[EVAL] Use processed cache: {use_processed_data}")
        print(f"[EVAL] Load side features: {load_side_features}")
        print(f"[EVAL] Remove sparse users/items: {remove_sparse}")
        print(f"[EVAL] Min interactions: {min_interactions}")
        print(f"[EVAL] Validation ratio: {val_size}")
        print(f"[EVAL] Test ratio: {test_size}")

    dm = ElearningDataModule(
        dataset=dataset,
        batch_size=batch_size,
        test_ratio=test_size,
        val_ratio=val_size,
        min_interactions=min_interactions,
        remove_sparse=remove_sparse,
        use_processed_data=use_processed_data,
        save_atomic_files=True,
        random_state=settings.state["random_state"],
    )

    dm.setup()

    results = eval_sota_models(
        models=settings.SOTA_MODELS,
        dm=dm,
        cfg_path=cfg_path,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        top_ks=top_ks,
        load_side_features=load_side_features,
    )

    print(results.to_string(index=False))
