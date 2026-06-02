from pathlib import Path
from typing import Annotated
from datetime import datetime

import pandas as pd
import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule
from ..evaluation import eval_proposed_model, eval_sota_models
from ..recsys import EDuRecConfig

app = typer.Typer(no_args_is_help=True)


@app.command(
    name="eval",
    help="Evaluate the proposed EDuRec model and RecBole SOTA models.",
)
def eval_models(
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
            help="Number of training epochs used by all evaluated models.",
        ),
    ] = settings.EPOCHS,
    lr: Annotated[
        float,
        typer.Option(
            "--lr",
            "-l",
            min=0.0,
            help="Learning rate used by all evaluated models.",
        ),
    ] = settings.LR,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            min=1,
            help="Batch size used by EDuRec preprocessing and RecBole.",
        ),
    ] = settings.BATCH_SIZE,
    patience: Annotated[
        int,
        typer.Option(
            "--patience",
            "-p",
            min=1,
            help="Early stopping patience used by all evaluated models.",
        ),
    ] = settings.PATIENCE,
    topks: Annotated[
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
    cfg_path: Annotated[
        Path | None,
        typer.Option(
            "--cfg-path",
            "-c",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Optional extra RecBole config file applied to SOTA models.",
        ),
    ] = None,
    sota_models: Annotated[
        list[str],
        typer.Option(
            "--sota-model",
            "-m",
            help="RecBole SOTA model to evaluate. Repeat this option for multiple models.",
        ),
    ] = settings.SOTA_MODELS,
    adaptive_k: Annotated[
        bool,
        typer.Option(
            "--adaptive-k/--fixed-k",
            "-a/-A",
            help="Use adaptive k to compute metrics that support it in the proposed model.",
        ),
    ] = settings.ADAPTIVE_K,
) -> None:
    eval_topks = list(topks)
    val_topk = max(eval_topks)
    val_ratio = 0.1
    test_ratio = 0.1

    if settings.state["verbose"]:
        print(f"[EVAL] Dataset: {dataset.value}")
        print("[EVAL] Proposed model: EDuRec")
        print(f"[EVAL] SOTA models: {', '.join(sota_models)}")
        if cfg_path is not None:
            print(f"[EVAL] Extra RecBole config: {cfg_path}")
        print(f"[EVAL] Epochs: {epochs}")
        print(f"[EVAL] Learning rate: {lr}")
        print(f"[EVAL] Batch size: {batch_size}")
        print(f"[EVAL] Patience: {patience}")
        print(f"[EVAL] Top-k values: {eval_topks}")
        print(f"[EVAL] Validation top-k: {val_topk}")
        print(f"[EVAL] Use processed cache: {use_processed_data}")
        print(f"[EVAL] Load side features: {load_side_features}")
        print(f"[EVAL] Remove sparse users/items: {remove_sparse}")
        print(f"[EVAL] Min interactions: {min_interactions}")
        print(f"[EVAL] Validation ratio: {val_ratio}")
        print(f"[EVAL] Test ratio: {test_ratio}")
        print(f"[EVAL] Adaptive k: {adaptive_k}")

    dm = ElearningDataModule(
        dataset=dataset,
        batch_size=batch_size,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        min_interactions=min_interactions,
        remove_sparse=remove_sparse,
        use_processed_data=use_processed_data,
        save_atomic_files=True,
        random_state=settings.state["random_state"],
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
        lr=lr,
        adaptive_k=adaptive_k,
        topks=eval_topks,
    )

    proposed_results = eval_proposed_model(
        dm=dm,
        cfg=cfg,
        epochs=epochs,
        val_topk=val_topk,
        patience=patience,
    )

    sota_results = eval_sota_models(
        models=sota_models,
        dm=dm,
        cfg_path=cfg_path,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        topks=eval_topks,
        load_side_features=load_side_features,
    )

    results = pd.concat(
        [pd.DataFrame([proposed_results]), sota_results],
        ignore_index=True,
        sort=False,
    )

    print(results)

    restults_path = Path(settings.RESULTS_FOLDER)
    restults_path.mkdir(parents=True, exist_ok=True)
    results.to_csv(restults_path / f"{dataset.value}.csv")
