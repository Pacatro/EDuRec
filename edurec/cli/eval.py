from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule
from ..evaluation import eval_model, eval_sota_models
from ..recsys import EDuRecConfig

app = typer.Typer(no_args_is_help=True)


@app.command(
    name="eval",
    help="Evaluate the proposed EDuRec model and RecBole SOTA models.",
)
def eval_models(
    dataset: Annotated[
        DatasetName | None,
        typer.Option("--dataset", "-d", help="Dataset to use."),
    ] = None,
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

    datasets = (
        [dataset]
        if dataset is not None
        else [
            # DatasetName.MARS,
            DatasetName.ITM,
            DatasetName.DORIS,
        ]
    )

    results_path = Path(settings.RESULTS_FOLDER) / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    results_path.mkdir(parents=True, exist_ok=True)

    sota_label = ", ".join(sota_models) if sota_models else "none"

    print("\n[EVAL] Evaluation run")
    print(f"[EVAL] Datasets: {', '.join(ds.value for ds in datasets)}")
    print(f"[EVAL] Models: EDuRec + {len(sota_models)} SOTA")
    print(f"[EVAL] Results folder: {results_path}")
    print(f"[EVAL] Top-k: {eval_topks} | val@{val_topk}\n")

    for dataset_idx, dataset in enumerate(datasets, start=1):
        batch_size = settings.BATCH_SIZE if dataset != DatasetName.ITM else 32
        print(f"[EVAL] [{dataset_idx}/{len(datasets)}] Dataset: {dataset.value}")
        print(f"[EVAL] Models: EDuRec, {sota_label}")
        print("[EVAL] Preparing data...")

        if settings.state["verbose"]:
            if cfg_path is not None:
                print(f"[EVAL] Extra RecBole config: {cfg_path}")
            print(
                "[EVAL] Config: "
                f"epochs={epochs}, lr={lr}, batch_size={batch_size}, "
                f"patience={patience}, adaptive_k={adaptive_k}"
            )
            print(
                "[EVAL] Data config: "
                f"use_processed={use_processed_data}, remove_sparse={remove_sparse}, "
                f"min_interactions={min_interactions}, "
                f"val_ratio={val_ratio}, test_ratio={test_ratio}"
            )

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

        split_sizes = {
            split: len(getattr(dm.artifacts, split))
            for split in ("train", "val", "test")
            if getattr(dm.artifacts, split) is not None
        }
        print(
            "[EVAL] Data ready: "
            f"users={dm.num_users:,}, items={dm.num_items:,}, "
            f"interactions={dm.num_interactions:,}, "
            f"sparsity={dm.sparsity:.4f}"
        )
        print(
            "[EVAL] Splits: "
            + ", ".join(f"{split}={size:,}" for split, size in split_sizes.items())
        )
        if settings.state["verbose"]:
            print(
                "[EVAL] Features: "
                f"context={dm.train_ds.num_ctx_feats}, "
                f"user_dense={dm.num_user_dense_feats}, "
                f"item_dense={dm.num_item_dense_feats}, "
                f"user_text={dm.num_user_text_feats}, "
                f"item_text={dm.num_item_text_feats}, "
                f"user_cat={len(dm.user_cat_cardinalities)}, "
                f"item_cat={len(dm.item_cat_cardinalities)}"
            )

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

        print("[EVAL] Running proposed model: EDuRec")

        dataset_started_at = datetime.now()

        artifacts_path = results_path / dataset.value / "artifacts"
        artifacts_path.mkdir(parents=True, exist_ok=True)

        proposed_results = eval_model(
            dm=dm,
            cfg=cfg,
            epochs=epochs,
            val_topk=val_topk,
            patience=patience,
            verbose=settings.state["verbose"],
            results_path=artifacts_path,
        )

        print(f"[EVAL] Running SOTA models ({len(sota_models)}): {sota_label}")
        sota_results = eval_sota_models(
            models=sota_models,
            dm=dm,
            cfg_path=cfg_path,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            topks=eval_topks,
            show_progress=settings.state["verbose"],
            results_path=artifacts_path,
        )

        results = pd.concat(
            [proposed_results, sota_results],
            ignore_index=True,
            sort=False,
        )

        csv_path = results_path / dataset.value / "final_results.csv"
        results.to_csv(csv_path, index=False)

        metric_cols = [col for col in results.columns if col != "model"]
        preferred_cols = ["model"] + sorted(
            metric_cols,
            key=lambda col: (
                col.split("@", maxsplit=1)[-1].zfill(4) if "@" in col else "0000",
                col,
            ),
        )
        elapsed = str(datetime.now() - dataset_started_at).split(".", maxsplit=1)[0]
        print("[EVAL] Results:")
        print(results[preferred_cols].round(4).to_string(index=False))
        print(f"[EVAL] Saved: {csv_path}")
        print(f"[EVAL] Finished {dataset.value} in {elapsed}\n")
