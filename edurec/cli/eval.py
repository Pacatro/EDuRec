from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule
from ..evaluation import eval_model, eval_sota_models, eval_upgpr
from ..recsys import EDuRecConfig, optimize_model
from .utils import build_config, datasets_to_run, parse_seeds, print_data_summary

app = typer.Typer(no_args_is_help=True)


def _save_seed_results(
    results: pd.DataFrame,
    dataset_root: Path,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for result in results.to_dict(orient="records"):
        model = str(result.pop("model"))
        row = {"model": model, "seed": seed, **result}
        model_root = dataset_root / model / f"seed_{seed}"
        model_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(
            model_root / settings.METRICS_FILENAME,
            index=False,
        )
        rows.append(row)

    return rows


@app.command(
    name="eval",
    help="Evaluate EDuRec against UPGPR and RecBole SOTA models.",
)
def eval_models(
    dataset: Annotated[
        DatasetName | None,
        typer.Option("--dataset", "-d", help="Dataset to use."),
    ] = None,
    seeds: Annotated[
        str,
        typer.Option("--seeds", "-s", help="Comma-separated seeds to run."),
    ] = "13,42,77,101,2026",
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
            help="Batch size used by EDuRec, UPGPR and RecBole.",
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
    n_trials: Annotated[
        int,
        typer.Option(
            "--trials",
            "-n",
            min=1,
            help="Number of trials when no optimized configuration exists.",
        ),
    ] = settings.OPTIM_N_TRIALS,
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
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Root folder for evaluation results."),
    ] = Path(settings.RESULTS_FOLDER) / "evaluations",
    configs_folder: Annotated[
        Path,
        typer.Option(
            "--configs-folder",
            "-C",
            help="Folder containing optimized EDuRec configurations.",
        ),
    ] = Path(settings.CONFIGS_FOLDER),
) -> None:
    parsed_seeds = parse_seeds(seeds)
    eval_topks = list(topks)
    val_topk = max(eval_topks)
    val_ratio = 0.1
    test_ratio = 0.1
    verbose = settings.state["verbose"]

    datasets = datasets_to_run(dataset)

    sota_label = ", ".join(sota_models) if sota_models else "none"

    print("\n[EVAL] Evaluation run")
    print(f"[EVAL] Datasets: {', '.join(ds.value for ds in datasets)}")
    print(f"[EVAL] Models: EDuRec, UPGPR + {len(sota_models)} SOTA")
    print(f"[EVAL] Seeds: {', '.join(str(seed) for seed in parsed_seeds)}")
    print(f"[EVAL] Results folder: {output_dir}")
    print(f"[EVAL] Configs folder: {configs_folder}")
    print(f"[EVAL] Top-k: {eval_topks} | val@{val_topk}\n")

    for dataset_idx, dataset in enumerate(datasets, start=1):
        batch_size = settings.BATCH_SIZE if dataset != DatasetName.ITM else 32
        dataset_root = output_dir / dataset.value
        dataset_root.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        dataset_started_at = datetime.now()
        config_path = configs_folder / f"config-{dataset.value}.yaml"
        optimized_cfg = EDuRecConfig.load(config_path) if config_path.exists() else None

        print(f"[EVAL] [{dataset_idx}/{len(datasets)}] Dataset: {dataset.value}")
        print(f"[EVAL] Models: EDuRec, UPGPR, {sota_label}")
        if optimized_cfg is not None:
            print(f"[EVAL] Using optimized config: {config_path}")

        if verbose:
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

        for seed in parsed_seeds:
            settings.seed_everything(seed)
            print(f"[EVAL] Preparing data for seed={seed}...")

            dm = ElearningDataModule(
                dataset=dataset,
                batch_size=batch_size,
                test_ratio=test_ratio,
                val_ratio=val_ratio,
                min_interactions=min_interactions,
                remove_sparse=remove_sparse,
                use_processed_data=use_processed_data,
                save_atomic_files=True,
                random_state=seed,
            )

            dm.setup()

            print_data_summary("EVAL", dm)

            if optimized_cfg is None:
                print(f"[EVAL] No optimized config found. Running {n_trials} trials...")
                study = optimize_model(
                    base_config=build_config(
                        dm,
                        lr=lr,
                        adaptive_k=adaptive_k,
                        topks=eval_topks,
                    ),
                    dm=dm,
                    n_trials=n_trials,
                    epochs=epochs,
                    patience=patience,
                    val_topk=val_topk,
                    verbose=verbose,
                    results_path=dataset_root / "optimization",
                )
                optimized_cfg = EDuRecConfig(**study.best_trial.user_attrs["config"])
                config_path.parent.mkdir(parents=True, exist_ok=True)
                optimized_cfg.save(config_path)
                print(f"[EVAL] Optimized config saved: {config_path}")

            cfg = replace(
                optimized_cfg,
                num_users=dm.num_users,
                num_items=dm.num_items,
                num_ctx_feats=dm.train_ds.num_ctx_feats,
                num_user_dense_feats=dm.num_user_dense_feats,
                num_item_dense_feats=dm.num_item_dense_feats,
                num_user_text_feats=dm.num_user_text_feats,
                num_item_text_feats=dm.num_item_text_feats,
                user_cat_cardinalities=dm.user_cat_cardinalities,
                item_cat_cardinalities=dm.item_cat_cardinalities,
                adaptive_k=adaptive_k,
                topks=eval_topks,
            )

            settings.seed_everything(seed)
            print(f"[EVAL] Running EDuRec | seed={seed}")
            proposed_results = eval_model(
                dm=dm,
                cfg=cfg,
                epochs=epochs,
                val_topk=val_topk,
                patience=patience,
                verbose=verbose,
            )
            rows.extend(_save_seed_results(proposed_results, dataset_root, seed))

            settings.seed_everything(seed)
            print(f"[EVAL] Running UPGPR | seed={seed}")
            upgpr_results = eval_upgpr(
                dm=dm,
                epochs=epochs,
                lr=lr,
                val_topk=val_topk,
                topks=eval_topks,
                patience=patience,
                adaptive_k=adaptive_k,
                verbose=verbose,
            )
            rows.extend(_save_seed_results(upgpr_results, dataset_root, seed))

            settings.seed_everything(seed)
            print(f"[EVAL] Running SOTA models | seed={seed}: {sota_label}")
            sota_results = eval_sota_models(
                models=sota_models,
                dm=dm,
                cfg_path=cfg_path,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                patience=patience,
                topks=eval_topks,
                results_path=dataset_root,
                show_progress=verbose,
            )
            rows.extend(_save_seed_results(sota_results, dataset_root, seed))

        results = pd.DataFrame(rows)
        csv_path = dataset_root / "evaluation_results.csv"
        results.to_csv(csv_path, index=False)

        metric_cols = [col for col in results.columns if col not in {"model", "seed"}]
        preferred_cols = ["model", "seed"] + sorted(
            metric_cols,
            key=lambda col: (
                col.split("@", maxsplit=1)[-1].zfill(4) if "@" in col else "0000",
                col,
            ),
        )
        print("[EVAL] Results:")
        print(results[preferred_cols].round(4).to_string(index=False))
        print(f"[EVAL] Saved: {csv_path}")
        elapsed = str(datetime.now() - dataset_started_at).split(".", maxsplit=1)[0]
        print(f"[EVAL] Finished {dataset.value} in {elapsed}\n")
