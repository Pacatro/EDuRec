import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule
from ..evaluation import eval_model, eval_sota_models, eval_upgpr
from ..recsys import EDuRecConfig
from ..recsys.configs import monitor_topk, resolve_train_config
from .utils import (
    build_config,
    config_paths,
    dataset_run_name,
    datasets_to_run,
    dataset_train_defaults,
    parse_seeds,
    print_data_summary,
    print_model_modules,
)

app = typer.Typer(no_args_is_help=True)


def _save_seed_results(
    results: pd.DataFrame,
    dataset_root: Path,
    seed: int,
) -> None:
    for result in results.to_dict(orient="records"):
        model = str(result.pop("model"))
        row = {"model": model, "seed": seed, **result}
        path = dataset_root / model / f"seed_{seed}" / settings.METRICS_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(path, index=False)


def _target_models(sota_models: list[str]) -> list[str]:
    return list(dict.fromkeys(["EDuRec", *sota_models]))


def _load_seed_result(
    dataset_root: Path,
    model: str,
    seed: int,
) -> dict[str, object] | None:
    path = dataset_root / model / f"seed_{seed}" / settings.METRICS_FILENAME
    if not path.exists():
        return None

    try:
        result = pd.read_csv(path).iloc[0].to_dict()
    except (IndexError, pd.errors.EmptyDataError, OSError):
        return None

    result["model"] = str(result.get("model", model))
    result["seed"] = int(result.get("seed", seed))
    return result  # type: ignore


def _pending_models_by_seed(
    dataset_root: Path,
    models: list[str],
    seeds: list[int],
) -> dict[int, list[str]]:
    pending = {}
    for seed in seeds:
        missing = [
            model
            for model in models
            if _load_seed_result(dataset_root, model, seed) is None
        ]
        if missing:
            pending[seed] = missing
    return pending


def _collect_seed_results(
    dataset_root: Path,
    models: list[str],
    seeds: list[int],
) -> list[dict[str, object]]:
    rows = []
    for seed in seeds:
        for model in models:
            row = _load_seed_result(dataset_root, model, seed)
            if row is not None:
                rows.append(row)
    return rows


def _summarize_seed_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate numeric evaluation metrics across seeds for each model."""
    metric_cols = [col for col in results.columns if col not in {"model", "seed"}]
    summary_rows: list[dict[str, str]] = []

    for model, model_results in results.groupby("model", sort=False):
        row = {"model": str(model)}
        for metric in metric_cols:
            values = pd.to_numeric(model_results[metric], errors="coerce").dropna()
            if values.empty:
                continue

            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            row[metric] = f"{mean:.4f} ± {std:.4f}"
        summary_rows.append(row)

    return pd.DataFrame(summary_rows, columns=["model", *metric_cols])


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
    ] = "13,42,77",
    epochs: Annotated[
        int | None,
        typer.Option(
            "--epochs",
            "-e",
            min=1,
            help="Number of training epochs. Uses the saved training config if omitted.",
        ),
    ] = None,
    lr: Annotated[
        float | None,
        typer.Option(
            "--lr",
            "-l",
            min=0.0,
            help="Learning rate. Uses the saved training config if omitted.",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Maximum number of interactions to use before splitting.",
        ),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            "-b",
            min=1,
            help="Batch size used by EDuRec, UPGPR and RecBole. "
            "Uses the saved training config if omitted.",
        ),
    ] = None,
    patience: Annotated[
        int | None,
        typer.Option(
            "--patience",
            "-p",
            min=1,
            help="Early stopping patience. Uses the saved training config if omitted.",
        ),
    ] = None,
    topks: Annotated[
        list[int] | None,
        typer.Option(
            "--top-k",
            "-k",
            min=1,
            help="Top-k values to evaluate. Repeat this option for multiple values. "
            "Uses the saved training config if omitted.",
        ),
    ] = None,
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
        bool | None,
        typer.Option(
            "--adaptive-k/--fixed-k",
            "-a/-A",
            help="Use adaptive k to compute metrics that support it in the proposed "
            "model. Uses the saved training config if omitted.",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Root folder for evaluation results."),
    ] = Path(settings.RESULTS_FOLDER) / "evaluations",
    configs_folder: Annotated[
        Path,
        typer.Option(
            "--configs-folder",
            "-C",
            help="Folder containing saved EDuRec configurations.",
        ),
    ] = Path(settings.CONFIGS_FOLDER),
) -> None:
    parsed_seeds = parse_seeds(seeds)
    val_ratio = settings.VAL_RATIO
    test_ratio = settings.TEST_RATIO
    verbose = settings.state["verbose"]

    datasets = datasets_to_run(dataset)

    models = _target_models(sota_models)

    print("\n[EVAL] Evaluation run")
    print(f"[EVAL] Datasets: {', '.join(ds.value for ds in datasets)}")
    print(f"[EVAL] Models: EDuRec, UPGPR + {len(sota_models)} SOTA")
    print(f"[EVAL] Seeds: {', '.join(str(seed) for seed in parsed_seeds)}")
    print(f"[EVAL] Results folder: {output_dir}")
    print(f"[EVAL] Configs folder: {configs_folder}\n")

    for dataset_idx, dataset_name in enumerate(datasets, start=1):
        run_name = dataset_run_name(dataset_name, limit)
        dataset_root = output_dir / run_name
        dataset_root.mkdir(parents=True, exist_ok=True)
        dataset_started_at = datetime.datetime.now(datetime.UTC)
        model_config_path, train_config_path = config_paths(configs_folder, run_name)
        train_cfg = resolve_train_config(
            cli={
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "patience": patience,
                "topks": topks,
                "adaptive_k": adaptive_k,
            },
            saved_path=train_config_path,
            defaults=dataset_train_defaults(dataset_name),
        )
        val_topk = monitor_topk(None, train_cfg)
        pending_by_seed = _pending_models_by_seed(dataset_root, models, parsed_seeds)
        needs_edurec = any(
            "EDuRec" in pending_models for pending_models in pending_by_seed.values()
        )
        saved_cfg = (
            EDuRecConfig.load(model_config_path)
            if needs_edurec and model_config_path.exists()
            else None
        )

        print(f"[EVAL] [{dataset_idx}/{len(datasets)}] Dataset: {run_name}")
        print(f"[EVAL] Top-k: {train_cfg.topks} | val@{val_topk}")
        print(
            f"[EVAL] Models: EDuRec, UPGPR, {', '.join(sota_models) if sota_models else 'none'}"
        )
        if not pending_by_seed:
            print("[EVAL] All requested seeds are already evaluated. Skipping runs.")
        if saved_cfg is not None:
            print(f"[EVAL] Using saved model config: {model_config_path}")
        if train_config_path.exists():
            print(f"[EVAL] Using saved training config: {train_config_path}")

        if verbose:
            if cfg_path is not None:
                print(f"[EVAL] Extra RecBole config: {cfg_path}")
            print(
                "[EVAL] Config: "
                f"epochs={train_cfg.epochs}, lr={train_cfg.lr}, "
                f"batch_size={train_cfg.batch_size}, patience={train_cfg.patience}, "
                f"adaptive_k={train_cfg.adaptive_k}"
            )
            print(
                "[EVAL] Data config: "
                f"use_processed={use_processed_data}, remove_sparse={remove_sparse}, "
                f"min_interactions={min_interactions}, "
                f"val_ratio={val_ratio}, test_ratio={test_ratio}, limit={limit}"
            )

        for seed, pending_models in pending_by_seed.items():
            settings.seed_everything(seed)
            print(
                f"[EVAL] Preparing data for seed={seed} "
                f"({', '.join(pending_models)})..."
            )

            dm = ElearningDataModule(
                dataset=dataset_name,
                batch_size=train_cfg.batch_size,
                test_ratio=test_ratio,
                val_ratio=val_ratio,
                min_interactions=min_interactions,
                remove_sparse=remove_sparse,
                use_processed_data=use_processed_data,
                save_atomic_files=True,
                random_state=seed,
                limit=limit,
            )

            dm.prepare_data()
            dm.setup()

            print_data_summary("EVAL", dm)

            if "EDuRec" in pending_models:
                if saved_cfg is None:
                    cfg = build_config(dm)
                else:
                    cfg = build_config(dm, base=saved_cfg)
                print_model_modules("EVAL", cfg)
                settings.seed_everything(seed)
                print(f"[EVAL] Running EDuRec | seed={seed}")
                proposed_results = eval_model(
                    dm=dm,
                    cfg=cfg,
                    train_cfg=train_cfg,
                    val_topk=val_topk,
                    verbose=verbose,
                )
                _save_seed_results(proposed_results, dataset_root, seed)

            if "UPGPR" in pending_models:
                settings.seed_everything(seed)
                print(f"[EVAL] Running UPGPR | seed={seed}")
                upgpr_results = eval_upgpr(
                    dm=dm,
                    epochs=train_cfg.epochs,
                    lr=train_cfg.lr,
                    val_topk=val_topk,
                    topks=train_cfg.topks,
                    patience=train_cfg.patience,
                    adaptive_k=train_cfg.adaptive_k,
                    verbose=verbose,
                )
                _save_seed_results(upgpr_results, dataset_root, seed)

            pending_sota_models = [
                model for model in sota_models if model in pending_models
            ]
            if pending_sota_models:
                settings.seed_everything(seed)
                pending_sota_label = ", ".join(pending_sota_models)
                print(f"[EVAL] Running SOTA models | seed={seed}: {pending_sota_label}")
                sota_results = eval_sota_models(
                    models=pending_sota_models,
                    dm=dm,
                    cfg_path=cfg_path,
                    epochs=train_cfg.epochs,
                    lr=train_cfg.lr,
                    batch_size=train_cfg.batch_size,
                    patience=train_cfg.patience,
                    topks=train_cfg.topks,
                    results_path=dataset_root,
                    show_progress=verbose,
                )
                _save_seed_results(sota_results, dataset_root, seed)

        rows = _collect_seed_results(dataset_root, models, parsed_seeds)
        results = pd.DataFrame(rows)
        csv_path = dataset_root / "evaluation_results.csv"
        results.to_csv(csv_path, index=False)

        summary = _summarize_seed_results(results)
        summary_path = dataset_root / "evaluation_summary.csv"
        summary.to_csv(summary_path, index=False)

        print("[EVAL] Results:")
        print(summary)
        print(f"[EVAL] Saved: {csv_path}")
        print(f"[EVAL] Saved summary: {summary_path}")
        now = datetime.datetime.now(datetime.UTC)
        elapsed = str(now - dataset_started_at).split(".", maxsplit=1)[0]
        print(f"[EVAL] Finished {run_name} in {elapsed}\n")
