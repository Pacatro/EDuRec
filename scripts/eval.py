"""Evaluate EDuRec against baseline models.

Usage: uv run scripts/eval.py -d coco --only-proposed
"""

import argparse
import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from common import (
    add_configs_arg,
    add_dataset_args,
    add_model_args,
    add_preprocessing_args,
    add_runtime_args,
    add_seeds_args,
    add_training_args,
    build_config,
    data_config,
    datasets_to_run,
    parse_seeds,
    print_data_summary,
    setup_runtime,
)

from edurec import settings
from edurec.datasets import ElearningDataModule
from edurec.evaluation import (
    baseline_labels,
    baseline_train_defaults,
    eval_model,
    eval_recbole_baselines,
    get_baseline,
)
from edurec.recsys import ARCH_LABELS, ModelArch, ModelConfig
from edurec.recsys.configs import monitor_topk, resolve_train_config
from edurec.recsys.ranking import EVALUATION_PROTOCOL

# Dedicated MLflow experiment for evaluation runs, separate from training (TFM)
# and HPO runs.
MLFLOW_EXPERIMENT_NAME = "EVAL"


def _save_seed_results(results: pd.DataFrame, dataset_root: Path, seed: int) -> None:
    for result in results.to_dict(orient="records"):
        model = str(result.pop("model"))
        row = {
            "model": model,
            **result,
            "seed": seed,
            "evaluation_protocol": EVALUATION_PROTOCOL,
        }
        path = (
            dataset_root / model / f"seed_{seed}" / f"{settings.METRICS_FILENAME}.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(path, index=False)


def _selected_baselines(baseline_models: list[str]) -> list[str]:
    """Resolve the requested baselines, expanding ``all`` to every available one."""
    if "all" in baseline_models:
        return list(dict.fromkeys([*settings.BASELINES, *baseline_labels()]))
    return list(dict.fromkeys(baseline_models))


def _target_models(
    proposed_label: str,
    baseline_models: list[str],
    only_proposed: bool = False,
    only_baselines: bool = False,
) -> list[str]:
    models: list[str] = []
    if not only_baselines:
        models.append(proposed_label)
    if not only_proposed:
        models.extend(baseline_models)
    return list(dict.fromkeys(models))


def _load_seed_result(
    dataset_root: Path,
    model: str,
    seed: int,
) -> dict[str, object] | None:
    path = dataset_root / model / f"seed_{seed}" / f"{settings.METRICS_FILENAME}.csv"
    if not path.exists():
        return None

    try:
        # CSV column labels are always strings, so the returned row has str keys.
        result = cast(dict[str, Any], pd.read_csv(path).iloc[0].to_dict())
    except (IndexError, pd.errors.EmptyDataError, OSError):
        return None

    if result.get("evaluation_protocol") != EVALUATION_PROTOCOL:
        return None

    result["model"] = str(result.get("model", model))
    result["seed"] = int(result.get("seed", seed))
    return result


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
    metric_cols = [
        col
        for col in results.columns
        if col not in {"model", "seed", "evaluation_protocol"}
    ]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate EDuRec against baseline models."
    )
    add_runtime_args(parser)
    add_dataset_args(parser)
    add_preprocessing_args(parser)
    add_model_args(parser)
    add_training_args(parser)
    add_seeds_args(parser)
    add_configs_arg(parser)
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        action="append",
        dest="top_k",
        help="Top-k values to evaluate. Repeat this option for multiple values.",
    )
    parser.add_argument(
        "--baseline-model",
        "-m",
        action="append",
        dest="baseline_model",
        help="Baseline model to evaluate (for example KGAN, LightGCN or SASRec). "
        "Repeat for multiple models, or use `all`.",
    )
    only = parser.add_mutually_exclusive_group()
    only.add_argument(
        "--only-proposed",
        action="store_true",
        help="Only evaluate the proposed model, skipping baseline models.",
    )
    only.add_argument(
        "--only-baselines",
        action="store_true",
        help="Only evaluate baseline models, skipping the proposed model.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path(settings.RESULTS_FOLDER) / "evaluations",
        help="Root folder for evaluation results.",
    )
    parser.add_argument(
        "--logger",
        "-L",
        action="store_true",
        help="Log the proposed model training to the EVAL MLflow experiment. "
        "Each run is named <arch>-<dataset>-<seed>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_runtime(args)

    parsed_seeds = parse_seeds(args.seeds)
    # ``all`` selects every RecBole and native baseline.
    selected_baselines = _selected_baselines(args.baseline_model or settings.BASELINES)
    resolved_arch = args.arch if args.arch is not None else ModelArch.KG_RNN
    proposed_label = ARCH_LABELS[resolved_arch.value]
    datasets = datasets_to_run(args.dataset)

    models = _target_models(
        proposed_label,
        selected_baselines,
        only_proposed=args.only_proposed,
        only_baselines=args.only_baselines,
    )

    print("\n[EVAL] Evaluation run")
    print(f"[EVAL] Datasets: {', '.join(ds.value for ds in datasets)}")
    print(f"[EVAL] Models: {', '.join(models)}")
    print(f"[EVAL] Seeds: {', '.join(str(seed) for seed in parsed_seeds)}")
    print(f"[EVAL] Results folder: {args.output_dir}")
    print(f"[EVAL] Configs folder: {args.configs_folder}\n")

    for dataset_idx, dataset_name in enumerate(datasets, start=1):
        dataset_label = dataset_name.value
        dataset_root = args.output_dir / dataset_label
        dataset_root.mkdir(parents=True, exist_ok=True)
        dataset_started_at = datetime.datetime.now(datetime.UTC)
        run_config_path = (
            args.configs_folder / f"{resolved_arch.value}-{dataset_label}.yaml"
        )
        train_cfg = resolve_train_config(
            cli={
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "topks": args.top_k,
                "adaptive_k": args.adaptive_k,
            },
            saved_path=run_config_path,
        )
        val_topk = monitor_topk(None, train_cfg)
        pending_by_seed = _pending_models_by_seed(dataset_root, models, parsed_seeds)
        needs_proposed = not args.only_baselines and any(
            proposed_label in pending_models
            for pending_models in pending_by_seed.values()
        )
        saved_cfg = (
            ModelConfig.load(run_config_path)
            if needs_proposed and run_config_path.exists()
            else None
        )

        print(f"[EVAL] [{dataset_idx}/{len(datasets)}] Dataset: {dataset_label}")
        print(f"[EVAL] Top-k: {train_cfg.topks} | val@{val_topk}")
        print(f"[EVAL] Models: {', '.join(models)}")
        if not pending_by_seed:
            print("[EVAL] All requested seeds are already evaluated. Skipping runs.")
        if saved_cfg is not None:
            print(f"[EVAL] Using saved model config: {run_config_path}")
        if run_config_path.exists():
            print(f"[EVAL] Using saved training config: {run_config_path}")

        if args.verbose:
            print(
                "[EVAL] Config: "
                f"epochs={train_cfg.epochs}, lr={train_cfg.lr}, "
                f"batch_size={train_cfg.batch_size}, patience={train_cfg.patience}, "
                f"adaptive_k={train_cfg.adaptive_k}"
            )
            print(
                "[EVAL] Data config: "
                f"use_processed={args.use_processed}, "
                f"remove_sparse={args.remove_sparse}, "
                f"max_interactions={args.max_interactions or 'none'}, "
                f"val_ratio={args.val_size}, test_ratio={args.test_size}"
            )

        for seed, pending_models in pending_by_seed.items():
            settings.seed_everything(seed)
            print(
                f"[EVAL] Preparing data for seed={seed} "
                f"({', '.join(pending_models)})..."
            )

            dm = ElearningDataModule(
                config=data_config(
                    args,
                    dataset_name,
                    random_state=seed,
                    deduplicate=train_cfg.deduplicate_interactions,
                ),
                batch_size=train_cfg.batch_size,
                use_processed_data=args.use_processed,
                max_history=train_cfg.max_history,
            )

            dm.prepare_data()
            dm.setup()

            print_data_summary("EVAL", dm)

            if not args.only_baselines and proposed_label in pending_models:
                cfg = build_config(dm, base=saved_cfg, arch=resolved_arch)
                settings.seed_everything(seed)
                mlflow_run_name = (
                    f"{resolved_arch.value}-{dataset_label}-seed={seed}"
                    if args.logger
                    else None
                )
                if mlflow_run_name is not None and args.verbose:
                    print(f"[EVAL] Logger: MLflow, run_name={mlflow_run_name}")
                print(f"[EVAL] Running {proposed_label} | seed={seed}")
                proposed_results = eval_model(
                    dm=dm,
                    cfg=cfg,
                    train_cfg=train_cfg,
                    val_topk=val_topk,
                    compile=args.compile,
                    verbose=args.verbose,
                    run_name=mlflow_run_name,
                    experiment_name=MLFLOW_EXPERIMENT_NAME,
                )
                _save_seed_results(proposed_results, dataset_root, seed)

            pending_baselines = [
                model for model in selected_baselines if model in pending_models
            ]
            for baseline in pending_baselines:
                settings.seed_everything(seed)
                print(f"[EVAL] Running baseline {baseline} | seed={seed}")
                runner = get_baseline(baseline)
                if runner is not None:
                    baseline_train_cfg = resolve_train_config(
                        cli={
                            "epochs": args.epochs,
                            "lr": args.lr,
                            "batch_size": args.batch_size,
                            "patience": args.patience,
                            # Query construction and ranking stay common to
                            # all models in this evaluation run.
                            "topks": train_cfg.topks,
                            "adaptive_k": train_cfg.adaptive_k,
                            "max_history": train_cfg.max_history,
                            "deduplicate_interactions": train_cfg.deduplicate_interactions,
                        },
                        defaults=baseline_train_defaults(baseline) or train_cfg,
                    )
                    baseline_run_name = (
                        f"{baseline}-{dataset_label}-seed={seed}"
                        if args.logger
                        else None
                    )
                    baseline_results = runner(
                        dm=dm,
                        train_cfg=baseline_train_cfg,
                        val_topk=val_topk,
                        compile=args.compile,
                        verbose=args.verbose,
                        run_name=baseline_run_name,
                        experiment_name=MLFLOW_EXPERIMENT_NAME,
                    )
                else:
                    baseline_results = eval_recbole_baselines(
                        models=[baseline],
                        dm=dm,
                        epochs=train_cfg.epochs,
                        lr=train_cfg.lr,
                        batch_size=train_cfg.batch_size,
                        patience=train_cfg.patience,
                        topks=train_cfg.topks,
                        adaptive_k=train_cfg.adaptive_k,
                        show_progress=args.verbose,
                    )
                _save_seed_results(baseline_results, dataset_root, seed)

        rows = _collect_seed_results(dataset_root, models, parsed_seeds)
        results = pd.DataFrame(rows)
        suffix = (
            "_baselines"
            if args.only_baselines
            else "_proposed"
            if args.only_proposed
            else ""
        )
        csv_name = f"evaluation_results{suffix}.csv"
        summary_name = f"evaluation_summary{suffix}.csv"

        csv_path = dataset_root / csv_name
        results.to_csv(csv_path, index=False)

        summary = _summarize_seed_results(results)
        summary_path = dataset_root / summary_name
        summary.to_csv(summary_path, index=False)

        print("[EVAL] Results:")
        print(summary)
        print(f"[EVAL] Saved: {csv_path}")
        print(f"[EVAL] Saved summary: {summary_path}")
        now = datetime.datetime.now(datetime.UTC)
        elapsed = str(now - dataset_started_at).split(".", maxsplit=1)[0]
        print(f"[EVAL] Finished {dataset_label} in {elapsed}\n")


if __name__ == "__main__":
    main()
