"""Optimize EDuRec hyperparameters with Optuna.

Usage: uv run scripts/hpo.py -d coco --trials 30
"""

import argparse
from dataclasses import replace
from pathlib import Path

from common import (
    add_configs_arg,
    add_dataset_args,
    add_model_args,
    add_preprocessing_args,
    add_runtime_args,
    build_config,
    data_config,
    datasets_to_run,
    print_data_summary,
    setup_runtime,
)

from edurec import settings
from edurec.datasets import ElearningDataModule
from edurec.recsys import ModelArch, ModelConfig, optimize_model
from edurec.recsys.configs import TrainConfig, save_run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a hyperparameter optimization for the model."
    )
    add_runtime_args(parser)
    parser.add_argument("--random-state", "-r", type=int)
    add_dataset_args(parser)
    add_preprocessing_args(parser)
    add_model_args(parser, with_arch=False)
    add_configs_arg(parser)
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=settings.EPOCHS,
        help="Number of training epochs used by all evaluated models.",
    )
    parser.add_argument(
        "--search-epochs",
        type=int,
        default=settings.OPTIM_SEARCH_EPOCHS,
        help="Max epochs per optimization trial. The best configuration is "
        "saved with the full --epochs budget for the final run.",
    )
    parser.add_argument(
        "--patience",
        "-p",
        type=int,
        default=settings.PATIENCE,
        help="Early stopping patience used by all evaluated models.",
    )
    parser.add_argument(
        "--trials",
        "-n",
        type=int,
        default=settings.OPTIM_N_TRIALS,
        help="Number of optimization trials to perform.",
    )
    parser.add_argument("--batch-size", "-b", type=int, default=settings.BATCH_SIZE)
    parser.add_argument(
        "--log-mlflow",
        "-L",
        action="store_true",
        help="Log trials to the HPO MLflow experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_runtime(args, random_state=args.random_state)

    datasets = datasets_to_run(args.dataset)

    results_root = Path(settings.RESULTS_FOLDER) / "optimization"
    results_root.mkdir(parents=True, exist_ok=True)

    print("\n[OPTIM] Hyperparameter optimization run")
    print(f"[OPTIM] Datasets: {', '.join(ds.value for ds in datasets)}")
    print(f"[OPTIM] Results folder: {results_root}")
    print(f"[OPTIM] Configs folder: {args.configs_folder}")
    print(
        f"[OPTIM] Trials: {args.trials} | Search epochs: {args.search_epochs} | "
        f"Final epochs: {args.epochs}"
    )

    for dataset_name in datasets:
        run_name = dataset_name.value
        resolved_arch = ModelArch.KG_RNN
        if args.verbose:
            print(
                f"[OPTIM] Config: epochs={args.epochs}, batch_size={args.batch_size}, "
                f"trials={args.trials}, patience={args.patience}"
            )
            print(
                "[OPTIM] Data config: "
                f"use_processed={args.use_processed}, "
                f"remove_sparse={args.remove_sparse}, "
                f"max_interactions={args.max_interactions or 'none'}, "
                f"val_ratio={args.val_size}, test_ratio={args.test_size}"
            )

        dm = ElearningDataModule(
            config=data_config(
                args,
                dataset_name,
                random_state=args.random_state,
            ),
            batch_size=args.batch_size,
            use_processed_data=args.use_processed,
            # Trials are short and often pruned; reusing workers would keep
            # paused worker processes alive across trials, which makes the
            # known "can only test a child process" DataLoader fork warning
            # (pytorch/pytorch#149837) far more likely.
            persistent_workers=False,
        )

        dm.prepare_data()
        dm.setup()

        print_data_summary("OPTIM", dm)

        dataset_results_path = results_root / run_name

        base_cfg = build_config(dm, arch=resolved_arch)
        base_train_cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )

        study = optimize_model(
            base_config=base_cfg,
            base_train_config=base_train_cfg,
            dm=dm,
            n_trials=args.trials,
            epochs=args.epochs,
            patience=args.patience,
            search_epochs=args.search_epochs,
            compile=args.compile,
            verbose=args.verbose,
            results_path=dataset_results_path,
            log_mlflow=args.log_mlflow,
        )
        best_attrs = study.best_trial.user_attrs
        if "config" not in best_attrs or "train_config" not in best_attrs:
            raise RuntimeError(
                "The best Optuna trial is missing its configuration metadata; "
                "delete the study database and rerun the optimization."
            )
        best_cfg = ModelConfig(**best_attrs["config"])
        best_train_cfg = TrainConfig(**best_attrs["train_config"])
        # Trials used the reduced search budget; the saved configuration keeps
        # the full epoch budget so training and evaluation run to completion.
        best_train_cfg = replace(best_train_cfg, epochs=args.epochs)

        print(
            f"[OPTIM] Best NDCG {study.best_value} in trial: {study.best_trial.number}"
        )
        print("[OPTIM] Params:", study.best_params)

        reusable_cfg_path = (
            args.configs_folder / f"{resolved_arch.value}-{run_name}.yaml"
        )
        result_cfg_path = results_root / f"{resolved_arch.value}-{run_name}.yaml"
        save_run_config(result_cfg_path, best_cfg, best_train_cfg)
        save_run_config(reusable_cfg_path, best_cfg, best_train_cfg)
        print("[OPTIM] Saved result config:", result_cfg_path)
        print("[OPTIM] Saved reusable config:", reusable_cfg_path)
        print("[OPTIM] Trials log:", dataset_results_path / "trials.csv")
        print("[OPTIM] Study storage:", dataset_results_path / "study.db")


if __name__ == "__main__":
    main()
