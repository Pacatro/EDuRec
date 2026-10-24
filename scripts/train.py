"""Train EDuRec.

Usage: uv run scripts/train.py -d explicit_mars
"""

import argparse
import datetime
from pathlib import Path

from common import (
    add_configs_arg,
    add_model_args,
    add_preprocessing_args,
    add_runtime_args,
    add_selection_args,
    add_training_args,
    build_config,
    data_config,
    datasets_to_run,
    print_data_summary,
    setup_runtime,
)

from edurec import settings
from edurec.datasets import ElearningDataModule
from edurec.recsys import ModelArch, ModelConfig, RecSys, arch_label, train_model
from edurec.recsys.configs import monitor_topk, resolve_train_config
from edurec.recsys.io import save_metrics, save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the model.")
    add_runtime_args(parser)
    add_selection_args(parser)
    add_preprocessing_args(parser)
    add_model_args(parser)
    add_training_args(parser)
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        help="Validation cutoff for early stopping. Defaults to the maximum saved top-k.",
    )
    parser.add_argument(
        "--random-state", "-r", type=int, help="Seed for training and data splits."
    )
    parser.add_argument("--debug", "-D", action="store_true", help="Fast debug run.")
    parser.add_argument("--save-model", "-S", action="store_true")
    parser.add_argument(
        "--models-folder", "-M", type=Path, default=Path(settings.MODELS_FOLDER)
    )
    add_configs_arg(parser)
    parser.add_argument("--run-name", "-N", help="Optional MLflow run name.")
    parser.add_argument(
        "--gcl",
        dest="use_gcl",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable GCL without changing its configured weight.",
    )
    parser.add_argument(
        "--gcl-weight",
        type=float,
        help="KGSeq contrastive loss weight; zero disables GCL.",
    )
    parser.add_argument(
        "--gcl-temperature", type=float, help="Positive InfoNCE temperature."
    )
    parser.add_argument(
        "--gcl-edge-dropout",
        type=float,
        help="Edge removal probability in each KG view.",
    )
    parser.add_argument(
        "--gcl-max-items",
        type=int,
        help="Maximum unique contrastive anchors per batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_runtime(args, random_state=args.random_state)

    started_at = datetime.datetime.now(datetime.UTC)
    training_root = Path(settings.RESULTS_FOLDER) / "training"
    datasets = datasets_to_run(args.dataset)

    for dataset_idx, dataset_name in enumerate(datasets, start=1):
        dataset_label = dataset_name.value
        resolved_arch = args.arch if args.arch is not None else ModelArch.KG_RNN
        run_config_path = (
            args.configs_folder / f"{resolved_arch.value}-{dataset_label}.yaml"
        )
        train_cfg = resolve_train_config(
            cli={
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "adaptive_k": args.adaptive_k,
            },
            saved_path=run_config_path,
        )
        val_topk = monitor_topk(args.top_k, train_cfg)

        print("\n[TRAIN] Training run")
        print(f"[TRAIN] Dataset {dataset_idx}/{len(datasets)}: {dataset_label}")
        print(f"[TRAIN] Model: {resolved_arch.value}")
        print(f"[TRAIN] Monitor: val/ndcg@{val_topk}")
        print(f"[TRAIN] Save model: {args.save_model}")
        print("[TRAIN] Preparing data...")

        if args.verbose:
            print(
                "[TRAIN] Config: "
                f"epochs={train_cfg.epochs}, lr={train_cfg.lr}, "
                f"batch_size={train_cfg.batch_size}, patience={train_cfg.patience}, "
                f"adaptive_k={train_cfg.adaptive_k}, debug={args.debug}"
            )

            if args.run_name:
                print(f"[TRAIN] Logger: MLflow, run_name={args.run_name}")

            print(
                "[TRAIN] Data config: "
                f"use_processed={args.use_processed}, "
                f"remove_sparse={args.remove_sparse}, "
                f"min_interactions={args.min_interactions}, "
                f"val_ratio={args.val_size}, test_ratio={args.test_size}"
            )

        dm = ElearningDataModule(
            config=data_config(
                args,
                dataset_name,
                random_state=args.random_state,
                deduplicate=train_cfg.deduplicate_interactions,
            ),
            batch_size=train_cfg.batch_size,
            use_processed_data=args.use_processed,
            max_history=train_cfg.max_history,
        )

        dm.prepare_data()
        dm.setup()

        print_data_summary("TRAIN", dm)

        gcl_overrides = {
            "use_gcl": args.use_gcl,
            "gcl_weight": args.gcl_weight,
            "gcl_temperature": args.gcl_temperature,
            "gcl_edge_dropout": args.gcl_edge_dropout,
            "gcl_max_items": args.gcl_max_items,
        }

        if run_config_path.exists():
            cfg = build_config(
                dm,
                base=ModelConfig.load(run_config_path),
                arch=resolved_arch,
                **gcl_overrides,
            )
            print(f"[TRAIN] Using saved model config: {run_config_path}")
        else:
            cfg = build_config(dm, arch=resolved_arch, **gcl_overrides)
            print("[TRAIN] No saved model config found; using the full model.")

        if run_config_path.exists():
            print(f"[TRAIN] Using saved training config: {run_config_path}")

        recsys = RecSys(
            cfg=cfg,
            knowledge_graph=dm.adapter.knowledge_graph,
            i_static_feats=dm.data.i_static_feats,
            u_static_feats=dm.data.u_static_feats,
            u_cat_feats=dm.data.u_cat_feats,
            train_cfg=train_cfg,
            val_topk=val_topk,
        )

        print(f"[TRAIN] Training {arch_label(cfg)}...")

        trainer, best_model_path, timer = train_model(
            model=recsys,
            dm=dm,
            debug=args.debug,
            epochs=train_cfg.epochs,
            patience=train_cfg.patience,
            run_name=args.run_name,
            monitor=recsys.monitor,
            compile=args.compile,
            verbose=args.verbose,
        )

        if args.debug:
            elapsed = str(datetime.datetime.now(datetime.UTC) - started_at).split(
                ".", maxsplit=1
            )[0]
            print("[TRAIN] Debug mode: skipping evaluation")
            print(f"[TRAIN] Finished {dataset_name.value} in {elapsed}\n")
            return

        metrics = dict(
            trainer.test(ckpt_path="best", datamodule=dm, weights_only=False)[0]
        )
        metrics["training_time_s"] = timer.time_elapsed("train")
        metrics["inference_time_s"] = timer.time_elapsed("test")

        print(f"[TRAIN] Training time: {metrics['training_time_s']}")
        print(f"[TRAIN] Inference time: {metrics['inference_time_s']}")
        print(f"[TRAIN] Best checkpoint: {best_model_path}")

        metrics_path = save_metrics(metrics, dataset_name.value, training_root)
        print(f"[TRAIN] Metrics saved: {metrics_path}")

        if args.save_model and trainer.is_global_zero:
            model_file_path, model_config_path = save_model(
                model_config=cfg,
                dataset_name=dataset_label,
                best_model_path=best_model_path,
                models_folder=args.models_folder,
            )
            print(f"[TRAIN] Model weights saved: {model_file_path}")
            print(f"[TRAIN] Model config saved: {model_config_path}")

        now = datetime.datetime.now(datetime.UTC)
        elapsed = str(now - started_at).split(".", maxsplit=1)[0]
        print(f"[TRAIN] Finished {dataset_name.value} in {elapsed}\n")


if __name__ == "__main__":
    main()
