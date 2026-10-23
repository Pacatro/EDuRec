"""Run KGSeq ablation variants.

Usage: uv run scripts/ablation.py -d coco --seeds 13,42,77
"""

import argparse
import datetime
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from common import (
    add_configs_arg,
    add_model_args,
    add_preprocessing_args,
    add_runtime_args,
    add_seeds_args,
    add_selection_args,
    add_training_args,
    build_config,
    datasets_to_run,
    parse_seeds,
    print_data_summary,
    setup_runtime,
)

from edurec import settings
from edurec.datasets import ElearningDataModule
from edurec.evaluation.ablation import (
    ABLATIONS,
    ablation_applicable,
    get_ablation_config,
)
from edurec.recsys import ModelArch, ModelConfig, RecSys, train_model
from edurec.recsys.configs import resolve_train_config

# Dedicated MLflow experiment for ablation runs, separate from training (TFM),
# evaluation (EVAL), and HPO runs.
MLFLOW_EXPERIMENT_NAME = "ABLATION"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KGSeq ablation variants.")
    add_runtime_args(parser)
    add_selection_args(parser)
    add_preprocessing_args(parser)
    add_model_args(parser)
    add_training_args(parser)
    add_seeds_args(parser)
    add_configs_arg(parser)
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=settings.TOP_K,
        help="Validation cutoff used by the ablation models.",
    )
    parser.add_argument(
        "--gcl-weight",
        type=float,
        help="Override the GCL loss weight for the base config. The "
        "leave-one-out `no_gcl` variant disables GCL entirely.",
    )
    parser.add_argument("--debug", "-D", action="store_true", help="Fast debug run.")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path(settings.RESULTS_FOLDER) / "ablations",
        help="Root folder for ablation results.",
    )
    parser.add_argument(
        "--logger",
        "-L",
        action="store_true",
        help="Log each ablation variant to the ABLATION MLflow experiment. "
        "Each run is named <arch>-<dataset>-<variant>-seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_runtime(args)

    parsed_seeds = parse_seeds(args.seeds)
    datasets = datasets_to_run(args.dataset)
    started_at = datetime.datetime.now(datetime.UTC)

    variants = list(ABLATIONS)

    print("\n[ABLATION] ablation run")
    print(f"[ABLATION] Datasets: {', '.join(ds.value for ds in datasets)}")
    print(f"[ABLATION] Architecture: {(args.arch or ModelArch.KG_RNN).value}")
    print(f"[ABLATION] Variants: {', '.join(variants)}")
    print(f"[ABLATION] Seeds: {', '.join(str(seed) for seed in parsed_seeds)}")
    print(f"[ABLATION] Output folder: {args.output_dir}")

    for dataset_name in datasets:
        run_name = dataset_name.value
        resolved_arch = args.arch if args.arch is not None else ModelArch.KG_RNN
        dataset_root = args.output_dir / run_name
        dataset_root.mkdir(parents=True, exist_ok=True)
        run_config_path = args.configs_folder / f"{resolved_arch.value}-{run_name}.yaml"
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
        rows = []

        for seed in parsed_seeds:
            settings.seed_everything(seed)

            dm = ElearningDataModule(
                dataset=dataset_name,
                batch_size=train_cfg.batch_size,
                test_ratio=args.test_size,
                val_ratio=args.val_size,
                use_processed_data=args.use_processed,
                random_state=seed,
                min_interactions=args.min_interactions,
                remove_sparse=args.remove_sparse,
                max_history=train_cfg.max_history,
                deduplicate_interactions=train_cfg.deduplicate_interactions,
                max_users=args.max_users,
                max_interactions=args.max_interactions,
                selection_seed=args.selection_seed,
                max_items=args.max_items,
                community_fields=args.community_fields,
                min_history_retention=args.min_history_retention,
            )
            dm.prepare_data()
            dm.setup()
            print_data_summary("ABLATION", dm)
            knowledge_graph = dm.adapter.get_knowledge_graph()

            if run_config_path.exists():
                print("[ABLATION] Using existing model config file:", run_config_path)
                base_cfg = build_config(
                    dm,
                    base=ModelConfig.load(run_config_path),
                    arch=resolved_arch,
                    gcl_weight=args.gcl_weight,
                )
            else:
                print(
                    "[ABLATION] No model config file found, creating new config for dataset:",
                    run_name,
                )
                base_cfg = build_config(
                    dm, arch=resolved_arch, gcl_weight=args.gcl_weight
                )

            for variant in variants:
                settings.seed_everything(seed)
                cfg = get_ablation_config(base_cfg, variant)
                applicable = ablation_applicable(base_cfg, variant)
                variant_root = dataset_root / variant / f"seed_{seed}"
                variant_root.mkdir(parents=True, exist_ok=True)
                cfg.save(variant_root / "config.yaml")

                print(f"[ABLATION] {run_name} | {variant} | seed={seed}")
                if not cfg.has_history and cfg.arch == ModelArch.SASREC_TEXT:
                    print(
                        f"[ABLATION] SKIPPING variant {variant!r}: "
                        "the model requires sequential history."
                    )
                    row: dict[str, float | int | str] = {
                        "variant": variant,
                        "seed": seed,
                        "applicable": 0,
                        "module_graph": int(cfg.graph_mode == "kg"),
                        "module_sequence": int(cfg.has_history),
                        "arch": str(cfg.arch),
                        "graph_mode": cfg.graph_mode,
                        "rnn_type": cfg.rnn_type,
                        "use_text_features": int(cfg.use_text_features),
                        "use_item_features": int(cfg.use_item_features),
                        "use_gcl": int(cfg.gcl_enabled),
                        "gcl_weight": cfg.gcl_weight,
                        "use_user_features": int(cfg.use_user_features),
                        "use_interaction_features": int(cfg.use_interaction_features),
                        "use_time_features": int(cfg.use_time_features),
                        "use_attention_pooling": int(cfg.use_attention_pooling),
                        "use_item_bias": int(cfg.use_item_bias),
                        "num_parameters": 0,
                        "training_time_s": 0.0,
                        "inference_time_s": 0.0,
                    }
                    rows.append(row)
                    pd.DataFrame([row]).to_csv(
                        variant_root / settings.METRICS_FILENAME,
                        index=False,
                    )
                    pd.DataFrame([asdict(cfg)]).to_csv(
                        variant_root / "config.csv",
                        index=False,
                    )
                    continue

                if not applicable:
                    print(
                        f"[ABLATION] WARNING: variant {variant!r} disables a module "
                        f"that {run_name} does not provide; it is a no-op and its "
                        "row is marked applicable=0."
                    )

                model = RecSys(
                    cfg=cfg,
                    knowledge_graph=knowledge_graph,
                    i_static_feats=dm.i_static_feats,
                    u_static_feats=dm.u_static_feats,
                    u_cat_feats=dm.u_cat_feats,
                    train_cfg=train_cfg,
                    val_topk=args.top_k,
                )
                num_parameters = sum(
                    param.numel() for param in model.parameters() if param.requires_grad
                )

                mlflow_run_name = (
                    f"{resolved_arch.value}-{run_name}-{variant}-seed={seed}"
                    if args.logger
                    else None
                )
                if mlflow_run_name is not None and args.verbose:
                    print(f"[ABLATION] Logger: MLflow, run_name={mlflow_run_name}")

                with TemporaryDirectory(
                    prefix=f"edurec-ablation-{variant}-{seed}-"
                ) as tmp:
                    trainer, best_model_path, timer = train_model(
                        model=model,
                        dm=dm,
                        debug=args.debug,
                        epochs=train_cfg.epochs,
                        patience=train_cfg.patience,
                        monitor=model.monitor,
                        compile=args.compile,
                        verbose=args.verbose,
                        default_root_dir=tmp,
                        run_name=mlflow_run_name,
                        experiment_name=MLFLOW_EXPERIMENT_NAME,
                        params={"variant": variant},
                    )

                    training_time = timer.time_elapsed("train")
                    metrics = (
                        {}
                        if args.debug
                        else trainer.test(
                            ckpt_path="best",
                            datamodule=dm,
                            weights_only=False,
                        )[0]
                    )
                    inference_time = 0.0 if args.debug else timer.time_elapsed("test")

                row: dict[str, float | int | str] = {
                    "variant": variant,
                    "seed": seed,
                    "applicable": int(applicable),
                    "module_graph": int(cfg.graph_mode == "kg"),
                    "module_sequence": int(cfg.has_history),
                    "arch": str(cfg.arch),
                    "graph_mode": cfg.graph_mode,
                    "rnn_type": cfg.rnn_type,
                    "use_text_features": int(cfg.use_text_features),
                    "use_item_features": int(cfg.use_item_features),
                    "use_gcl": int(cfg.gcl_enabled),
                    "gcl_weight": cfg.gcl_weight,
                    "use_user_features": int(cfg.use_user_features),
                    "use_interaction_features": int(cfg.use_interaction_features),
                    "use_time_features": int(cfg.use_time_features),
                    "use_attention_pooling": int(cfg.use_attention_pooling),
                    "use_item_bias": int(cfg.use_item_bias),
                    **{
                        name.removeprefix("test/"): value
                        for name, value in metrics.items()
                        if isinstance(value, int | float)
                    },
                    "num_parameters": num_parameters,
                    "training_time_s": training_time,
                    "inference_time_s": inference_time,
                }
                rows.append(row)

                pd.DataFrame([row]).to_csv(
                    variant_root / settings.METRICS_FILENAME,
                    index=False,
                )
                pd.DataFrame([asdict(cfg)]).to_csv(
                    variant_root / "config.csv",
                    index=False,
                )

                if args.verbose:
                    print(f"[ABLATION] Best checkpoint: {best_model_path}")

        aggregate = pd.DataFrame(rows)
        aggregate_path = dataset_root / "ablation_results.csv"
        aggregate.to_csv(aggregate_path, index=False)
        print(f"[ABLATION] Saved aggregate table: {aggregate_path}")

    finish = datetime.datetime.now(datetime.UTC)
    elapsed = str(finish - started_at).split(".", maxsplit=1)[0]
    print(f"[ABLATION] Finished in {elapsed}")


if __name__ == "__main__":
    main()
