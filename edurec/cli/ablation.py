from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import pandas as pd
import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule
from ..recsys import EDuRecConfig, RecSys, train_model
from ..evaluation.ablation import (
    ABLATIONS,
    CONTENT_ABLATIONS,
    get_ablation_config,
    get_content_ablation_config,
)
from .utils import (
    build_config,
    dataset_config_path,
    dataset_run_name,
    datasets_to_run,
    parse_seeds,
    print_data_summary,
)

app = typer.Typer(no_args_is_help=True)


@app.command(name="ablation", help="Run EDuRec ablation variants.")
def run_ablation(
    dataset: Annotated[DatasetName | None, typer.Option("--dataset", "-d")] = None,
    seeds: Annotated[
        str,
        typer.Option("--seeds", "-s", help="Comma-separated seeds to run."),
    ] = "13,42,77,101,2026",
    include_content: Annotated[
        bool,
        typer.Option("--include-content/--main-only", help="Run content ablations."),
    ] = True,
    epochs: Annotated[int, typer.Option("--epochs", "-e", min=1)] = settings.EPOCHS,
    lr: Annotated[float, typer.Option("--lr", "-l")] = settings.LR,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Maximum number of interactions to use before splitting.",
        ),
    ] = None,
    batch_size: Annotated[
        int, typer.Option("--batch_size", "-b", min=1)
    ] = settings.BATCH_SIZE,
    patience: Annotated[
        int, typer.Option("--patience", "-p", min=1)
    ] = settings.PATIENCE,
    val_size: Annotated[float, typer.Option("--val_size", "-v")] = settings.VAL_RATIO,
    test_size: Annotated[
        float, typer.Option("--test_size", "-t")
    ] = settings.TEST_RATIO,
    top_k: Annotated[int, typer.Option("--top_k", "-k")] = settings.TOP_K,
    remove_sparse: Annotated[
        bool, typer.Option("--remove_sparse", "-R")
    ] = settings.REMOVE_SPARSE,
    min_interactions: Annotated[
        int, typer.Option("--min_interactions", "-i")
    ] = settings.MIN_INTERACTIONS,
    adaptive_k: Annotated[
        bool, typer.Option("--adaptive_k", "-a")
    ] = settings.ADAPTIVE_K,
    use_processed_data: Annotated[
        bool, typer.Option("--use_processed", "-P")
    ] = settings.SAVE_DATA,
    debug: Annotated[bool, typer.Option("--debug", "-D")] = False,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Root folder for ablation results."),
    ] = Path(settings.RESULTS_FOLDER) / "ablations",
) -> None:
    parsed_seeds = parse_seeds(seeds)
    datasets = datasets_to_run(dataset)
    started_at = datetime.now()
    verbose = settings.state["verbose"]

    variants = list(ABLATIONS) + (list(CONTENT_ABLATIONS) if include_content else [])

    print("\n[ABLATION] EDuRec ablation run")
    print(f"[ABLATION] Datasets: {', '.join(ds.value for ds in datasets)}")
    print(f"[ABLATION] Variants: {', '.join(variants)}")
    print(f"[ABLATION] Seeds: {', '.join(str(seed) for seed in parsed_seeds)}")
    print(f"[ABLATION] Output folder: {output_dir}")

    for dataset_name in datasets:
        run_name = dataset_run_name(dataset_name, limit)
        dataset_root = output_dir / run_name
        dataset_root.mkdir(parents=True, exist_ok=True)
        base_cfg_path = dataset_config_path(
            settings.CONFIGS_FOLDER,
            dataset_name,
            limit,
        )
        rows = []

        for seed in parsed_seeds:
            settings.seed_everything(seed)

            dm = ElearningDataModule(
                dataset=dataset_name,
                batch_size=batch_size,
                test_ratio=test_size,
                val_ratio=val_size,
                use_processed_data=use_processed_data,
                random_state=seed,
                min_interactions=min_interactions,
                remove_sparse=remove_sparse,
                save_atomic_files=False,
                limit=limit,
            )
            dm.setup()
            print_data_summary("ABLATION", dm)
            inter_graph = dm.build_inter_graph()

            if base_cfg_path.exists():
                print("[ABLATION] Using existing config file:", base_cfg_path)
                base_cfg = EDuRecConfig.load(base_cfg_path)
            else:
                print(
                    "[ABLATION] No config file found, creating new config for dataset:",
                    run_name,
                )
                base_cfg = build_config(
                    dm,
                    lr=lr,
                    adaptive_k=adaptive_k,
                    topks=settings.TOP_KS,
                )

            for variant in variants:
                settings.seed_everything(seed)
                cfg = (
                    get_ablation_config(base_cfg, variant)
                    if variant in ABLATIONS
                    else get_content_ablation_config(base_cfg, variant)
                )
                variant_root = dataset_root / variant / f"seed_{seed}"
                variant_root.mkdir(parents=True, exist_ok=True)
                cfg.save(variant_root / "config.yaml")

                print(f"[ABLATION] {run_name} | {variant} | seed={seed}")

                model = RecSys(
                    cfg=cfg,
                    inter_graph=inter_graph,
                    u_static_feats=dm.u_static_feats,
                    i_static_feats=dm.i_static_feats,
                    val_topk=top_k,
                )
                num_parameters = sum(
                    param.numel() for param in model.parameters() if param.requires_grad
                )

                with TemporaryDirectory(
                    prefix=f"edurec-ablation-{variant}-{seed}-"
                ) as tmp:
                    trainer, best_model_path = train_model(
                        model=model,
                        dm=dm,
                        debug=debug,
                        epochs=epochs,
                        patience=patience,
                        monitor=model.monitor,
                        compile=False,
                        verbose=verbose,
                        default_root_dir=tmp,
                    )

                    metrics = (
                        {}
                        if debug
                        else trainer.test(
                            ckpt_path="best",
                            datamodule=dm,
                            weights_only=False,
                        )[0]
                    )

                row: dict[str, float | int | str] = {
                    "variant": variant,
                    "seed": seed,
                    **{
                        name.removeprefix("test/"): value
                        for name, value in metrics.items()
                        if isinstance(value, int | float)
                    },
                    "num_parameters": num_parameters,
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

                if verbose:
                    print(f"[ABLATION] Best checkpoint: {best_model_path}")

        aggregate = pd.DataFrame(rows)
        aggregate_path = dataset_root / "ablation_results.csv"
        aggregate.to_csv(aggregate_path, index=False)
        print(f"[ABLATION] Saved aggregate table: {aggregate_path}")

    elapsed = str(datetime.now() - started_at).split(".", maxsplit=1)[0]
    print(f"[ABLATION] Finished in {elapsed}")
