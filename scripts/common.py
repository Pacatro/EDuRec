import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from edurec import settings
from edurec.datasets import DatasetName, ElearningDataModule, dataset_loaders
from edurec.recsys import ModelArch, ModelConfig


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--verbose", action="store_true")


def add_selection_args(
    parser: argparse.ArgumentParser,
    dataset_default: DatasetName | None = None,
) -> None:
    parser.add_argument(
        "--dataset",
        "-d",
        type=DatasetName,
        choices=list(DatasetName),
        default=dataset_default,
    )
    parser.add_argument(
        "--max-users",
        type=int,
        help="Maximum users retained within the community.",
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        help="Maximum community interactions across all splits; never cut histories to fit.",
    )
    parser.add_argument("--max-items", type=int, help="Maximum items in the community.")
    parser.add_argument(
        "--community-fields",
        nargs="+",
        help="Semantic item fields; defaults to the source schema.",
    )
    parser.add_argument("--min-history-retention", type=float, default=0.7)
    parser.add_argument("--selection-seed", type=int, default=settings.SELECTION_SEED)


def add_preprocessing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min-interactions", "-i", type=int, default=settings.MIN_INTERACTIONS
    )
    parser.add_argument(
        "--remove-sparse",
        action=argparse.BooleanOptionalAction,
        default=settings.REMOVE_SPARSE,
    )
    parser.add_argument(
        "--use-processed",
        "-P",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse data/processed/<dataset> only when its manifest matches the "
        "current data options.",
    )
    parser.add_argument("--val-size", type=float, default=settings.VAL_RATIO)
    parser.add_argument("--test-size", type=float, default=settings.TEST_RATIO)


def add_model_args(parser: argparse.ArgumentParser, with_arch: bool = True) -> None:
    if with_arch:
        parser.add_argument("--arch", "-A", type=ModelArch, choices=list(ModelArch))
    parser.add_argument(
        "--compile", action="store_true", default=settings.COMPILE_MODEL
    )


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", "-e", type=int)
    parser.add_argument("--lr", "-l", type=float)
    parser.add_argument("--batch-size", "-b", type=int)
    parser.add_argument("--patience", "-p", type=int)
    parser.add_argument(
        "--adaptive-k",
        "-a",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use adaptive k. Uses the saved training config if omitted.",
    )


def add_seeds_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seeds",
        "-s",
        default="13,42,77",
        help="Comma-separated seeds to run.",
    )


def add_configs_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--configs-folder",
        "-C",
        type=Path,
        default=Path(settings.CONFIGS_FOLDER),
        help="Folder containing saved configurations.",
    )


def setup_runtime(args: argparse.Namespace, random_state: int | None = None) -> None:
    settings.state["verbose"] = args.verbose
    settings.state["device"] = args.device
    settings.seed_everything(random_state)


def datasets_to_run(dataset: DatasetName | None) -> list[DatasetName]:
    if dataset is not None:
        return [dataset]
    return [ds for ds in dataset_loaders if ds.value not in settings.DISABLED_DATASETS]


def parse_seeds(seeds: str) -> list[int]:
    parsed = [int(seed.strip()) for seed in seeds.split(",") if seed.strip()]
    if not parsed:
        raise ValueError("At least one seed is required.")
    return parsed


def build_config(
    dm: ElearningDataModule,
    base: ModelConfig | None = None,
    **overrides: Any,
) -> ModelConfig:
    dataset_config = {
        "num_users": dm.num_users,
        "num_items": dm.num_items,
        "num_item_dense_feats": dm.num_item_dense_feats,
        "num_item_text_feats": dm.num_item_text_feats,
        "num_user_dense_feats": dm.num_user_dense_feats,
        "num_interaction_dense_feats": dm.num_interaction_dense_feats,
        "user_cat_cardinalities": dm.user_cat_cardinalities,
        "interaction_cat_cardinalities": dm.interaction_cat_cardinalities,
        "kg_node_counts": dm.kg_node_counts,
        "kg_edge_types": [list(edge) for edge in dm.kg_edge_types],
        "has_history": dm.has_history,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    effective = {**dataset_config, **overrides}
    if dm.dataset_name == DatasetName.DORIS:
        effective["use_time_features"] = False
    if dm.dataset_name == DatasetName.ITM:
        effective.update(has_history=False, use_time_features=False)
    if base is not None:
        return replace(base, **effective)

    return ModelConfig(**effective)


def print_data_summary(prefix: str, dm: ElearningDataModule) -> None:
    split_sizes = {
        split: len(data)
        for split in ("train", "val", "test")
        if (data := getattr(dm.artifacts, split)) is not None
    }
    print(
        f"[{prefix}] Data ready: users={dm.num_users:,}, items={dm.num_items:,}, "
        f"interactions={dm.num_interactions:,}, sparsity={dm.sparsity:.4f}, "
        f"feedback={dm.feedback_type}, split={dm.split_strategy}"
    )
    print(
        f"[{prefix}] Splits: "
        + ", ".join(f"{split}={size:,}" for split, size in split_sizes.items())
    )
    print(
        f"[{prefix}] Selection: community, seed={dm.selection_seed}, "
        f"max_items={dm.max_items or 'reachable'}, "
        f"max_users={dm.max_users or 'all'}, "
        f"max_interactions={dm.max_interactions or 'all'}"
    )
    report = dm.artifacts.reports.get("selection_report")
    if isinstance(report, dict):
        print(
            f"[{prefix}] Selected {report['selected_users']:,}/{report['eligible_users']:,} users; "
            f"{report['selected_interactions']:,}/{report['eligible_interactions']:,} interactions"
        )
    if not dm.is_explicit:
        negatives = dm.train_ds.negative_item_ids
        print(
            f"[{prefix}] Train negatives: "
            f"{settings.TRAIN_NEGATIVES_PER_POSITIVE} per positive "
            f"({negatives.numel():,} precomputed)"
        )

    if settings.state["verbose"]:
        print(
            f"[{prefix}] Features: "
            f"user_dense={dm.num_user_dense_feats}, "
            f"item_dense={dm.num_item_dense_feats}, "
            f"user_text={dm.num_user_text_feats}, "
            f"item_text={dm.num_item_text_feats}, "
            f"user_cat={len(dm.user_cat_cardinalities)}, "
            f"item_cat={len(dm.item_cat_cardinalities)}"
        )
