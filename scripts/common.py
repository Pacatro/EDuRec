import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from edurec import settings
from edurec.datasets import (
    DATASETS,
    CommunityConfig,
    DataConfig,
    DatasetName,
    ElearningDataModule,
)
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
    return [ds for ds in DATASETS if ds.value not in settings.DISABLED_DATASETS]


def data_config(
    args: argparse.Namespace,
    dataset: DatasetName,
    *,
    random_state: int | None = None,
    deduplicate: bool = settings.DEDUPLICATE_INTERACTIONS,
) -> DataConfig:
    """Build the data configuration from the shared CLI arguments."""
    community = CommunityConfig(
        max_users=args.max_users,
        max_interactions=args.max_interactions,
        max_items=args.max_items,
        seed=args.selection_seed,
        fields=args.community_fields,
        min_history_retention=args.min_history_retention,
    )

    return DataConfig(
        dataset=dataset,
        test_ratio=args.test_size,
        val_ratio=args.val_size,
        min_interactions=args.min_interactions,
        remove_sparse=args.remove_sparse,
        deduplicate=deduplicate,
        random_state=random_state,
        community=community,
    )


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
    data = dm.data
    counts_users = data.data_processor.counts("users")
    counts_items = data.data_processor.counts("items")
    interaction_features = dm.adapter.interaction_features
    dataset_config = {
        "num_users": data.num_users,
        "num_items": data.num_items,
        "num_item_dense_feats": counts_items.dense,
        "num_item_text_feats": counts_items.text,
        "num_user_dense_feats": counts_users.dense,
        "num_interaction_dense_feats": len(interaction_features.dense_cols),
        "user_cat_cardinalities": data.data_processor.categorical_cardinalities(
            "users"
        ),
        "interaction_cat_cardinalities": interaction_features.cat_cardinalities,
        "kg_node_counts": dm.adapter.kg_node_counts,
        "kg_edge_types": [list(edge) for edge in dm.adapter.kg_edge_types],
        "has_history": dm.has_history,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    effective = {**dataset_config, **overrides}
    if dm.config.dataset == DatasetName.DORIS:
        effective["use_time_features"] = False
    if dm.config.dataset == DatasetName.ITM:
        effective.update(has_history=False, use_time_features=False)
    if base is not None:
        return replace(base, **effective)

    return ModelConfig(**effective)


def print_data_summary(prefix: str, dm: ElearningDataModule) -> None:
    data = dm.data
    split_sizes = {split: len(frame) for split, frame in data.splits().items()}
    print(
        f"[{prefix}] Data ready: users={data.num_users:,}, items={data.num_items:,}, "
        f"interactions={data.num_interactions:,}, sparsity={data.sparsity:.4f}, "
        f"feedback={data.feedback_type}, split={data.split_strategy}"
    )
    print(
        f"[{prefix}] Splits: "
        + ", ".join(f"{split}={size:,}" for split, size in split_sizes.items())
    )
    community = dm.config.community
    print(
        f"[{prefix}] Selection: community, seed={community.seed}, "
        f"max_items={community.max_items or 'reachable'}, "
        f"max_users={community.max_users or 'all'}, "
        f"max_interactions={community.max_interactions or 'all'}"
    )
    report = data.reports.get("selection_report")
    if isinstance(report, dict):
        print(
            f"[{prefix}] Selected {report['selected_users']:,}/{report['eligible_users']:,} users; "
            f"{report['selected_interactions']:,}/{report['eligible_interactions']:,} interactions"
        )
    if not data.is_explicit:
        negatives = dm.train_ds.negative_item_ids
        print(
            f"[{prefix}] Train negatives: "
            f"{settings.TRAIN_NEGATIVES_PER_POSITIVE} per positive "
            f"({negatives.numel():,} precomputed)"
        )

    if settings.state["verbose"]:
        counts_users = data.data_processor.counts("users")
        counts_items = data.data_processor.counts("items")
        print(
            f"[{prefix}] Features: "
            f"user_dense={counts_users.dense}, "
            f"item_dense={counts_items.dense}, "
            f"user_text={counts_users.text}, "
            f"item_text={counts_items.text}, "
            f"user_cat={counts_users.categorical}, "
            f"item_cat={counts_items.categorical}"
        )
