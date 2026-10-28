import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from edurec import settings
from edurec.datasets import (
    DATASETS,
    DataConfig,
    DatasetName,
    ElearningDataModule,
)
from edurec.datasets.preprocessing.tabular import (
    categorical_cardinalities,
    feature_columns,
)
from edurec.recsys import EdurecDataAdapter, ModelArch, ModelConfig


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--verbose", action="store_true")


def add_dataset_args(
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


def add_preprocessing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-interactions",
        "-i",
        type=int,
        default=settings.MAX_INTERACTIONS,
        help="Maximum interactions, keeping whole user histories. MOOCCubeX "
        "samples by activity before preprocessing; other sources sample after "
        "sparsity filtering. Defaults to no limit.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=42,
        help="Reproducible MOOCCubeX sampling seed, independent of training seeds.",
    )
    parser.add_argument(
        "--max-exclusive-concepts",
        type=int,
        default=None,
        help="MOOCCubeX: cap concepts linked to only one retained course; keep "
        "all shared concepts. Defaults to no cap; 0 removes exclusive concepts.",
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
    random_state: int | None = None,
    deduplicate: bool = settings.DEDUPLICATE_INTERACTIONS,
) -> DataConfig:
    """Build the data configuration from the shared CLI arguments."""
    return DataConfig(
        dataset=dataset,
        test_ratio=args.test_size,
        val_ratio=args.val_size,
        max_interactions=args.max_interactions,
        sampling_seed=args.sampling_seed,
        max_exclusive_concepts=args.max_exclusive_concepts,
        remove_sparse=args.remove_sparse,
        deduplicate=deduplicate,
        random_state=random_state,
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
    adapter = dm.require_adapter(EdurecDataAdapter)
    interaction_features = adapter.interaction_features
    dataset_config = {
        "num_users": data.num_users,
        "num_items": data.num_items,
        "num_item_dense_feats": data.i_static_feats.shape[1],
        "num_item_text_feats": data.i_static_feats.shape[1]
        - len(feature_columns(data.transforms["items"], "num")),
        "num_user_dense_feats": data.u_static_feats.shape[1],
        "num_interaction_dense_feats": len(interaction_features.dense_cols),
        "user_cat_cardinalities": categorical_cardinalities(data.transforms["users"]),
        "interaction_cat_cardinalities": interaction_features.cat_cardinalities,
        "kg_node_counts": {
            node: adapter.knowledge_graph[node].num_nodes
            for node in adapter.knowledge_graph.node_types
            if node not in {"user", "item"}
        },
        "kg_edge_types": [list(edge) for edge in adapter.knowledge_graph.edge_types],
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
    if not data.is_explicit and isinstance(dm.adapter, EdurecDataAdapter):
        negatives = dm.adapter.dataset("train").negative_item_ids
        print(
            f"[{prefix}] Train negatives: "
            f"{settings.TRAIN_NEGATIVES_PER_POSITIVE} per positive "
            f"({negatives.numel():,} precomputed)"
        )

    if settings.state["verbose"]:
        user_text = data.u_static_feats.shape[1] - len(
            feature_columns(data.transforms["users"], "num")
        )
        item_text = data.i_static_feats.shape[1] - len(
            feature_columns(data.transforms["items"], "num")
        )
        print(
            f"[{prefix}] Features: "
            f"user_dense={data.u_static_feats.shape[1]}, "
            f"item_dense={data.i_static_feats.shape[1]}, "
            f"user_text={user_text}, "
            f"item_text={item_text}, "
            f"user_cat={data.u_cat_feats.shape[1]}, "
            f"item_cat={len(categorical_cardinalities(data.transforms['items']))}"
        )
