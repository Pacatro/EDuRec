import argparse
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData


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
    parser.add_argument(
        "--max-interactions",
        type=int,
        help="Approximate total interaction budget; sample whole users by train "
        "activity after splitting. Disabled by default; not a strict row cap.",
    )
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--max-exclusive-concepts",
        type=int,
        help="Maximum degree-one concepts per course; keep all shared concepts. "
        "Disabled by default; zero removes all exclusive concepts.",
    )


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
        remove_sparse=args.remove_sparse,
        deduplicate=deduplicate,
        random_state=random_state,
        max_interactions=args.max_interactions,
        sampling_seed=args.sampling_seed,
        max_exclusive_concepts=args.max_exclusive_concepts,
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


def graph_statistics(graph: "HeteroData") -> dict[str, Any]:
    """Node and relation counts of a heterogeneous graph.

    Forward relations and their explicit ``rev_`` mirrors are counted
    separately, so the reported edge total matches what message passing sees
    while the per-relation breakdown stays readable.
    """
    nodes = {
        node_type: int(graph[node_type].num_nodes) for node_type in graph.node_types
    }
    forward: dict[str, int] = {}
    reverse: dict[str, int] = {}
    for source, relation, target in graph.edge_types:
        key = f"{source} -{relation}-> {target}"
        count = int(graph[source, relation, target].num_edges)
        (reverse if relation.startswith("rev_") else forward)[key] = count

    return {
        "nodes": nodes,
        "forward_edges": forward,
        "reverse_edges": reverse,
        "num_nodes": sum(nodes.values()),
        "num_forward_edges": sum(forward.values()),
        "num_reverse_edges": sum(reverse.values()),
        "num_relations": len(graph.edge_types),
    }


def print_graph_summary(prefix: str, dm: ElearningDataModule) -> None:
    """Print the knowledge-graph size next to the tabular data summary."""
    adapter = dm.adapter
    if not isinstance(adapter, EdurecDataAdapter):
        return

    stats = graph_statistics(adapter.knowledge_graph)
    print(
        f"[{prefix}] Graph: nodes={stats['num_nodes']:,}, "
        f"edges={stats['num_forward_edges'] + stats['num_reverse_edges']:,} "
        f"(forward={stats['num_forward_edges']:,}, "
        f"reverse={stats['num_reverse_edges']:,}), "
        f"relations={stats['num_relations']}"
    )
    print(
        f"[{prefix}] Graph nodes: "
        + ", ".join(f"{node}={count:,}" for node, count in stats["nodes"].items())
    )
    if stats["forward_edges"]:
        print(
            f"[{prefix}] Graph edges: "
            + ", ".join(
                f"{edge}={count:,}" for edge, count in stats["forward_edges"].items()
            )
        )


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

    print_graph_summary(prefix, dm)

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
