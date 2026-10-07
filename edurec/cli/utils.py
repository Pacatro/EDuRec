from dataclasses import replace
from pathlib import Path
from typing import Any

import typer

from edurec import settings
from edurec.datasets import DatasetName, ElearningDataModule, dataset_loaders
from edurec.recsys import ModelArch, ModelConfig
from edurec.recsys.configs import TrainConfig


def datasets_to_run(dataset: DatasetName | None) -> list[DatasetName]:
    return [dataset] if dataset is not None else list(dataset_loaders)


def config_paths(
    configs_folder: Path,
    run_name: str,
    arch: ModelArch | str,
) -> tuple[Path, Path]:
    """Return the per-architecture model and training config paths for a run.

    File names match the copies written under ``results/optimization`` so a
    dataset's saved configuration can be correlated across both folders.
    """
    arch_value = ModelArch(arch).value
    return (
        Path(configs_folder) / "model" / f"config-{run_name}-{arch_value}.yaml",
        Path(configs_folder) / "train" / f"train-config-{run_name}-{arch_value}.yaml",
    )


DATASET_TRAIN_DEFAULTS: dict[DatasetName, TrainConfig] = {
    DatasetName.ITM: TrainConfig(batch_size=32),
}


def dataset_train_defaults(dataset: DatasetName) -> TrainConfig | None:
    """Per-dataset training defaults, applied when no config file exists."""
    return DATASET_TRAIN_DEFAULTS.get(dataset)


def parse_seeds(seeds: str) -> list[int]:
    parsed = [int(seed.strip()) for seed in seeds.split(",") if seed.strip()]
    if not parsed:
        raise typer.BadParameter("At least one seed is required.")
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
    if base is not None:
        return replace(base, **dataset_config, **overrides)

    return ModelConfig(**dataset_config, **overrides)


def print_model_modules(prefix: str, cfg: ModelConfig) -> None:
    """Print the effective model modules in a compact form."""
    graph = "ON" if cfg.graph_mode == "kg" else "OFF"
    sequence = "ON" if cfg.has_history else "OFF"
    user = "ON" if cfg.use_user_features and cfg.user_state.is_active else "OFF"
    context = (
        "ON"
        if cfg.use_interaction_features and cfg.interaction_context.is_active
        else "OFF"
    )
    options = (
        f"text={'ON' if cfg.use_text_features else 'OFF'}, "
        f"graph={graph}, "
        f"rnn={cfg.rnn_type}, "
        f"user={user}, "
        f"context={context}, "
        f"time={'ON' if cfg.use_time_features else 'OFF'}, "
        f"scorer=mlp, "
        f"item_bias={'ON' if cfg.use_item_bias else 'OFF'}"
    )
    print(f"[{prefix}] Model modules: arch={cfg.arch}, sequence={sequence}, {options}")


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
