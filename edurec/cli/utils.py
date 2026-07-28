from typing import Any

import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule, dataset_loaders
from ..recsys import EDuRecConfig


def datasets_to_run(dataset: DatasetName | None) -> list[DatasetName]:
    return [dataset] if dataset is not None else list(dataset_loaders)


def dataset_run_name(dataset: DatasetName, limit: int | None = None) -> str:
    return dataset.value if limit is None else f"{dataset.value}_limit_{limit}"


def parse_seeds(seeds: str) -> list[int]:
    parsed = [int(seed.strip()) for seed in seeds.split(",") if seed.strip()]
    if not parsed:
        raise typer.BadParameter("At least one seed is required.")
    return parsed


def build_config(dm: ElearningDataModule, **kwargs: Any) -> EDuRecConfig:
    return EDuRecConfig(
        num_users=dm.num_users,
        num_items=dm.num_items,
        num_ctx_feats=dm.train_ds.num_ctx_feats,
        num_user_dense_feats=dm.num_user_dense_feats,
        num_item_dense_feats=dm.num_item_dense_feats,
        num_user_text_feats=dm.num_user_text_feats,
        num_item_text_feats=dm.num_item_text_feats,
        user_cat_cardinalities=dm.user_cat_cardinalities,
        item_cat_cardinalities=dm.item_cat_cardinalities,
        has_history=dm.has_history,
        **kwargs,
    )


def print_model_modules(prefix: str, cfg: EDuRecConfig) -> None:
    """Print the effective model modules in a compact form."""
    status = ", ".join(
        f"{name}={'ON' if enabled else 'OFF'}"
        for name, enabled in cfg.available_modules.items()
    )
    print(f"[{prefix}] Model modules: {status}")


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
    if dm.limit is not None:
        print(f"[{prefix}] Interaction limit: {dm.limit:,}")
    if not dm.is_explicit:
        negatives = dm.train_ds.negative_item_ids
        print(
            f"[{prefix}] Train negatives: "
            f"{settings.TRAIN_NEGATIVES_PER_POSITIVE} per positive "
            f"({negatives.numel():,} precomputed)"
        )

    if settings.state["verbose"]:
        print(
            f"[{prefix}] Features: context={dm.train_ds.num_ctx_feats}, "
            f"user_dense={dm.num_user_dense_feats}, "
            f"item_dense={dm.num_item_dense_feats}, "
            f"user_text={dm.num_user_text_feats}, "
            f"item_text={dm.num_item_text_feats}, "
            f"user_cat={len(dm.user_cat_cardinalities)}, "
            f"item_cat={len(dm.item_cat_cardinalities)}"
        )
