"""Show dataset information.

Usage: uv run scripts/dataset.py -d coco
"""

import argparse
import json

import pandas as pd
from common import add_dataset_args

from edurec import settings
from edurec.datasets import (
    DataConfig,
    DatasetName,
    ElearningDataModule,
)
from edurec.datasets.preprocessing.tabular import categorical_cardinalities


def main() -> None:
    parser = argparse.ArgumentParser(description="Show dataset information.")
    add_dataset_args(parser, dataset_default=DatasetName.EXPLICIT_MARS)
    parser.add_argument(
        "--max-rows",
        "-m",
        type=int,
        default=10,
        help="Number of rows to display per table.",
    )
    parser.add_argument(
        "--max-interactions",
        "-i",
        type=int,
        default=settings.MAX_INTERACTIONS,
        help="Maximum interactions after filtering; whole user histories are "
        "selected. Defaults to no limit. Minimum interactions per user/item are "
        "read from settings.MIN_INTERACTIONS (or MOOCCUBEX_MIN_CONNECTIONS for "
        "MOOCCubeX).",
    )
    args = parser.parse_args()

    config = DataConfig(
        dataset=args.dataset,
        max_interactions=args.max_interactions,
    )

    dm = ElearningDataModule(config=config, batch_size=settings.BATCH_SIZE)

    dm.prepare_data()
    dm.setup()

    data = dm.data
    print(f"Dataset name: {dm.config.dataset.value}")
    print(f"Feedback type: {data.feedback_type}")
    print(f"Split strategy: {data.split_strategy}")
    if not data.is_explicit:
        print(
            "Training negatives: "
            f"{settings.TRAIN_NEGATIVES_PER_POSITIVE} per positive interaction"
        )
    print(f"Dataset sparsity: {data.sparsity}")
    print(f"Number of users: {data.num_users}")
    print(f"Number of items: {data.num_items}")
    print(f"Number of interactions: {data.num_interactions}")
    print(
        f"Number of user features: {data.u_static_feats.shape[1] + data.u_cat_feats.shape[1]}"
    )
    item_categorical = len(categorical_cardinalities(data.transforms["items"]))
    print(f"Number of item features: {data.i_static_feats.shape[1] + item_categorical}")

    for label, report in (("source", data.reports.get("source_report")),):
        if report is not None:
            print(f"\nDataset {label} diagnostics:")
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    max_rows = args.max_rows
    if dm.build is not None:
        raw = dm.build.raw
        user_features = raw.user_features
        item_features = raw.item_features
        interactions = raw.interactions
    else:
        # Loaded from the processed cache: only the processed tables remain.
        user_features = data.user_features
        item_features = data.item_features
        interactions = pd.concat(data.splits().values(), ignore_index=True)

    print(f"\nUser information:\n{user_features.head(max_rows)}\n")
    print(f"Item information:\n{item_features.head(max_rows)}\n")
    print(f"Interactions:\n{interactions.head(max_rows)}")


if __name__ == "__main__":
    main()
