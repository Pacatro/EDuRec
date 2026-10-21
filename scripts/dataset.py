"""Show dataset information.

Usage: uv run scripts/dataset.py -d coco --max-users 200
"""

import argparse
import json

from common import add_selection_args

from edurec import settings
from edurec.datasets import DatasetName, ElearningDataModule


def main() -> None:
    parser = argparse.ArgumentParser(description="Show dataset information.")
    add_selection_args(parser, dataset_default=DatasetName.EXPLICIT_MARS)
    parser.add_argument(
        "--max-rows",
        "-m",
        type=int,
        default=10,
        help="Number of rows to display per table.",
    )
    args = parser.parse_args()

    dm = ElearningDataModule(
        dataset=args.dataset,
        batch_size=settings.BATCH_SIZE,
        test_ratio=settings.TEST_RATIO,
        val_ratio=settings.VAL_RATIO,
        use_processed_data=False,
        remove_sparse=False,
        max_users=args.max_users,
        selection_seed=args.selection_seed,
    )

    dm.prepare_data()
    dm.setup()

    print(f"Dataset name: {args.dataset.value}")
    print(f"Feedback type: {dm.feedback_type}")
    print(f"Split strategy: {dm.split_strategy}")
    if not dm.is_explicit:
        print(
            "Training negatives: "
            f"{settings.TRAIN_NEGATIVES_PER_POSITIVE} per positive interaction"
        )
    print(f"Dataset sparsity: {dm.sparsity}")
    print(f"Number of users: {dm.num_users}")
    print(f"Number of items: {dm.num_items}")
    print(f"Number of interactions: {dm.num_interactions}")
    print(f"Number of user features: {dm.num_user_feats}")
    print(f"Number of item features: {dm.num_item_feats}")

    raw_dataset = dm.raw_dataset
    assert raw_dataset is not None

    if raw_dataset.selection_report is not None:
        print("\nDataset selection diagnostics:")
        print(json.dumps(raw_dataset.selection_report, indent=2, ensure_ascii=False))

    max_rows = args.max_rows
    print(f"\nUser information:\n{raw_dataset.user_features.head(max_rows)}\n")
    print(f"Item information:\n{raw_dataset.item_features.head(max_rows)}\n")
    print(f"Interactions:\n{raw_dataset.interactions.head(max_rows)}")


if __name__ == "__main__":
    main()
