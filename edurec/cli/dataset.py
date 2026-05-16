from typing import Annotated

import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule

app = typer.Typer(no_args_is_help=True)


@app.command(name="dataset", help="Print dataset information.")
def dataset_command(
    dataset: Annotated[
        DatasetName, typer.Option("--dataset", "-d", help="Dataset to use")
    ] = DatasetName.MARS,
    max_rows: Annotated[
        int, typer.Option("--max_rows", "-m", help="Maximum number of rows to show")
    ] = 10,
):
    dm = ElearningDataModule(
        dataset=dataset,
        batch_size=settings.RANKER_BATCH_SIZE,
        test_ratio=settings.TEST_RATIO,
        val_ratio=settings.VAL_RATIO,
        use_processed_data=False,
        remove_sparse=False,
    )

    print(f"Dataset name: {dataset.value}")
    print(f"Dataset sparsity: {dm.sparsity}")
    print(f"Number of users: {dm.num_users}")
    print(f"Number of items: {dm.num_items}")
    print(f"Number of interactions: {dm.num_interactions}")
    print(f"Number of user features: {dm.num_user_feats}")
    print(f"Number of item features: {dm.num_item_feats}")
    print(f"Number of interactions context features: {dm.num_ctx_feats}")

    assert dm.raw_dataset is not None

    print(f"\nUser information:\n{dm.raw_dataset.u_feats.head(max_rows)}\n")
    print(f"Item information:\n{dm.raw_dataset.i_feats.head(max_rows)}\n")
    print(f"Interactions:\n{dm.raw_dataset.interactions.head(max_rows)}")
