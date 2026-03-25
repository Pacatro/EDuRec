from typing import Annotated, cast

import lightning as L
import torch
import typer

from .. import config
from ..datasets import DatasetName, ElearningDataModule
from ..recsys.engine import RecSys
from ..recsys.io import load_model, save_metrics

app = typer.Typer(no_args_is_help=True)


@app.command(
    name="test",
    help="Load the most recent saved model and evaluate it on the test split.",
)
def test_recsys(
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ] = DatasetName.MARS,
    batch_size: Annotated[
        int, typer.Option("--batch_size", "-b", help="Batch size")
    ] = config.BATCH_SIZE,
    val_size: Annotated[
        float, typer.Option("--val_size", "-v", help="Validation size")
    ] = config.VAL_RATIO,
    test_size: Annotated[
        float, typer.Option("--test_size", "-t", help="Test size")
    ] = config.TEST_RATIO,
    top_k: Annotated[
        int, typer.Option("--top_k", "-k", help="Top-k value")
    ] = config.TOP_K,
    adaptive_k: Annotated[
        bool,
        typer.Option(
            "--adaptive_k", "-a", help="Use adaptive k to compute some metrics"
        ),
    ] = config.ADAPTIVE_K,
    n_neg_train: Annotated[
        int,
        typer.Option(
            "--n_neg_train", help="Number of negatives to sample for training"
        ),
    ] = config.N_NEG_TRAIN,
    n_neg_val: Annotated[
        int,
        typer.Option(
            "--n_neg_val", help="Number of negatives to sample for validation"
        ),
    ] = config.N_NEG_VAL,
    n_neg_test: Annotated[
        int,
        typer.Option("--n_neg_test", help="Number of negatives to sample for testing"),
    ] = config.N_NEG_TEST,
    use_procesed_data: Annotated[
        bool, typer.Option("--use_processed", "-P", help="Use saved processed data")
    ] = config.SAVE_DATA,
    models_folder: Annotated[
        str,
        typer.Option(
            "--models-folder", "-M", help="Folder where saved models are stored."
        ),
    ] = config.MODELS_FOLDER,
):
    model_path, cfg = load_model(
        models_folder=models_folder, dataset_name=dataset.value
    )

    dm = ElearningDataModule(
        dataset=dataset,
        batch_size=batch_size,
        test_ratio=test_size,
        val_ratio=val_size,
        use_processed_data=use_procesed_data,
        random_state=config.state["random_state"],
        n_neg_train=n_neg_train,
        n_neg_val=n_neg_val,
        n_neg_test=n_neg_test,
    )
    dm.setup()

    assert dm.u_static is not None and dm.i_static is not None

    model = RecSys.load_from_checkpoint(
        checkpoint_path=str(model_path),
        cfg=cfg,
        inter_graph=dm.create_inter_graph(),
        u_static=dm.u_static,
        i_static=dm.i_static,
        top_k=top_k,
        adaptive_k=adaptive_k,
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    trainer = L.Trainer(
        accelerator=config.state["device"],
        devices="auto",
        logger=False,
    )
    test_results = trainer.test(model=model, datamodule=dm, weights_only=False)[0]

    save_metrics(cast(dict[str, float], test_results), model_path.parent)
