from typing import Annotated, cast

import lightning as L
import torch
import typer

from .. import settings
from ..datasets import DatasetName, ElearningDataModule
from ..recsys.io import load_model, save_metrics
from ..recsys.recsys import RecSys
from .utils import dataset_run_name, print_model_modules

app = typer.Typer(no_args_is_help=True)


@app.command(
    name="test",
    help="Load the most recent saved model and evaluate it on the test split.",
)
def test_recsys(
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ] = DatasetName.EXPLICIT_MARS,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "-l",
            min=1,
            help="Maximum number of interactions to use before splitting.",
        ),
    ] = None,
    batch_size: Annotated[
        int, typer.Option("--batch_size", "-b", help="Batch size")
    ] = settings.BATCH_SIZE,
    val_size: Annotated[
        float, typer.Option("--val_size", "-v", help="Validation size")
    ] = settings.VAL_RATIO,
    test_size: Annotated[
        float, typer.Option("--test_size", "-t", help="Test size")
    ] = settings.TEST_RATIO,
    top_k: Annotated[
        int, typer.Option("--top_k", "-k", help="Top-k value")
    ] = settings.TOP_K,
    adaptive_k: Annotated[
        bool,
        typer.Option(
            "--adaptive_k", "-a", help="Use adaptive k to compute some metrics"
        ),
    ] = settings.ADAPTIVE_K,
    use_processed_data: Annotated[
        bool, typer.Option("--use_processed", "-P", help="Use saved processed data")
    ] = settings.SAVE_DATA,
    remove_sparse: Annotated[
        bool, typer.Option("--remove_sparse", "-R", help="Remove users")
    ] = settings.REMOVE_SPARSE,
    models_folder: Annotated[
        str,
        typer.Option(
            "--models-folder", "-M", help="Folder where saved models are stored."
        ),
    ] = settings.MODELS_FOLDER,
) -> None:
    model_path, cfg = load_model(
        models_folder=models_folder,
        dataset_name=dataset_run_name(dataset, limit),
    )

    dm = ElearningDataModule(
        dataset=dataset,
        batch_size=batch_size,
        test_ratio=test_size,
        val_ratio=val_size,
        use_processed_data=use_processed_data,
        random_state=settings.state["random_state"],
        remove_sparse=remove_sparse,
        limit=limit,
    )
    dm.setup()
    test_graph = dm.build_inter_graph()
    print_model_modules("TEST", cfg)

    model = RecSys.load_from_checkpoint(
        checkpoint_path=str(model_path),
        cfg=cfg,
        inter_graph=test_graph,
        u_static_feats=dm.u_static_feats,
        i_static_feats=dm.i_static_feats,
        val_topk=top_k,
        adaptive_k=adaptive_k,
        map_location=torch.device("cpu"),
        weights_only=False,
        strict=False,
    )

    trainer = L.Trainer(
        accelerator=settings.state["device"],
        devices="auto",
        logger=False,
    )
    test_results = trainer.test(model=model, datamodule=dm, weights_only=False)[0]

    save_metrics(cast(dict[str, float], test_results), model_path.parent)
