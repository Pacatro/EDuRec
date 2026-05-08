from typing import Annotated

import typer

from .. import settings
from ..datasets import DatasetName, Phase
from ..evaluation.ghost_eval import eval_ghost
from ..recsys import generate_candidates
from ..recsys.io import load_model
from ..recsys.retrieval import Retrieval, RetrievalConfig

app = typer.Typer(no_args_is_help=True)


@app.command(name="eval", help="Evaluate the GHOST model.")
def eval_command(
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ] = DatasetName.MARS,
    batch_size: Annotated[
        int, typer.Option("--batch_size", "-b", help="Batch size")
    ] = settings.RANKER_BATCH_SIZE,
    top_k: Annotated[
        int, typer.Option("--top_k", "-k", help="Top-k value")
    ] = settings.RANKER_TOP_K,
    candidate_top_n: Annotated[
        int,
        typer.Option(
            "--candidate_top_n",
            "-n",
            help="Number of retrieval candidates generated before reranking",
        ),
    ] = settings.TOP_N,
    remove_sparse: Annotated[
        bool,
        typer.Option(
            "--remove_sparse",
            "-R",
            help="Remove users with less than MIN_INTERACTIONS interactions",
        ),
    ] = settings.REMOVE_SPARSE,
    min_interactions: Annotated[
        int,
        typer.Option(
            "--min_interactions",
            "-I",
            help="Minimum number of interactions per user",
        ),
    ] = settings.MIN_INTERACTIONS,
    models_folder: Annotated[
        str,
        typer.Option("--models-folder", "-M", help="Folder where models are stored."),
    ] = settings.MODELS_FOLDER,
) -> None:
    from ..datasets import ElearningDataModule

    dm = ElearningDataModule(
        dataset=dataset,
        batch_size=batch_size,
        test_ratio=settings.TEST_RATIO,
        val_ratio=settings.VAL_RATIO,
        use_processed_data=True,
        random_state=settings.state["random_state"],
        min_interactions=min_interactions,
        remove_sparse=remove_sparse,
    )

    dm.setup(phase=Phase.RETRIEVAL)

    retrieval_model_path, retrieval_cfg = load_model(
        models_folder=models_folder,
        dataset_name=dataset.value,
        phase=Phase.RETRIEVAL,
    )

    if not isinstance(retrieval_cfg, RetrievalConfig):
        raise RuntimeError("The loaded model is not a retrieval model.")

    assert dm.u_static_feats is not None and dm.i_static_feats is not None

    retrieval = Retrieval.load_from_checkpoint(
        checkpoint_path=str(retrieval_model_path),
        cfg=retrieval_cfg,
        u_static_feats=dm.u_static_feats,
        i_static_feats=dm.i_static_feats,
        top_k=top_k,
        map_location="cpu",
        weights_only=False,
    )

    if settings.state["verbose"]:
        print(f"[EVAL] Generating top-{candidate_top_n} candidates per query")

    generate_candidates(
        retrieval=retrieval,
        dm=dm,
        top_n=candidate_top_n,
        i_static_feats=dm.i_static_feats,
    )

    metrics = eval_ghost(
        dataset=dataset,
        batch_size=batch_size,
        val_size=settings.VAL_RATIO,
        test_size=settings.TEST_RATIO,
        use_processed_data=settings.SAVE_DATA,
        remove_sparse=remove_sparse,
        min_interactions=min_interactions,
        top_ks=[5, 10, 20],
    )

    if settings.state["verbose"]:
        print(f"\n[EVAL] Evaluation results for dataset: {dataset.value}")
        print(metrics.to_string())
    else:
        print(metrics.to_string())
