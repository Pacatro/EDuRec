from typing import Annotated, cast

import torch
import typer

from .. import config
from ..datasets import DatasetName, ElearningDataModule, Phase
from ..pipelines.candidates import generate_candidates
from ..pipelines.training import train_model
from ..recsys.io import load_model, save_model
from ..recsys.ghost import GhostConfig
from ..recsys.ranker import Ranker
from ..recsys.retrieval import Retrieval
from ..recsys.two_tower import RetrievalConfig

app = typer.Typer(no_args_is_help=True)


def _print_common_info(
    dataset: DatasetName,
    use_processed_data: bool,
    remove_sparse: bool,
    min_interactions: int,
) -> None:
    if not config.state["verbose"]:
        return

    print(f"[TRAIN] Using dataset {dataset.value}")
    print(f"[TRAIN] Removing sparse users: {remove_sparse}")
    print(f"[TRAIN] Minimum interactions per user: {min_interactions}")

    if use_processed_data:
        print(f"[TRAIN] Using saved processed data from {config.PROCESSED_FOLDER}")
    else:
        print(f"[TRAIN] Processing raw data from {config.DATA_FOLDER}/{dataset.value}")


def _print_datamodule_stats(dm: ElearningDataModule) -> None:
    if not config.state["verbose"]:
        return

    print(f"[TRAIN] Dataset sparsity: {dm.sparsity}")
    print(f"[TRAIN] Number of users: {dm.num_users}")
    print(f"[TRAIN] Number of items: {dm.num_items}")
    print(f"[TRAIN] Number of interactions: {dm.num_interactions}")
    print(f"[TRAIN] Number of user features: {dm.num_user_feats}")
    print(f"[TRAIN] Number of item features: {dm.num_item_feats}")
    print(f"[TRAIN] Number of interactions context features: {dm.num_ctx_feats}")


@app.command(name="train_retrieval", help="Train the retrieval model.")
def train_retrieval_command(
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ] = DatasetName.MARS,
    epochs: Annotated[
        int, typer.Option("--epochs", "-e", help="Number of epochs")
    ] = config.RETRIEVAL_EPOCHS,
    lr: Annotated[
        float, typer.Option("--lr", "-l", help="Learning rate")
    ] = config.RETRIEVAL_LR,
    batch_size: Annotated[
        int, typer.Option("--batch_size", "-b", help="Batch size")
    ] = config.RETRIEVAL_BATCH_SIZE,
    patience: Annotated[
        int, typer.Option("--patience", "-p", help="Patience")
    ] = config.RETRIEVAL_PATIENCE,
    val_size: Annotated[
        float, typer.Option("--val_size", "-v", help="Validation size")
    ] = config.VAL_RATIO,
    test_size: Annotated[
        float, typer.Option("--test_size", "-t", help="Test size")
    ] = config.TEST_RATIO,
    top_k: Annotated[
        int, typer.Option("--top_k", "-k", help="Top-k value")
    ] = config.RETRIEVAL_TOP_K,
    remove_sparse: Annotated[
        bool,
        typer.Option(
            "--remove_sparse",
            "-R",
            help="Remove users with less than MIN_INTERACTIONS interactions",
        ),
    ] = config.REMOVE_SPARSE,
    min_interactions: Annotated[
        int,
        typer.Option(
            "--min_interactions",
            "-I",
            help="Minimum number of interactions per user",
        ),
    ] = config.MIN_INTERACTIONS,
    use_logger: Annotated[
        bool, typer.Option("--use_logger", "-L", help="Use MLFlow logger")
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", "-D", help="Debug mode")] = False,
    save: Annotated[
        bool, typer.Option("--save_model", "-S", help="Save model")
    ] = False,
    use_processed_data: Annotated[
        bool, typer.Option("--use_processed", "-P", help="Use saved processed data")
    ] = config.SAVE_DATA,
    models_folder: Annotated[
        str,
        typer.Option(
            "--models-folder", "-M", help="Folder where save the trained model."
        ),
    ] = config.MODELS_FOLDER,
) -> None:
    dm = ElearningDataModule(
        dataset=dataset,
        batch_size=batch_size,
        test_ratio=test_size,
        val_ratio=val_size,
        use_processed_data=use_processed_data,
        random_state=config.state["random_state"],
        min_interactions=min_interactions,
        remove_sparse=remove_sparse,
    )
    _print_common_info(dataset, use_processed_data, remove_sparse, min_interactions)

    dm.setup(phase=Phase.RETRIEVAL)
    _print_datamodule_stats(dm)

    cfg = RetrievalConfig(
        num_users=dm.num_users,
        num_items=dm.num_items,
        num_ctx_feats=dm.train_ds.num_ctx_feats,
        num_user_dense_feats=dm.num_user_dense_feats,
        num_item_dense_feats=dm.num_item_dense_feats,
        user_cat_cardinalities=dm.user_cat_cardinalities,
        item_cat_cardinalities=dm.item_cat_cardinalities,
    )

    assert dm.u_static_feats is not None and dm.i_static_feats is not None

    retrieval = Retrieval(
        cfg=cfg,
        u_static_feats=dm.u_static_feats,
        i_static_feats=dm.i_static_feats,
        lr=lr,
        top_k=top_k,
    )

    trainer, best_model_path = train_model(
        model=retrieval,
        dm=dm,
        top_k=top_k,
        debug=debug,
        use_logger=use_logger,
        epochs=epochs,
        patience=patience,
        monitor=f"val/Recall@{top_k}",
    )

    if debug:
        print("[TRAIN] Debug mode: Skipping evaluation")
        return

    test_results = trainer.test(
        ckpt_path="best",
        datamodule=dm,
        weights_only=False,
    )[0]

    if save:
        model_file_path, model_config_path, metrics_path = save_model(
            model_config=cfg,
            dataset_name=dataset.value,
            best_model_path=best_model_path,
            models_folder=models_folder,
            metrics=cast(dict[str, float], test_results),
        )

        if config.state["verbose"]:
            print(f"Model weights saved in: {model_file_path}")
            print(f"Model config saved in: {model_config_path}")
            print(f"Metrics saved in: {metrics_path}")


@app.command(name="train_ranker", help="Train the reranker model.")
def train_ranker_command(
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ] = DatasetName.MARS,
    epochs: Annotated[
        int, typer.Option("--epochs", "-e", help="Number of epochs")
    ] = config.RANKER_EPOCHS,
    lr: Annotated[
        float, typer.Option("--lr", "-l", help="Learning rate")
    ] = config.RANKER_LR,
    batch_size: Annotated[
        int, typer.Option("--batch_size", "-b", help="Batch size")
    ] = config.RANKER_BATCH_SIZE,
    patience: Annotated[
        int, typer.Option("--patience", "-p", help="Patience")
    ] = config.RANKER_PATIENCE,
    val_size: Annotated[
        float, typer.Option("--val_size", "-v", help="Validation size")
    ] = config.VAL_RATIO,
    test_size: Annotated[
        float, typer.Option("--test_size", "-t", help="Test size")
    ] = config.TEST_RATIO,
    top_k: Annotated[
        int, typer.Option("--top_k", "-k", help="Top-k value")
    ] = config.RANKER_TOP_K,
    candidate_top_n: Annotated[
        int,
        typer.Option(
            "--candidate_top_n",
            help="Number of retrieval candidates generated before reranking",
        ),
    ] = config.TOP_N,
    remove_sparse: Annotated[
        bool,
        typer.Option(
            "--remove_sparse",
            "-R",
            help="Remove users with less than MIN_INTERACTIONS interactions",
        ),
    ] = config.REMOVE_SPARSE,
    min_interactions: Annotated[
        int,
        typer.Option(
            "--min_interactions",
            "-I",
            help="Minimum number of interactions per user",
        ),
    ] = config.MIN_INTERACTIONS,
    adaptive_k: Annotated[
        bool,
        typer.Option(
            "--adaptive_k", "-a", help="Use adaptive k to compute some metrics"
        ),
    ] = config.ADAPTIVE_K,
    use_logger: Annotated[
        bool, typer.Option("--use_logger", "-L", help="Use MLFlow logger")
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", "-D", help="Debug mode")] = False,
    save: Annotated[
        bool, typer.Option("--save_model", "-S", help="Save model")
    ] = False,
    use_processed_data: Annotated[
        bool, typer.Option("--use_processed", "-P", help="Use saved processed data")
    ] = config.SAVE_DATA,
    models_folder: Annotated[
        str,
        typer.Option(
            "--models-folder", "-M", help="Folder where save and load models."
        ),
    ] = config.MODELS_FOLDER,
) -> None:
    dm = ElearningDataModule(
        dataset=dataset,
        batch_size=batch_size,
        test_ratio=test_size,
        val_ratio=val_size,
        use_processed_data=use_processed_data,
        random_state=config.state["random_state"],
        min_interactions=min_interactions,
        remove_sparse=remove_sparse,
    )
    _print_common_info(dataset, use_processed_data, remove_sparse, min_interactions)

    dm.setup(phase=Phase.RETRIEVAL)
    _print_datamodule_stats(dm)

    retrieval_model_path, retrieval_cfg = load_model(
        models_folder=models_folder,
        dataset_name=dataset.value,
        model_type=Phase.RETRIEVAL,
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
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    if config.state["verbose"]:
        print(f"[TRAIN] Loaded retrieval model from: {retrieval_model_path}")
        print(f"[TRAIN] Generating top-{candidate_top_n} candidates per query")

    candidates = generate_candidates(retrieval=retrieval, dm=dm, top_n=candidate_top_n)
    print(candidates)
    dm.setup(phase=Phase.RANKING)

    cfg = GhostConfig(
        num_users=dm.num_users,
        num_items=dm.num_items,
        num_ctx_feats=dm.train_ds.num_ctx_feats,
        num_user_dense_feats=dm.num_user_dense_feats,
        num_item_dense_feats=dm.num_item_dense_feats,
        user_cat_cardinalities=dm.user_cat_cardinalities,
        item_cat_cardinalities=dm.item_cat_cardinalities,
    )

    reranker = Ranker(
        cfg=cfg,
        inter_graph=dm.create_inter_graph(),
        u_static_feats=dm.u_static_feats,
        i_static_feats=dm.i_static_feats,
        lr=lr,
        top_k=top_k,
        adaptive_k=adaptive_k,
    )

    trainer, best_model_path = train_model(
        model=reranker,
        dm=dm,
        top_k=top_k,
        debug=debug,
        use_logger=use_logger,
        epochs=epochs,
        patience=patience,
        monitor=f"val/NDCG@{top_k}",
    )

    if debug:
        print("[TRAIN] Debug mode: Skipping evaluation")
        return

    test_results = trainer.test(
        ckpt_path="best",
        datamodule=dm,
        weights_only=False,
    )[0]

    if save:
        model_file_path, model_config_path, metrics_path = save_model(
            model_config=cfg,
            dataset_name=dataset.value,
            best_model_path=best_model_path,
            models_folder=models_folder,
            metrics=cast(dict[str, float], test_results),
        )

        if config.state["verbose"]:
            print(f"Model weights saved in: {model_file_path}")
            print(f"Model config saved in: {model_config_path}")
            print(f"Metrics saved in: {metrics_path}")
