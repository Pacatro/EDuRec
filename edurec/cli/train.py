from pathlib import Path
from typing import Annotated, cast

import torch
import typer

from .. import config
from ..datasets import DatasetName, ElearningDataModule, Phase
from ..recsys import generate_candidates, train_model
from ..recsys.ghost import GhostConfig
from ..recsys.io import load_model, save_model
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


def _build_datamodule(
    *,
    dataset: DatasetName,
    batch_size: int,
    val_size: float,
    test_size: float,
    use_processed_data: bool,
    remove_sparse: bool,
    min_interactions: int,
) -> ElearningDataModule:
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
    return dm


def _build_retrieval_config(dm: ElearningDataModule) -> RetrievalConfig:
    return RetrievalConfig(
        num_users=dm.num_users,
        num_items=dm.num_items,
        num_ctx_feats=dm.train_ds.num_ctx_feats,
        num_user_dense_feats=dm.num_user_dense_feats,
        num_item_dense_feats=dm.num_item_dense_feats,
        user_cat_cardinalities=dm.user_cat_cardinalities,
        item_cat_cardinalities=dm.item_cat_cardinalities,
    )


def _build_ranker_config(dm: ElearningDataModule) -> GhostConfig:
    return GhostConfig(
        num_users=dm.num_users,
        num_items=dm.num_items,
        num_ctx_feats=dm.train_ds.num_ctx_feats,
        num_user_dense_feats=dm.num_user_dense_feats,
        num_item_dense_feats=dm.num_item_dense_feats,
        user_cat_cardinalities=dm.user_cat_cardinalities,
        item_cat_cardinalities=dm.item_cat_cardinalities,
    )


def _test_best_checkpoint(
    trainer,
    dm: ElearningDataModule,
    debug: bool,
) -> dict[str, float] | None:
    if debug:
        print("[TRAIN] Debug mode: Skipping evaluation")
        return None

    return cast(
        dict[str, float],
        trainer.test(
            ckpt_path="best",
            datamodule=dm,
            weights_only=False,
        )[0],
    )


def _save_trained_model(
    *,
    cfg: GhostConfig | RetrievalConfig,
    dataset: DatasetName,
    models_folder: str,
    best_model_path: Path,
    metrics: dict[str, float] | None,
    save: bool,
) -> None:
    if not save or metrics is None:
        return

    model_file_path, model_config_path, metrics_path = save_model(
        model_config=cfg,
        dataset_name=dataset.value,
        best_model_path=best_model_path,
        models_folder=models_folder,
        metrics=metrics,
    )

    if config.state["verbose"]:
        print(f"Model weights saved in: {model_file_path}")
        print(f"Model config saved in: {model_config_path}")
        print(f"Metrics saved in: {metrics_path}")


def _train_retrieval(
    *,
    dm: ElearningDataModule,
    dataset: DatasetName,
    lr: float,
    top_k: int,
    epochs: int,
    patience: int,
    debug: bool,
    use_logger: bool,
    save: bool,
    models_folder: str,
) -> tuple[Retrieval, RetrievalConfig]:
    dm.setup(phase=Phase.RETRIEVAL)
    _print_datamodule_stats(dm)

    cfg = _build_retrieval_config(dm)

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

    metrics = _test_best_checkpoint(trainer, dm, debug)
    if debug:
        return retrieval, cfg

    best_retrieval = Retrieval.load_from_checkpoint(
        checkpoint_path=str(best_model_path),
        cfg=cfg,
        u_static_feats=dm.u_static_feats,
        i_static_feats=dm.i_static_feats,
        top_k=top_k,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    _save_trained_model(
        cfg=cfg,
        dataset=dataset,
        models_folder=models_folder,
        best_model_path=best_model_path,
        metrics=metrics,
        save=save,
    )
    return best_retrieval, cfg


def _load_retrieval(
    *,
    dm: ElearningDataModule,
    dataset: DatasetName,
    top_k: int,
    models_folder: str,
) -> tuple[Retrieval, RetrievalConfig]:
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
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    if config.state["verbose"]:
        print(f"[TRAIN] Loaded retrieval model from: {retrieval_model_path}")

    return retrieval, retrieval_cfg


def _train_ranker(
    *,
    dm: ElearningDataModule,
    dataset: DatasetName,
    lr: float,
    top_k: int,
    epochs: int,
    patience: int,
    adaptive_k: bool,
    debug: bool,
    use_logger: bool,
    save: bool,
    models_folder: str,
) -> tuple[Ranker, GhostConfig]:
    dm.setup(phase=Phase.RANKING)

    assert dm.u_static_feats is not None and dm.i_static_feats is not None

    cfg = _build_ranker_config(dm)
    ranker = Ranker(
        cfg=cfg,
        inter_graph=dm.create_inter_graph(),
        u_static_feats=dm.u_static_feats,
        i_static_feats=dm.i_static_feats,
        lr=lr,
        top_k=top_k,
        adaptive_k=adaptive_k,
    )

    trainer, best_model_path = train_model(
        model=ranker,
        dm=dm,
        top_k=top_k,
        debug=debug,
        use_logger=use_logger,
        epochs=epochs,
        patience=patience,
        monitor=f"val/NDCG@{top_k}",
    )

    metrics = _test_best_checkpoint(trainer, dm, debug)
    _save_trained_model(
        cfg=cfg,
        dataset=dataset,
        models_folder=models_folder,
        best_model_path=best_model_path,
        metrics=metrics,
        save=save,
    )
    return ranker, cfg


@app.command(name="train_retrieval", help="Train the retrieval model.")
def train_retrieval_command(
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ] = DatasetName.MARS,
    epochs: Annotated[
        int, typer.Option("--epochs", "-e", help="Number of epochs")
    ] = config.EPOCHS,
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
    dm = _build_datamodule(
        dataset=dataset,
        batch_size=batch_size,
        val_size=val_size,
        test_size=test_size,
        use_processed_data=use_processed_data,
        remove_sparse=remove_sparse,
        min_interactions=min_interactions,
    )
    _train_retrieval(
        dm=dm,
        dataset=dataset,
        lr=lr,
        top_k=top_k,
        epochs=epochs,
        patience=patience,
        debug=debug,
        use_logger=use_logger,
        save=save,
        models_folder=models_folder,
    )


@app.command(name="train_ranker", help="Train the reranker model.")
def train_ranker_command(
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ] = DatasetName.MARS,
    epochs: Annotated[
        int, typer.Option("--epochs", "-e", help="Number of epochs")
    ] = config.EPOCHS,
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
    dm = _build_datamodule(
        dataset=dataset,
        batch_size=batch_size,
        val_size=val_size,
        test_size=test_size,
        use_processed_data=use_processed_data,
        remove_sparse=remove_sparse,
        min_interactions=min_interactions,
    )
    dm.setup(phase=Phase.RETRIEVAL)
    _print_datamodule_stats(dm)

    retrieval, _ = _load_retrieval(
        dm=dm,
        dataset=dataset,
        top_k=top_k,
        models_folder=models_folder,
    )

    assert dm.i_static_feats is not None

    if config.state["verbose"]:
        print(f"[TRAIN] Generating top-{candidate_top_n} candidates per query")

    generate_candidates(
        retrieval=retrieval,
        dm=dm,
        top_n=candidate_top_n,
        i_static_feats=dm.i_static_feats,
    )

    _train_ranker(
        dm=dm,
        dataset=dataset,
        lr=lr,
        top_k=top_k,
        epochs=epochs,
        patience=patience,
        adaptive_k=adaptive_k,
        debug=debug,
        use_logger=use_logger,
        save=save,
        models_folder=models_folder,
    )


@app.command(name="train", help="Train retrieval and ranker end-to-end.")
def train_all_command(
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ] = DatasetName.MARS,
    retrieval_epochs: Annotated[
        int,
        typer.Option("--retrieval-epochs", help="Number of retrieval training epochs"),
    ] = config.EPOCHS,
    retrieval_lr: Annotated[
        float,
        typer.Option("--retrieval-lr", help="Retrieval learning rate"),
    ] = config.RETRIEVAL_LR,
    retrieval_batch_size: Annotated[
        int,
        typer.Option("--retrieval-batch-size", help="Retrieval batch size"),
    ] = config.RETRIEVAL_BATCH_SIZE,
    retrieval_patience: Annotated[
        int,
        typer.Option("--retrieval-patience", help="Retrieval early stopping"),
    ] = config.RETRIEVAL_PATIENCE,
    retrieval_top_k: Annotated[
        int,
        typer.Option("--retrieval-top-k", help="Retrieval top-k metric"),
    ] = config.RETRIEVAL_TOP_K,
    ranker_epochs: Annotated[
        int,
        typer.Option("--ranker-epochs", help="Number of ranker training epochs"),
    ] = config.EPOCHS,
    ranker_lr: Annotated[
        float,
        typer.Option("--ranker-lr", help="Ranker learning rate"),
    ] = config.RANKER_LR,
    ranker_batch_size: Annotated[
        int,
        typer.Option("--ranker-batch-size", help="Ranker batch size"),
    ] = config.RANKER_BATCH_SIZE,
    ranker_patience: Annotated[
        int,
        typer.Option("--ranker-patience", help="Ranker early stopping"),
    ] = config.RANKER_PATIENCE,
    ranker_top_k: Annotated[
        int,
        typer.Option("--ranker-top-k", help="Ranker top-k metric"),
    ] = config.RANKER_TOP_K,
    candidate_top_n: Annotated[
        int,
        typer.Option(
            "--candidate-top-n",
            help="Number of retrieval candidates generated before reranking",
        ),
    ] = config.TOP_N,
    val_size: Annotated[
        float, typer.Option("--val_size", "-v", help="Validation size")
    ] = config.VAL_RATIO,
    test_size: Annotated[
        float, typer.Option("--test_size", "-t", help="Test size")
    ] = config.TEST_RATIO,
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
    retrain_retrieval: Annotated[
        bool,
        typer.Option(
            "--retrain-retrieval",
            help="Train a fresh retrieval model even if one already exists on disk",
        ),
    ] = False,
    use_logger: Annotated[
        bool, typer.Option("--use_logger", "-L", help="Use MLFlow logger")
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", "-D", help="Debug mode")] = False,
    save: Annotated[
        bool, typer.Option("--save_model", "-S", help="Save trained models")
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
    dm = _build_datamodule(
        dataset=dataset,
        batch_size=retrieval_batch_size,
        val_size=val_size,
        test_size=test_size,
        use_processed_data=use_processed_data,
        remove_sparse=remove_sparse,
        min_interactions=min_interactions,
    )

    if retrain_retrieval:
        if config.state["verbose"]:
            print("[TRAIN] Forced retrieval retraining.")
        retrieval, _ = _train_retrieval(
            dm=dm,
            dataset=dataset,
            lr=retrieval_lr,
            top_k=retrieval_top_k,
            epochs=retrieval_epochs,
            patience=retrieval_patience,
            debug=debug,
            use_logger=use_logger,
            save=save,
            models_folder=models_folder,
        )
    else:
        dm.setup(phase=Phase.RETRIEVAL)
        _print_datamodule_stats(dm)
        try:
            retrieval, _ = _load_retrieval(
                dm=dm,
                dataset=dataset,
                top_k=retrieval_top_k,
                models_folder=models_folder,
            )
        except FileNotFoundError:
            if config.state["verbose"]:
                print("[TRAIN] No saved retrieval model found. Training a new one.")
            retrieval, _ = _train_retrieval(
                dm=dm,
                dataset=dataset,
                lr=retrieval_lr,
                top_k=retrieval_top_k,
                epochs=retrieval_epochs,
                patience=retrieval_patience,
                debug=debug,
                use_logger=use_logger,
                save=save,
                models_folder=models_folder,
            )

    assert dm.i_static_feats is not None

    if config.state["verbose"]:
        print(f"[TRAIN] Generating top-{candidate_top_n} candidates per query")

    generate_candidates(
        retrieval=retrieval,
        dm=dm,
        top_n=candidate_top_n,
        i_static_feats=dm.i_static_feats,
    )

    dm.batch_size = ranker_batch_size
    _train_ranker(
        dm=dm,
        dataset=dataset,
        lr=ranker_lr,
        top_k=ranker_top_k,
        epochs=ranker_epochs,
        patience=ranker_patience,
        adaptive_k=adaptive_k,
        debug=debug,
        use_logger=use_logger,
        save=save,
        models_folder=models_folder,
    )
