from pathlib import Path
from typing import cast

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from edurec import config

from ..datasets import ElearningDataModule
from ..recsys import Reranker, Retrieval


def train_reranker(
    reranker: Reranker,
    dm: ElearningDataModule,
    top_k: int,
    debug: bool,
    use_logger: bool,
    epochs: int,
    patience: int,
) -> tuple[L.Trainer, Path]:
    model_name = reranker.model_name

    reranker = (
        cast(Reranker, torch.compile(reranker))
        if not debug and config.COMPILE_MODEL
        else reranker
    )

    early_stopping = EarlyStopping(
        monitor=f"val/NDCG@{top_k}",
        patience=patience,
        mode="max",
        min_delta=config.DELTA,
        verbose=True,
    )
    checkpoint = ModelCheckpoint(
        monitor=f"val/NDCG@{top_k}",
        mode="max",
        save_top_k=1,
        filename=f"best_{model_name}",
    )

    train_logger = None
    if use_logger and not debug:
        train_logger = WandbLogger(
            project=config.EXPERIMENT_NAME,
            name=f"train_{dm.dataset_name}_top-{top_k}",
        )

    trainer = L.Trainer(
        logger=train_logger,
        max_epochs=epochs,
        accelerator=config.state["device"],
        devices="auto",
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint],
        fast_dev_run=debug,
    )

    trainer.fit(reranker, datamodule=dm)

    return trainer, Path(checkpoint.best_model_path)


def train_retrieval(
    retrieval: Retrieval,
    dm: ElearningDataModule,
    top_k: int,
    debug: bool,
    use_logger: bool,
    epochs: int,
    patience: int,
) -> tuple[L.Trainer, Path]:
    model_name = retrieval.model_name

    retrieval = (
        cast(Retrieval, torch.compile(retrieval))
        if not debug and config.COMPILE_MODEL
        else retrieval
    )

    early_stopping = EarlyStopping(
        monitor=f"val/Recall@{top_k}",
        patience=patience,
        mode="max",
        min_delta=config.DELTA,
        verbose=True,
    )
    checkpoint = ModelCheckpoint(
        monitor=f"val/Recall@{top_k}",
        mode="max",
        save_top_k=1,
        filename=f"best_{model_name}",
    )

    train_logger = None
    if use_logger and not debug:
        train_logger = WandbLogger(
            project=config.EXPERIMENT_NAME,
            name=f"train_retrieval_{dm.dataset_name}_top-{top_k}",
        )

    trainer = L.Trainer(
        logger=train_logger,
        max_epochs=epochs,
        accelerator=config.state["device"],
        devices="auto",
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint],
        fast_dev_run=debug,
    )

    trainer.fit(retrieval, datamodule=dm)

    return trainer, Path(checkpoint.best_model_path)
