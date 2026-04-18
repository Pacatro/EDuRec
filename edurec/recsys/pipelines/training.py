from pathlib import Path
from typing import cast

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from edurec import config

from ...datasets import ElearningDataModule
from .. import Ranker, Retrieval

type Model = Ranker | Retrieval


def train_model(
    model: Model,
    dm: ElearningDataModule,
    top_k: int,
    debug: bool,
    use_logger: bool,
    epochs: int,
    patience: int,
    monitor: str,
) -> tuple[L.Trainer, Path]:
    model_name = model.model_name

    model = (
        cast(Model, torch.compile(model))
        if not debug and config.COMPILE_MODEL
        else model
    )

    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode="max",
        min_delta=config.DELTA,
        verbose=True,
    )
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode="max",
        save_top_k=1,
        filename=f"best_{model_name}",
    )

    train_logger = None
    if use_logger and not debug:
        train_logger = WandbLogger(
            project=config.EXPERIMENT_NAME,
            name=f"train_{model_name}_{dm.dataset_name}_top-{top_k}",
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

    trainer.fit(model, datamodule=dm)

    return trainer, Path(checkpoint.best_model_path)
