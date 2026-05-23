from pathlib import Path
from typing import cast

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .. import settings
from ..datasets import ElearningDataModule
from .recsys import RecSys


def train_model(
    model: RecSys,
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
        cast(RecSys, torch.compile(model))
        if not debug and settings.COMPILE_MODEL
        else model
    )

    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode="max",
        min_delta=settings.DELTA,
        verbose=True,
    )
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode="max",
        save_top_k=1,
        filename=f"best_{model_name}",
        save_weights_only=True,
    )

    logger = (
        WandbLogger(
            project=settings.EXPERIMENT_NAME,
            name=f"train_{model_name}_{dm.dataset_name}_top-{top_k}",
        )
        if use_logger and not debug
        else None
    )

    trainer = L.Trainer(
        logger=logger,
        max_epochs=epochs,
        accelerator=settings.state["device"],
        devices="auto",
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint],
        fast_dev_run=debug,
    )

    trainer.fit(model, datamodule=dm)

    return trainer, Path(checkpoint.best_model_path)
