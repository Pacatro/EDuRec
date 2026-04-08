from pathlib import Path
from typing import cast

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from edurec import config

from ..datasets import ElearningDataModule
from .engine import RecSys


def train_recsys(
    recsys: RecSys,
    dm: ElearningDataModule,
    top_k: int,
    debug: bool,
    use_logger: bool,
    epochs: int,
    patience: int,
    monitor: str,
) -> tuple[L.Trainer, Path]:
    model_name = recsys.model_name

    recsys = (
        cast(RecSys, torch.compile(recsys))
        if not debug and config.COMPILE_MODEL
        else recsys
    )

    resolved_monitor = monitor if "Loss" in monitor else f"{monitor}@{top_k}"
    mode = "min" if "Loss" in monitor else "max"

    early_stopping = EarlyStopping(
        monitor=resolved_monitor,
        patience=patience,
        mode=mode,
        min_delta=config.DELTA,
        verbose=True,
    )
    checkpoint = ModelCheckpoint(
        monitor=resolved_monitor,
        mode=mode,
        save_top_k=1,
        filename=f"best_{model_name}",
    )

    train_logger = None
    if use_logger and not debug:
        train_logger = WandbLogger(
            project=config.EXPERIMENT_NAME,
            name=f"train_{model_name}_top-{top_k}",
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

    trainer.fit(recsys, datamodule=dm)

    return trainer, Path(checkpoint.best_model_path)
