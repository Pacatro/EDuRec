from collections.abc import Sequence
from pathlib import Path
from typing import cast

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import WandbLogger

from .. import settings
from ..datasets import ElearningDataModule


def train_model(
    model: L.LightningModule,
    dm: ElearningDataModule,
    debug: bool,
    epochs: int,
    patience: int,
    monitor: str,
    experiment_name: str | None = None,
    compile: bool = settings.COMPILE_MODEL,
    verbose: bool = False,
    callbacks: Sequence[Callback] = (),
    default_root_dir: Path | str | None = None,
) -> tuple[L.Trainer, Path, Timer]:
    model_name = model.model_name

    if compile:
        model = cast(L.LightningModule, torch.compile(model))

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
        WandbLogger(project=settings.EXPERIMENT_NAME, name=experiment_name)
        if experiment_name is not None and not debug
        else None
    )

    timer = Timer()

    trainer = L.Trainer(
        logger=logger,
        # profiler="simple",
        max_epochs=epochs,
        accelerator=settings.state["device"],
        devices="auto",
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint, timer, *callbacks],
        fast_dev_run=debug,
        enable_progress_bar=verbose,
        default_root_dir=default_root_dir,
    )

    trainer.fit(model, datamodule=dm)

    return trainer, Path(checkpoint.best_model_path), timer
