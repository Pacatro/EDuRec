from collections.abc import Sequence
from pathlib import Path
from typing import cast

import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch

from .. import settings
from ..datasets import ElearningDataModule


def train_model(
    model: L.LightningModule,
    dm: ElearningDataModule,
    debug: bool,
    use_logger: bool,
    epochs: int,
    patience: int,
    monitor: str,
    compile: bool = settings.COMPILE_MODEL,
    verbose: bool = False,
    callbacks: Sequence[Callback] = (),
    default_root_dir: Path | str | None = None,
) -> tuple[L.Trainer, Path]:
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
        WandbLogger(
            project=settings.EXPERIMENT_NAME,
            name=f"train_{model_name}_{dm.dataset_name}",
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
        callbacks=[early_stopping, checkpoint, *callbacks],
        fast_dev_run=debug,
        enable_progress_bar=verbose,
        default_root_dir=default_root_dir,
    )

    trainer.fit(model, datamodule=dm)

    return trainer, Path(checkpoint.best_model_path)
