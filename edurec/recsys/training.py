from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import lightning as L
import mlflow
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.data.pandas_dataset import from_pandas

from edurec import settings
from edurec.datasets import ElearningDataModule


def log_datasets(logger: MLFlowLogger, dm: ElearningDataModule) -> None:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=logger.run_id):
        for context, df in dm.artifacts.splits().items():
            dataset = from_pandas(
                df,
                name=f"{dm.data_variant}",
                source=f"{settings.PROCESSED_FOLDER}/{dm.data_variant}",
                targets=settings.RATING_COL if dm.is_explicit else settings.ITEM_COL,
            )
            mlflow.log_input(dataset, context=context)


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
    limit_val_batches: float | None = None,
) -> tuple[L.Trainer, Path, Timer]:
    model_name = type(model).__name__

    if compile:
        model = cast(L.LightningModule, torch.compile(model))

    mode = "min" if monitor.lower().endswith("loss") else "max"

    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        min_delta=settings.DELTA,
        verbose=True,
    )
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        filename=f"best_{model_name}",
        save_weights_only=True,
    )

    logger = (
        MLFlowLogger(
            experiment_name=settings.EXPERIMENT_NAME,
            run_name=experiment_name,
            tracking_uri=settings.MLFLOW_TRACKING_URI,
        )
        if experiment_name is not None and not debug
        else None
    )

    timer = Timer()
    if logger is not None and default_root_dir is None:
        default_root_dir = Path("mlflow_checkpoints")

    trainer_kwargs: dict[str, Any] = {}
    if limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = limit_val_batches

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
        **trainer_kwargs,
    )

    if logger is not None:
        log_datasets(logger, dm)

    trainer.fit(model, datamodule=dm)

    return trainer, Path(checkpoint.best_model_path), timer
