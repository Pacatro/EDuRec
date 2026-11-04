from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import MLFlowLogger

from edurec import settings
from edurec.datasets import ElearningDataModule
from edurec.recsys.configs import monitor_mode


def _resolve_limit_val_batches(
    dm: ElearningDataModule, limit_val_batches: float | None
) -> int | float | None:
    """Resolve a validation-batch fraction to an explicit batch count.

    Lightning rejects a fraction that maps to fewer than one batch.
    """
    if limit_val_batches is None or not 0 < limit_val_batches < 1:
        return limit_val_batches
    num_val_batches = len(dm.val_dataloader())
    return max(1, int(limit_val_batches * num_val_batches))


def train_model(
    model: L.LightningModule,
    dm: ElearningDataModule,
    debug: bool,
    epochs: int,
    patience: int,
    monitor: str,
    run_name: str | None = None,
    experiment_name: str = settings.EXPERIMENT_NAME,
    compile: bool = settings.COMPILE_MODEL,
    verbose: bool = False,
    callbacks: Sequence[Callback] = (),
    default_root_dir: Path | str | None = None,
    limit_val_batches: float | None = None,
    logger: MLFlowLogger | None = None,
    params: Mapping[str, Any] | None = None,
) -> tuple[L.Trainer, Path, Timer]:
    dm.validate_topks(
        [*getattr(model, "topks", []), getattr(model, "val_topk", settings.TOP_K)]
    )
    model_name = type(model).__name__

    # ``torch.compile`` returns a wrapper rather than a ``LightningModule``, but
    # the wrapper exposes the same training API.
    module: Any = torch.compile(model) if compile else model

    mode = monitor_mode(monitor)

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

    if debug:
        logger = None
    elif logger is None and run_name is not None:
        logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=settings.MLFLOW_TRACKING_URI,
        )

    # Extra run parameters (e.g. the ablation variant) that are not part of the
    # model's own hyperparameters.
    if logger is not None and params:
        logger.log_hyperparams(dict(params))

    timer = Timer()
    if logger is not None and default_root_dir is None:
        default_root_dir = Path("mlflow_checkpoints")

    trainer_kwargs: dict[str, Any] = {}
    limit = _resolve_limit_val_batches(dm, limit_val_batches)
    if limit is not None:
        trainer_kwargs["limit_val_batches"] = limit

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

    trainer.fit(module, datamodule=dm)

    return trainer, Path(checkpoint.best_model_path), timer
