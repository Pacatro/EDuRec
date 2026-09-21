from pathlib import Path

import pandas as pd

from .. import settings
from ..datasets import ElearningDataModule
from ..recsys import ModelConfig, RecSys, train_model
from ..recsys.configs import TrainConfig


def eval_model(
    dm: ElearningDataModule,
    cfg: ModelConfig,
    train_cfg: TrainConfig | None = None,
    val_topk: int = settings.TOP_K,
    compile: bool = settings.COMPILE_MODEL,
    results_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    recsys = RecSys(
        cfg=cfg,
        inter_graph=dm.build_inter_graph(),
        u_static_feats=dm.u_static_feats,
        i_static_feats=dm.i_static_feats,
        train_cfg=train_cfg,
        val_topk=val_topk,
    )

    trainer, _, timer = train_model(
        model=recsys,
        dm=dm,
        debug=False,
        epochs=train_cfg.epochs if train_cfg is not None else settings.EPOCHS,
        patience=train_cfg.patience if train_cfg is not None else settings.PATIENCE,
        monitor=recsys.monitor,
        compile=compile,
        verbose=verbose,
    )

    metrics = trainer.test(ckpt_path="best", datamodule=dm, weights_only=False)[0]
    metrics = {name.removeprefix("test/"): value for name, value in metrics.items()}
    results = pd.DataFrame(
        [
            {
                "model": "EDuRec",
                **metrics,
                "training_time_s": timer.time_elapsed("train"),
                "inference_time_s": timer.time_elapsed("test"),
            }
        ]
    )

    if results_path is not None:
        results.to_csv(results_path / "EDuRec.csv", index=True)

    return results
