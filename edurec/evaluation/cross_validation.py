from enum import Enum
from typing import Any

import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold, LeaveOneOut
from torch import nn

from .. import config
from ..datasets import DatasetName, load_data
from ..training.engine import RecSys
from ..training.model import EDuRecConfig
from .cv_datamodule import CvElearningDataModule


class CVType(str, Enum):
    KFOLD = "kfold"
    LOO = "loo"


def cross_validate(
    dataset_name: DatasetName,
    model_class: type[nn.Module],
    lr: float = config.LR,
    n_splits: int = config.K,
    epochs: int = config.EPOCHS,
    cv_type: CVType = CVType.KFOLD,
    top_k: int = config.TOP_K,
    batch_size: int = config.BATCH_SIZE,
    patience: int = config.PATIENCE,
    delta: float = config.DELTA,
    verbose: bool = config.state["verbose"],
) -> pd.DataFrame:
    cv = (
        KFold(
            n_splits=n_splits, random_state=config.state["random_state"], shuffle=True
        )
        if cv_type == CVType.KFOLD
        else LeaveOneOut()
    )

    df = load_data(dataset_name)

    fold_metrics = []
    n_folds = cv.get_n_splits(X=df)

    for fold, (train_idx, val_idx) in enumerate(cv.split(df), start=1):
        print(f"Fold {fold}/{n_folds}")

        dm = CvElearningDataModule(df, batch_size, train_idx, val_idx)

        model_config = EDuRecConfig(
            n_users=dm.num_users,
            n_items=dm.num_items,
            cat_cardinalities=dm.cat_cardinalities,
            numeric_features=dm.numeric_features,
        )

        model = model_class(model_config)

        recsys = RecSys(model=model, top_k=top_k, threshold=dm.threshold, lr=lr)

        earlystop = EarlyStopping(
            monitor="val/MSE",
            patience=patience,
            mode="min",
            min_delta=delta,
            verbose=verbose,
        )

        ckpt = ModelCheckpoint(
            monitor="val/MSE",
            mode="min",
            save_top_k=1,
            filename=f"fold{fold}_best_model",
        )

        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=config.state["device"],
            callbacks=[earlystop, ckpt],
            log_every_n_steps=10,
            enable_model_summary=False,
            inference_mode=False,
            enable_progress_bar=verbose,
        )

        trainer.fit(recsys, datamodule=dm)

        recsys = RecSys.load_from_checkpoint(
            ckpt.best_model_path,
            model=model,
            top_k=top_k,
            threshold=dm.threshold,
        )

        metrics = trainer.validate(recsys, datamodule=dm)[0]

        fold_metrics.append(metrics)

    all_metrics = pd.DataFrame(fold_metrics)
    avg_metrics = all_metrics.mean()
    std_metrics = all_metrics.std()

    return pd.DataFrame({"mean": avg_metrics, "std": std_metrics})


# TODO: Completar esta función
def recbole_cross_validate(
    dataset_name: DatasetName,
    model: str,
    threshold: float,
    lr: float = config.LR,
    n_splits: int = config.K,
    epochs: int = config.EPOCHS,
    cv_type: CVType = CVType.KFOLD,
    top_k: int = config.TOP_K,
    batch_size: int = config.BATCH_SIZE,
    data_path: str = "./recbole_cv_data",
    extra_config: dict[str, Any] | None = None,
    verbose: bool = config.state["verbose"],
):
    df = load_data(dataset_name)

    df["recommend"] = df[config.RATING_COL] >= threshold
