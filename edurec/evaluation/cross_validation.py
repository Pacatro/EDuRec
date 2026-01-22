from enum import Enum

import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold, LeaveOneOut
from torch import nn

from edurec.evaluation.cv_datamodule import CvElearningDataModule

from .. import config
from ..data.datasets import DatasetName, load_raw_data
from ..models.engine import RecSys


class CVType(str, Enum):
    kfold = ""
    loo = "loo"


def cross_validate(
    dataset_name: DatasetName,
    model_class: type[nn.Module],
    lr: float = 0.001,
    n_splits: int = 5,
    epochs: int = 100,
    cv_type: CVType = CVType.kfold,
    top_k: int = 10,
    batch_size: int = 128,
    patience: int = 5,
    delta: float = 0.001,
    verbose: bool = False,
) -> pd.DataFrame:
    cv = (
        KFold(
            n_splits=n_splits, random_state=config.state["random_state"], shuffle=True
        )
        if cv_type == CVType.kfold
        else LeaveOneOut()
    )

    df = load_raw_data(dataset_name)

    fold_metrics = []
    n_folds = cv.get_n_splits(X=df)

    for fold, (train_idx, val_idx) in enumerate(cv.split(df), start=1):
        print(f"Fold {fold}/{n_folds}")

        dm = CvElearningDataModule(dataset_name, batch_size, train_idx, val_idx)

        model_config = {
            "n_users": dm.num_users,
            "n_items": dm.num_items,
            "cat_cardinalities": dm.cat_cardinalities,
            "cont_features": dm.numeric_features,
        }

        model = model_class(**model_config)

        recsys = RecSys(
            model=model,
            min_rating=dm.min_rating,
            max_rating=dm.max_rating,
            top_k=top_k,
            threshold=dm.threshold,
            lr=lr,
        )

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
            devices="auto",
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
