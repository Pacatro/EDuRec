import shutil
from enum import Enum
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed
from sklearn.model_selection import KFold, LeaveOneOut
from torch import nn

from .. import config
from ..datasets.utils import get_column_types, global_preprocessing
from ..training.engine import RecSys
from ..training.model import EDuRecConfig
from .cv_datamodule import CvElearningDataModule


class CVType(str, Enum):
    KFOLD = "kfold"
    LOO = "loo"


def cross_validate(
    df: pd.DataFrame,
    model_class: type[nn.Module],
    lr: float = config.LR,
    n_splits: int = config.K,
    epochs: int = config.EPOCHS,
    cv_type: CVType = CVType.KFOLD,
    top_k: int = config.TOP_K,
    batch_size: int = config.BATCH_SIZE,
    patience: int = config.PATIENCE,
    delta: float = config.DELTA,
    monitor: str = config.MONITOR,
    verbose: bool = config.state["verbose"],
) -> pd.DataFrame:
    cv = (
        KFold(
            n_splits=n_splits, random_state=config.state["random_state"], shuffle=True
        )
        if cv_type == CVType.KFOLD
        else LeaveOneOut()
    )

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

        recsys = RecSys(
            model=model,
            top_k=top_k,
            lr=lr,
            monitor=monitor,
            rating_loss_fn=nn.SmoothL1Loss(),
        )

        earlystop = EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode="min",
            min_delta=delta,
            verbose=verbose,
        )

        ckpt = ModelCheckpoint(
            monitor=monitor,
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
            ckpt.best_model_path, model=model, top_k=top_k, monitor=monitor
        )

        metrics = trainer.validate(recsys, datamodule=dm)[0]

        fold_metrics.append(metrics)

    all_metrics = pd.DataFrame(fold_metrics)
    avg_metrics = all_metrics.mean()
    std_metrics = all_metrics.std()

    return pd.DataFrame({"mean": avg_metrics, "std": std_metrics})


def _calc_fbeta_score(precision: float, recall: float, beta: float) -> float:
    return (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)


def _create_inter_dataset(
    train_processed: pd.DataFrame,
    test_processed: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    fold_path: Path,
) -> list[str]:
    rename_map = {}
    for c in numeric_cols:
        rename_map[c] = f"{c}:float"
    for c in categorical_cols:
        rename_map[c] = f"{c}:token"

    rename_map[config.USER_COL] = f"{config.USER_COL}:token"
    rename_map[config.ITEM_COL] = f"{config.ITEM_COL}:token"
    rename_map[config.TIME_COL] = f"{config.TIME_COL}:float"
    rename_map[config.RATING_COL] = f"{config.RATING_COL}:float"
    rename_map[config.RELEVANT_COL] = f"{config.RELEVANT_COL}:float"

    train_rec = train_processed.rename(columns=rename_map)
    test_rec = test_processed.rename(columns=rename_map)

    processed_cols = test_processed.columns.tolist()

    if fold_path.exists():
        shutil.rmtree(fold_path)

    fold_path.mkdir(parents=True)

    full_df = pd.concat([train_rec, test_rec], ignore_index=True)
    full_df.columns = [c.strip() for c in full_df.columns]
    full_df.to_csv(fold_path / f"{fold_path.name}.inter", sep="\t", index=False)

    return processed_cols


def sota_cross_validate(
    df: pd.DataFrame,
    model_class: str,
    lr: float = config.LR,
    n_splits: int = config.K,
    epochs: int = config.EPOCHS,
    cv_type: CVType = CVType.KFOLD,
    top_k: int = config.TOP_K,
    batch_size: int = config.BATCH_SIZE,
    patience: int = config.PATIENCE,
    extra_config: dict[str, Any] | None = None,
    verbose: bool = config.state["verbose"],
    **kwargs,
):
    _ = kwargs
    device = "cuda" if config.state["device"] == "auto" else "cpu"

    cv = (
        KFold(
            n_splits=n_splits, random_state=config.state["random_state"], shuffle=True
        )
        if cv_type == CVType.KFOLD
        else LeaveOneOut()
    )

    n_folds = cv.get_n_splits(X=df)

    test_size = 1 / n_folds
    train_size = 1 - test_size

    id_cols = [config.USER_COL, config.ITEM_COL]
    numeric_cols, categorical_lengths = get_column_types(df, id_cols)
    categorical_cols = list(categorical_lengths.keys())

    global_preprocessing(df)

    all_fold_results = []

    data_path = Path(config.DATA_FOLDER) / "sota_cv_data"

    for fold, (train_idx, test_idx) in enumerate(cv.split(df), start=1):
        print(f"Fold {fold}/{n_folds}")

        dm = CvElearningDataModule(
            df=df, batch_size=batch_size, train_idx=train_idx, val_idx=test_idx
        )

        fold_dataset_name = f"fold_{fold}"
        fold_path = data_path / fold_dataset_name

        processed_cols = _create_inter_dataset(
            train_processed=dm.train_df,
            test_processed=dm.val_df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            fold_path=fold_path,
        )

        print(f"Processed cols: {processed_cols}")

        parameter_dict = {
            "dataset": fold_dataset_name,
            "data_path": data_path,
            "model": model_class,
            "learning_rate": lr,
            "epochs": epochs,
            "stopping_step": patience,
            "train_batch_size": batch_size,
            "metrics": ["Precision", "Recall", "NDCG", "Hit", "MAP", "MRR"],
            "topk": [top_k],
            "valid_metric": f"NDCG@{top_k}",
            "eval_args": {
                "split": {"RS": [train_size, 0, test_size]},
                "order": "RO",
                "mode": "full",
            },
            "USER_ID_FIELD": config.USER_COL,
            "ITEM_ID_FIELD": config.ITEM_COL,
            "TIME_FIELD": config.TIME_COL,
            "RATING_FIELD": config.RATING_COL,
            "LABEL_FIELD": config.RELEVANT_COL,
            "load_col": {"inter": processed_cols},
            "checkpoint_dir": fold_path / "checkpoints",
            "seed": 42,
            "filter_net_empty": False,  # Not remove users without interactions
            "transform": None,  # Not apply transformations
        }

        if extra_config is not None:
            parameter_dict.update(extra_config)

        config_obj = Config(
            model=model_class, dataset=fold_dataset_name, config_dict=parameter_dict
        )

        init_seed(config_obj["seed"], config_obj["reproducibility"])

        dataset_obj = create_dataset(config_obj)
        train_data, _, test_data = data_preparation(config_obj, dataset_obj)
        model_obj = get_model(config_obj["model"])(config_obj, dataset_obj).to(device)
        trainer = get_trainer(config_obj["MODEL_TYPE"], config_obj["model"])(
            config_obj, model_obj
        )

        trainer.fit(train_data, None, saved=True, show_progress=verbose)

        fold_result = trainer.evaluate(
            test_data, load_best_model=True, show_progress=verbose
        )

        fold_result[f"f1@{top_k}"] = _calc_fbeta_score(
            fold_result[f"precision@{top_k}"],
            fold_result[f"recall@{top_k}"],
            beta=1.0,
        )

        all_fold_results.append(fold_result)

        # Cleanup folds to save space
        shutil.rmtree(fold_path, ignore_errors=True)

    all_metrics = pd.DataFrame(all_fold_results)
    avg_metrics = all_metrics.mean()
    std_metrics = all_metrics.std()

    return pd.DataFrame({"mean": avg_metrics, "std": std_metrics})
