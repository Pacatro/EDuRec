import pandas as pd
import lightning as L
from sklearn.model_selection import KFold, LeaveOneOut
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from enum import Enum

from .data import ElearningDataModule
from .engine import RecSys


class CVType(str, Enum):
    kfold = "kfold"
    loo = "loo"


def cross_validate(
    df: pd.DataFrame,
    model_class: type,
    lr: float = 0.001,
    n_splits: int = 5,
    random_state: int = 42,
    epochs: int = 100,
    cv_type: CVType = CVType.kfold,
    top_k: int = 10,
    batch_size: int = 128,
    patience: int = 5,
    delta: float = 0.001,
    ignored_cols: list[str] = [],
    verbose: bool = False,
) -> pd.DataFrame:
    """Perform cross-validation for the given model.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        model_class (type): The class of the model to be evaluated.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.001.
        n_splits (int, optional): The number of splits for the cross-validation. Defaults to 5.
        random_state (int, optional): The random state for the cross-validation. Defaults to 42.
        epochs (int, optional): The number of epochs for the training. Defaults to 100.
        cv_type (Literal["kfold", "loo"], optional): The type of cross-validation to use. Defaults to "kfold".
        k (int, optional): The number of recommendations to make. Defaults to 10.
        batch_size (int, optional): The batch size for training. Defaults to 128.
        patience (int, optional): The patience for early stopping. Defaults to 5.
        delta (float, optional): The minimum change to qualify as an improvement. Defaults to 0.001.
        ignored_cols (list[str], optional): The columns to ignore during training. Defaults to [].
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        pd.DataFrame: The cross-validation results.
    """
    cv = (
        KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        if cv_type == CVType.kfold
        else LeaveOneOut()
    )

    fold_metrics = []
    n_folds = cv.get_n_splits(X=df)

    for fold, (train_idx, test_idx) in enumerate(cv.split(df), start=1):
        print(f"Fold {fold}/{n_folds}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[test_idx].reset_index(drop=True)

        dm = ElearningDataModule(
            df=pd.concat([train_df, val_df], ignore_index=True),
            batch_size=batch_size,
            test_size=0,
            val_size=len(val_df) / (len(train_df) + len(val_df)),
        )

        model_config = {
            "n_users": dm.num_users,
            "n_items": dm.num_items,
            "cat_cardinalities": dm.cat_cardinalities,
            "cont_features": dm.cont_features,
        }

        model = model_class(**model_config)

        recsys = RecSys(model=model, threshold=dm.threshold, lr=lr)

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

    result = pd.DataFrame({"mean": avg_metrics, "std": std_metrics})
    return result
