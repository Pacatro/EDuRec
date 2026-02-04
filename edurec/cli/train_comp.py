import shutil
from pathlib import Path
from typing import Annotated

import lightning as L
import pandas as pd
import torch
import typer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from recbole.data import ModelType
from recbole.quick_start import run_recbole
from recbole.utils import get_model

from .. import config
from ..datasets import DatasetName
from ..training.datamodule import ElearningDataModule
from ..training.engine import RecSys
from ..training.model import EDuRecConfig, EDuRecMTL, EDuRecV1

app = typer.Typer(no_args_is_help=True)


def _create_inter_dataset(
    train_processed: pd.DataFrame,
    val_processed: pd.DataFrame,
    test_processed: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    dataset_name: str,
) -> tuple[list[str], Path, list[float]]:
    # 1. Definir rutas según tu estructura: config.DATA_FOLDER/inters/dataset_name/
    base_path = Path("inters") / dataset_name

    if base_path.exists():
        shutil.rmtree(base_path)
    base_path.mkdir(parents=True)

    # 2. Mapeo de nombres de columnas para RecBole
    rename_map = {c: f"{c}:float" for c in numeric_cols}
    rename_map.update({c: f"{c}:token" for c in categorical_cols})

    rename_map.update(
        {
            config.USER_COL: f"{config.USER_COL}:token",
            config.ITEM_COL: f"{config.ITEM_COL}:token",
            config.TIME_COL: f"{config.TIME_COL}:float",
            config.RATING_COL: f"{config.RATING_COL}:float",
            config.RELEVANT_COL: f"{config.RELEVANT_COL}:float",
        }
    )

    # 3. Guardar archivos individuales (opcional, para tu registro)
    train_rec = train_processed.rename(columns=rename_map)
    val_rec = val_processed.rename(columns=rename_map)
    test_rec = test_processed.rename(columns=rename_map)

    train_rec.to_csv(base_path / "train.inter", sep="\t", index=False)
    val_rec.to_csv(base_path / "val.inter", sep="\t", index=False)
    test_rec.to_csv(base_path / "test.inter", sep="\t", index=False)

    # 4. Crear el archivo unificado que RecBole requiere
    # Importante: El archivo debe llamarse igual que la carpeta
    full_df = pd.concat([train_rec, val_rec, test_rec], ignore_index=True)
    full_df.to_csv(base_path / f"{dataset_name}.inter", sep="\t", index=False)

    # Calcular ratios para que RecBole mantenga tu división original
    total = len(full_df)
    ratios = [len(train_rec) / total, len(val_rec) / total, len(test_rec) / total]

    processed_cols = [c.split(":")[0] for c in rename_map.values()]

    return processed_cols, base_path.parent, ratios


def train_sota(
    models: list[str],
    dm: ElearningDataModule,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    top_k: int,
) -> pd.DataFrame:
    dm.setup("fit")
    dm.setup("test")

    dataset_name = f"fold_{dm.dataset_name.value}"

    # Obtenemos los datos y la ruta base (inters/)
    assert dm.test_df is not None
    processed_cols, data_path, ratios = _create_inter_dataset(
        dm.train_df,
        dm.val_df,
        dm.test_df,
        numeric_cols=dm.numeric_features,
        categorical_cols=dm.categorical_features,
        dataset_name=dataset_name,
    )
    results = {}
    for model in models:
        print("[TRAIN] SOTA Model:", model)

        parameter_dict = {
            "dataset": dataset_name,
            "data_path": data_path,
            "model": model,
            "learning_rate": lr,
            "epochs": epochs,
            "stopping_step": patience,
            "train_batch_size": batch_size,
            "weight_decay": config.WEIGHT_DECAY,
            "metrics": ["Precision", "Recall", "NDCG", "Hit", "MAP", "MRR"],
            "topk": [top_k],
            "valid_metric": f"NDCG@{top_k}",
            "eval_args": {
                "split": {"RS": ratios},
                "order": "TO",
                "mode": "full",
            },
            "USER_ID_FIELD": config.USER_COL,
            "ITEM_ID_FIELD": config.ITEM_COL,
            "TIME_FIELD": config.TIME_COL,
            "RATING_FIELD": config.RATING_COL,
            "LABEL_FIELD": config.RELEVANT_COL,
            "load_col": {"inter": list(set(processed_cols))},  # Evitar duplicados
            "checkpoint_dir": data_path / dataset_name / "checkpoints",
            "train_neg_sample_args": {
                "strategy": "fixed",
                "sample_num": config.N_NEG_TRAIN,
                "distribution": "uniform",
            },
            "eval_neg_sample_args": {
                "strategy": "fixed",
                "sample_num": config.N_NEG_TEST,
                "distribution": "uniform",
            },
        }

        if config.state["random_state"] is not None:
            parameter_dict["seed"] = config.state["random_state"]
            parameter_dict["reproducibility"] = True

        model_class = get_model(model)

        if model_class.type == ModelType.SEQUENTIAL:
            parameter_dict["train_neg_sample_args"] = None
        elif model_class.type == ModelType.CONTEXT:
            processed_cols = [
                config.USER_COL,
                config.ITEM_COL,
                config.TIME_COL,
                config.RATING_COL,
                config.RELEVANT_COL,
            ]
            parameter_dict["load_col"] = {"inter": processed_cols}

        model_results = run_recbole(
            model=model, dataset=dataset_name, config_dict=parameter_dict
        )["test_result"]
        formatted_results = {
            f"test/{k.split('@')[0].capitalize()}@{top_k}": v
            for k, v in model_results.items()
        }
        results[model] = formatted_results

    resutls_df = pd.DataFrame(results).T
    return resutls_df


@app.command(help="Train the recommendation model.")
def train_comp(
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ] = DatasetName.MARS,
    epochs: Annotated[
        int, typer.Option("--epochs", "-e", help="Number of epochs")
    ] = config.EPOCHS,
    lr: Annotated[float, typer.Option("--lr", "-l", help="Learning rate")] = config.LR,
    batch_size: Annotated[
        int, typer.Option("--batch_size", "-b", help="Batch size")
    ] = config.BATCH_SIZE,
    patience: Annotated[
        int, typer.Option("--patience", "-p", help="Patience")
    ] = config.PATIENCE,
    val_size: Annotated[
        float, typer.Option("--val_size", "-v", help="Validation size")
    ] = config.VAL_SIZE,
    test_size: Annotated[
        float, typer.Option("--test_size", "-t", help="Test size")
    ] = config.TEST_SIZE,
    top_k: Annotated[
        int, typer.Option("--top_k", "-k", help="Top-k value")
    ] = config.TOP_K,
    monitor: Annotated[
        str, typer.Option("--monitor", "-m", help="Monitor metric")
    ] = config.MONITOR,
    debug: Annotated[bool, typer.Option("--debug", "-D", help="Debug mode")] = False,
):
    dm = ElearningDataModule(
        dataset=dataset,
        batch_size=batch_size,
        test_size=test_size,
        val_size=val_size,
        random_state=config.state["random_state"],
    )
    sota_df = train_sota(
        models=["BPR", "NeuMF", "DeepFM", "WideDeep", "GRU4Rec", "BERT4Rec"],
        dm=dm,
        lr=lr,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        top_k=top_k,
    )

    model_config = EDuRecConfig(
        n_users=dm.num_users,
        n_items=dm.num_items,
        numeric_features=dm.numeric_features,
        cat_cardinalities=dm.cat_cardinalities,
    )
    if config.state["verbose"]:
        print(f"[TRAIN] Dataset {dataset.value} sparsity: {dm.sparsity}")
        print(f"[TRAIN] Min rating: {dm.min_rating}")
        print(f"[TRAIN] Max rating: {dm.max_rating}")
        print(f"[TRAIN] Monitoring: {monitor}")

    edurec_results_list = []
    for model_class in [EDuRecMTL, EDuRecV1]:
        model = model_class(model_config)
        model_name = model.__class__.__name__
        print(f"\n[TRAIN] Starting training for: {model_name}")

        recsys = RecSys(
            model=model,
            top_k=top_k,
            lr=lr,
            monitor=monitor,
            weight_decay=config.WEIGHT_DECAY,
        )

        if not debug:
            torch.compile(recsys)

        early_stop_model = EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode="max",
            min_delta=config.DELTA,
            verbose=True,
        )
        checkpoint_model = ModelCheckpoint(
            monitor=monitor, mode="min", save_top_k=1, filename=f"best_{model_name}"
        )

        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator=config.state["device"],
            devices="auto",
            log_every_n_steps=10,
            callbacks=[early_stop_model, checkpoint_model],
            fast_dev_run=debug,
        )

        trainer.fit(recsys, datamodule=dm)

        if debug:
            continue

        # Evaluación
        test_results = trainer.test(model=recsys, datamodule=dm)
        if test_results:
            # Crear DF de una fila y añadir a la lista
            res_df = pd.DataFrame([test_results[0]])
            res_df.index = [model_name]
            edurec_results_list.append(res_df)

    if not debug and (edurec_results_list or not sota_df.empty):
        all_edurec_df = (
            pd.concat(edurec_results_list) if edurec_results_list else pd.DataFrame()
        )

        final_comparison_df = pd.concat([all_edurec_df, sota_df], axis=0)

        final_comparison_df.index.name = "model"
        final_comparison_df = final_comparison_df.reset_index()

        print("\n" + "=" * 50)
        print("FINAL COMPARISON TABLE")
        print("=" * 50)
        print(final_comparison_df)

        Path(config.RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)
        final_comparison_df.to_csv(
            f"{config.RESULTS_FOLDER}/comparison_{dataset.value}.csv", index=False
        )
