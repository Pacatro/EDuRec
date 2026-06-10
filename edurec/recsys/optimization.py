from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import optuna
from lightning.pytorch.callbacks import ModelCheckpoint
from torch_geometric.data import Data

from .. import settings
from ..datasets import ElearningDataModule
from .architecture import EDuRecConfig
from .recsys import RecSys
from .training import train_model


def _save_trials_callback(output_path: Path):
    def callback(study: optuna.Study, _: optuna.trial.FrozenTrial) -> None:
        study.trials_dataframe().to_csv(output_path, index=False)

    return callback


def objective(
    trial: optuna.Trial,
    base_config: EDuRecConfig,
    datamodule: ElearningDataModule,
    inter_graph: Data,
    epochs: int,
    patience: int,
    verbose: bool = False,
) -> float:
    emb_dim = trial.suggest_categorical(
        "emb_dim",
        sorted({64, settings.EMB_DIM, 256}),
    )

    scorer = trial.suggest_categorical(
        "scorer",
        ["linear", "single", "funnel"],
    )

    hidden_dims = {
        "linear": [],
        "single": [2 * emb_dim],
        "funnel": [2 * emb_dim, emb_dim],
    }[scorer]

    default_ff_multiplier = settings.FF_DIM // settings.EMB_DIM

    config = replace(
        base_config,
        # Representaciones
        emb_dim=emb_dim,
        # Encoder del grafo
        gnn_layers=trial.suggest_categorical(
            "gnn_layers", sorted({1, settings.GNN_LAYERS, 3})
        ),
        # Encoder secuencial
        n_heads=trial.suggest_categorical(
            "n_heads", sorted({2, settings.NUM_HEADS, 8})
        ),
        n_blocks=trial.suggest_categorical(
            "n_blocks", sorted({1, settings.NUM_BLOCKS, 3})
        ),
        ff_dim=emb_dim
        * trial.suggest_categorical(
            "ff_multiplier", sorted({2, default_ff_multiplier, 8})
        ),
        # Scorer
        hidden_dims=hidden_dims,
        # Regularization
        dropout=trial.suggest_categorical(
            "dropout", sorted({0.0, 0.1, settings.DROPOUT, 0.3})
        ),
        edge_dropout=trial.suggest_categorical(
            "edge_dropout", sorted({0.0, 0.1, settings.DROP_EDGES_P, 0.3})
        ),
        # GCL Loss
        temperature=trial.suggest_categorical(
            "temperature", sorted({0.05, 0.1, settings.TAU, 0.2})
        ),
        alpha=trial.suggest_categorical(
            "alpha", sorted({0.01, settings.LOSS_ALPHA, 0.1, 0.2})
        ),
        # Optimización
        lr=trial.suggest_categorical("lr", sorted({1e-4, settings.LR, 5e-4})),
        weight_decay=trial.suggest_categorical(
            "weight_decay", sorted({0.0, 1e-5, settings.WEIGHT_DECAY})
        ),
        # Predicción
        use_item_bias=trial.suggest_categorical("use_item_bias", [True, False]),
    )

    trial.set_user_attr("config", asdict(config))

    model = RecSys(
        cfg=config,
        inter_graph=inter_graph,
        u_static_feats=datamodule.u_static_feats,
        i_static_feats=datamodule.i_static_feats,
        val_topk=settings.TOP_K,
    )

    with TemporaryDirectory(prefix=f"edurec-optuna-{trial.number}-") as root_dir:
        trainer, _ = train_model(
            model=model,
            dm=datamodule,
            debug=False,
            use_logger=False,
            epochs=epochs,
            patience=patience,
            monitor=model.monitor,
            compile=False,
            verbose=verbose,
            default_root_dir=root_dir,
        )

    assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)

    score = trainer.checkpoint_callback.best_model_score

    if score is None:
        raise RuntimeError(f"No se ha registrado la métrica {model.monitor!r}.")

    return score.item()


def optimize_model(
    base_config: EDuRecConfig,
    dm: ElearningDataModule,
    n_trials: int,
    epochs: int,
    patience: int,
    verbose: bool = False,
    results_path: Path | None = None,
) -> optuna.Study:
    assert dm.is_processed, "Data must be processed before optimizing the model."

    inter_graph = dm.build_inter_graph()
    storage = None
    callbacks = None

    if results_path is not None:
        results_path.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{results_path / 'study.db'}"
        callbacks = [_save_trials_callback(results_path / "trials.csv")]

    study = optuna.create_study(
        direction="maximize",
        study_name=f"edurec-{dm.dataset_name.value}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=settings.state["random_state"],
            n_startup_trials=n_trials,
            multivariate=True,
        ),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            base_config,
            dm,
            inter_graph,
            epochs,
            patience,
            verbose,
        ),
        n_trials=n_trials,
        gc_after_trial=True,
        callbacks=callbacks,
    )

    return study
