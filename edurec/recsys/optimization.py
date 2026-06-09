from dataclasses import asdict, replace
from tempfile import TemporaryDirectory

import optuna
from lightning.pytorch.callbacks import ModelCheckpoint
from torch_geometric.data import Data

from .. import settings
from ..datasets import ElearningDataModule
from .architecture import EDuRecConfig
from .recsys import RecSys
from .training import train_model


def objective(
    trial: optuna.Trial,
    base_config: EDuRecConfig,
    datamodule: ElearningDataModule,
    inter_graph: Data,
    verbose: bool = False,
) -> float:
    emb_dim = trial.suggest_categorical("emb_dim", [64, 128, 256])

    scorer = trial.suggest_categorical("scorer", ["linear", "single", "funnel"])

    hidden_dims = {
        "linear": [],
        "single": [2 * emb_dim],
        "funnel": [2 * emb_dim, emb_dim],
    }[scorer]

    config = replace(
        base_config,
        emb_dim=emb_dim,
        n_heads=trial.suggest_categorical("n_heads", [1, 2, 4, 8]),
        gnn_layers=trial.suggest_int("gnn_layers", 1, 4),
        n_blocks=trial.suggest_int("n_blocks", 1, 4),
        ff_dim=emb_dim * trial.suggest_categorical("ff_multiplier", [2, 4]),
        hidden_dims=hidden_dims,
        dropout=trial.suggest_float("dropout", 0.0, 0.4),
        edge_dropout=trial.suggest_float("edge_dropout", 0.0, 0.5),
        temperature=trial.suggest_float("temperature", 0.05, 0.5, log=True),
        lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        alpha=trial.suggest_float("alpha", 1e-3, 0.5, log=True),
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
            epochs=settings.EPOCHS,
            patience=settings.PATIENCE,
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
    verbose: bool = False,
) -> optuna.Study:
    assert dm.is_processed, "Data must be processed before optimizing the model."

    inter_graph = dm.build_inter_graph()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.state["random_state"]),
    )

    study.optimize(
        lambda trial: objective(trial, base_config, dm, inter_graph, verbose),
        n_trials=n_trials,
        gc_after_trial=True,
    )

    return study


def get_best_config(study: optuna.Study) -> EDuRecConfig:
    return EDuRecConfig(**study.best_trial.user_attrs["config"])
