import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import optuna
from lightning.pytorch.callbacks import ModelCheckpoint
from torch_geometric.data import HeteroData

from .. import settings
from ..datasets import ElearningDataModule
from .archs.kg_rnn import KGRNN
from .configs import ModelConfig, TrainConfig
from .recsys import RecSys
from .training import train_model

# Bump whenever the search space or the objective changes so old studies are
# not silently resumed with incompatible trials.
OPTIMIZER_VERSION = 1


def _optim_digest(
    base_config: ModelConfig,
    base_train_config: TrainConfig,
    cache_params: Mapping[str, Any] | None,
) -> str:
    """Namespace a study by base config, processed data and optimizer version."""
    payload = {
        "optimizer_version": OPTIMIZER_VERSION,
        "model": asdict(base_config),
        "train": asdict(base_train_config),
        "cache_params": cache_params,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _save_trials_callback(output_path: Path):
    def callback(study: optuna.Study, _: optuna.trial.FrozenTrial) -> None:
        study.trials_dataframe().to_csv(output_path, index=False)

    return callback


def objective(
    trial: optuna.Trial,
    base_config: ModelConfig,
    base_train_config: TrainConfig,
    datamodule: ElearningDataModule,
    knowledge_graph: HeteroData,
    epochs: int,
    patience: int,
    val_topk: int = settings.TOP_K,
    verbose: bool = False,
    compile: bool = settings.COMPILE_MODEL,
) -> float:
    emb_dim = trial.suggest_categorical(
        "emb_dim", sorted({64, settings.EMB_DIM, 256, 512})
    )
    scorer = trial.suggest_categorical("scorer", ["linear", "single", "funnel"])

    hidden_dims = {
        "linear": [],
        "single": [2 * emb_dim],
        "funnel": [2 * emb_dim, emb_dim],
    }[scorer]

    config = replace(
        base_config,
        emb_dim=emb_dim,
        # Knowledge-graph encoder
        gnn_layers=trial.suggest_categorical(
            "gnn_layers", sorted({1, settings.GNN_LAYERS, 3, 4})
        ),
        # Recurrent sequence encoder
        seq_cell=trial.suggest_categorical("seq_cell", ["gru", "lstm"]),
        gru_hidden_dim=trial.suggest_categorical(
            "gru_hidden_dim",
            sorted({64, settings.GRU_HIDDEN_DIM, 256, 2 * emb_dim}),
        ),
        gru_layers=trial.suggest_categorical(
            "gru_layers", sorted({1, settings.GRU_LAYERS, 2, 3})
        ),
        # Scorer
        hidden_dims=hidden_dims,
        # Regularization
        dropout=trial.suggest_categorical(
            "dropout", sorted({0.0, 0.1, settings.DROPOUT, 0.3, 0.5})
        ),
        edge_dropout=trial.suggest_categorical(
            "edge_dropout", sorted({0.0, 0.1, settings.DROP_EDGES_P, 0.3, 0.5})
        ),
        # GCL loss
        temperature=trial.suggest_categorical(
            "temperature", sorted({0.05, 0.1, settings.TAU, 0.2, 0.5})
        ),
        # Item bias
        use_item_bias=trial.suggest_categorical("use_item_bias", [True, False]),
    )

    train_config = replace(
        base_train_config,
        # GCL loss
        alpha=trial.suggest_categorical(
            "alpha", sorted({0.01, settings.LOSS_ALPHA, 0.1, 0.2, 1.0})
        ),
        # Optimizer
        lr=trial.suggest_categorical("lr", sorted({1e-4, settings.LR, 5e-4, 1e-3})),
        weight_decay=trial.suggest_categorical(
            "weight_decay", sorted({0.0, 1e-5, settings.WEIGHT_DECAY, 1e-3})
        ),
    )

    trial.set_user_attr("config", asdict(config))
    trial.set_user_attr("train_config", asdict(train_config))

    model = RecSys(
        cfg=config,
        model=KGRNN(config),
        knowledge_graph=knowledge_graph,
        u_static_feats=datamodule.u_static_feats,
        i_static_feats=datamodule.i_static_feats,
        train_cfg=train_config,
        val_topk=val_topk,
    )

    with TemporaryDirectory(prefix=f"edurec-optuna-{trial.number}-") as root_dir:
        trainer, _, _ = train_model(
            model=model,
            dm=datamodule,
            debug=False,
            epochs=epochs,
            patience=patience,
            monitor=model.monitor,
            compile=compile,
            verbose=verbose,
            default_root_dir=root_dir,
        )

    assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)

    score = trainer.checkpoint_callback.best_model_score

    if score is None:
        raise RuntimeError(f"Metric {model.monitor!r} was not recorded.")

    return score.item()


def optimize_model(
    base_config: ModelConfig,
    base_train_config: TrainConfig,
    dm: ElearningDataModule,
    n_trials: int,
    epochs: int,
    patience: int,
    val_topk: int = settings.TOP_K,
    verbose: bool = False,
    results_path: Path | None = None,
    compile: bool = settings.COMPILE_MODEL,
) -> optuna.Study:
    assert dm.is_processed, "Data must be processed before optimizing the model."

    knowledge_graph = dm.build_knowledge_graph()
    storage = None
    callbacks = None

    if results_path is not None:
        results_path.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{results_path / 'study.db'}"
        callbacks = [_save_trials_callback(results_path / "trials.csv")]

    digest = _optim_digest(base_config, base_train_config, dm.cache_params)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"edurec-{dm.dataset_name.value}-{digest[:10]}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=settings.state["random_state"],
            n_startup_trials=min(10, n_trials),
            multivariate=True,
        ),
    )
    study.set_user_attr("search_space_hash", digest)

    study.optimize(
        lambda trial: objective(
            trial,
            base_config,
            base_train_config,
            dm,
            knowledge_graph,
            epochs,
            patience,
            val_topk=val_topk,
            verbose=verbose,
            compile=compile,
        ),
        n_trials=n_trials,
        gc_after_trial=True,
        callbacks=callbacks,
    )

    return study
