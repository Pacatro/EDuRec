import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import lightning as L
import optuna
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch_geometric.data import HeteroData

from edurec import settings
from edurec.datasets import ElearningDataModule
from edurec.recsys.configs import ModelArch, ModelConfig, TrainConfig
from edurec.recsys.recsys import RecSys
from edurec.recsys.training import train_model

# Bump whenever the search space or the objective changes so old studies are
# not silently resumed with incompatible trials.
OPTIMIZER_VERSION = 12

# Hyperband successive halving: keep one trial per ``reduction_factor`` at each
# resource level, starting from a single epoch.
_HYPERBAND_MIN_RESOURCE = 1
_HYPERBAND_REDUCTION_FACTOR = 3


def _optim_digest(
    base_config: ModelConfig,
    base_train_config: TrainConfig,
    data_config: Mapping[str, Any],
    *,
    val_topk: int = settings.TOP_K,
    compile: bool = settings.COMPILE_MODEL,
) -> str:
    """Namespace a study by base config, data selection and optimizer version."""
    payload = {
        "optimizer_version": OPTIMIZER_VERSION,
        "model": asdict(base_config),
        "train": asdict(base_train_config),
        "data": data_config,
        "val_topk": val_topk,
        "compile": compile,
        "random_state": settings.state["random_state"],
        "early_stopping_delta": settings.DELTA,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _save_trials_callback(output_path: Path):
    def callback(study: optuna.Study, _: optuna.trial.FrozenTrial) -> None:
        study.trials_dataframe().to_csv(output_path, index=False)

    return callback


class _OptunaPruningCallback(Callback):
    """Report the validation metric each epoch and stop pruned trials early."""

    def __init__(self, trial: optuna.Trial, monitor: str) -> None:
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return

        value = float(metric)
        if not math.isfinite(value):
            return

        step = trainer.current_epoch + 1
        self.trial.report(value, step=step)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned {self.monitor} at epoch {step}.")


def _suggest_configs(
    trial: optuna.Trial,
    base_config: ModelConfig,
    base_train_config: TrainConfig,
) -> tuple[ModelConfig, TrainConfig]:
    """Search eight independent parameters (864 full-KG combinations).

    Couple embedding and recurrent widths, share dropout across modules,
    and leave feature/ablation switches at their configured values.
    """
    if ModelArch(base_config.arch) != ModelArch.KG_RNN:
        raise ValueError("Hyperparameter optimization only supports kg_rnn.")
    emb_dim = trial.suggest_categorical("emb_dim", [128, 256])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.3])
    overrides: dict[str, Any] = {}
    if base_config.graph_mode == "kg":
        overrides.update(
            gnn_layers=trial.suggest_categorical("gnn_layers", [1, 2]),
            gnn_heads=4,
        )
    if base_config.use_gcl:
        overrides["gcl_weight"] = trial.suggest_categorical(
            "gcl_weight", [0.05, 0.1, 0.2]
        )
    config = replace(
        base_config,
        emb_dim=emb_dim,
        rnn_hidden_dim=emb_dim,
        rnn_type=trial.suggest_categorical("rnn_type", ["gru", "lstm"]),
        rnn_layers=trial.suggest_categorical("rnn_layers", [1, 2]),
        hidden_dims=[2 * emb_dim],
        use_user_id_embedding=False,
        use_item_bias=True,
        dropout=dropout,
        gnn_dropout=dropout,
        rnn_dropout=dropout,
        **overrides,
    )
    train_config = replace(
        base_train_config,
        lr=trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3]),
        weight_decay=trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3]),
    )
    return config, train_config


@contextmanager
def _trial_logger(
    trial: optuna.Trial, dataset_name: str, enabled: bool
) -> Iterator[MLFlowLogger | None]:
    """Give each trial its own run, including pruned and failed trials."""
    if not enabled:
        yield None
        return

    logger = MLFlowLogger(
        experiment_name="HPO",
        run_name=f"{dataset_name}-trial-{trial.number}",
        tracking_uri=settings.MLFLOW_TRACKING_URI,
        tags={
            "dataset": dataset_name,
            "study": trial.study.study_name,
            "trial_number": str(trial.number),
            "random_state": str(settings.state["random_state"]),
        },
        log_model=False,
    )
    trial.set_user_attr("mlflow_run_id", logger.run_id)
    try:
        logger.log_hyperparams(
            {f"hpo/{key}": value for key, value in trial.params.items()}
        )
        yield logger
    except optuna.TrialPruned:
        logger.experiment.set_tag(logger.run_id, "trial_state", "PRUNED")
        logger.experiment.set_terminated(logger.run_id, status="KILLED")
        raise
    except BaseException:
        logger.experiment.set_tag(logger.run_id, "trial_state", "FAIL")
        logger.finalize("failed")
        raise
    else:
        logger.experiment.set_tag(logger.run_id, "trial_state", "COMPLETE")
        logger.finalize("success")


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
    limit_val_batches: float | None = None,
    log_mlflow: bool = False,
) -> float:
    base_train_config = replace(
        base_train_config,
        epochs=epochs,
        patience=patience,
        batch_size=datamodule.batch_size,
    )
    config, train_config = _suggest_configs(trial, base_config, base_train_config)
    # Model initialization must not depend on earlier trials or resumed runs.
    settings.seed_everything(settings.state["random_state"])
    trial.set_user_attr("config", asdict(config))
    trial.set_user_attr("train_config", asdict(train_config))
    trial.set_user_attr("val_topk", val_topk)
    trial.set_user_attr("random_state", settings.state["random_state"])

    with _trial_logger(trial, datamodule.config.dataset.value, log_mlflow) as logger:
        model = RecSys(
            cfg=config,
            knowledge_graph=knowledge_graph,
            i_static_feats=datamodule.data.i_static_feats,
            u_static_feats=datamodule.data.u_static_feats,
            u_cat_feats=datamodule.data.u_cat_feats,
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
                callbacks=[_OptunaPruningCallback(trial, model.monitor)],
                default_root_dir=root_dir,
                limit_val_batches=limit_val_batches,
                logger=logger,
            )

        if not isinstance(trainer.checkpoint_callback, ModelCheckpoint):
            raise TypeError("Training did not provide a ModelCheckpoint callback.")

        score = trainer.checkpoint_callback.best_model_score

        if score is None:
            raise RuntimeError(f"Metric {model.monitor!r} was not recorded.")

        value = float(score.item())
        if not math.isfinite(value):
            raise RuntimeError(f"Metric {model.monitor!r} is not finite: {value}.")
        if logger is not None:
            logger.log_metrics({"best_ndcg": value})
        return value


def optimize_model(
    base_config: ModelConfig,
    base_train_config: TrainConfig,
    dm: ElearningDataModule,
    n_trials: int,
    epochs: int,
    patience: int,
    search_epochs: int | None = None,
    val_topk: int = settings.TOP_K,
    verbose: bool = False,
    results_path: Path | None = None,
    compile: bool = settings.COMPILE_MODEL,
    limit_val_batches: float | None = settings.OPTIM_LIMIT_VAL_BATCHES,
    log_mlflow: bool = False,
) -> optuna.Study:
    if not dm.is_processed:
        raise ValueError("Data must be processed before optimizing the model.")
    if n_trials < 1 or epochs < 1 or patience < 1 or val_topk < 1:
        raise ValueError("n_trials, epochs, patience and val_topk must be positive.")
    if ModelArch(base_config.arch) != ModelArch.KG_RNN:
        raise ValueError("Hyperparameter optimization only supports kg_rnn.")
    if not base_config.has_history and base_config.arch == ModelArch.SASREC_TEXT:
        raise ValueError("Optimization requires chronological history.")
    # Trials train with a reduced budget; the winning configuration is saved
    # with the full ``epochs`` for the final run.
    trial_epochs = epochs if search_epochs is None else min(search_epochs, epochs)
    if trial_epochs < 1:
        raise ValueError("search_epochs must be positive.")

    base_train_config = replace(
        base_train_config,
        epochs=trial_epochs,
        patience=patience,
        batch_size=dm.batch_size,
    )

    knowledge_graph = dm.adapter.knowledge_graph
    storage = None
    callbacks = None

    if results_path is not None:
        results_path.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{results_path / 'study.db'}"
        callbacks = [_save_trials_callback(results_path / "trials.csv")]

    data_config = dm.config
    community = data_config.community
    digest = _optim_digest(
        base_config,
        base_train_config,
        {
            "dataset": dm.data_variant,
            "min_interactions": data_config.min_interactions,
            "test_ratio": data_config.test_ratio,
            "val_ratio": data_config.val_ratio,
            "max_users": community.max_users,
            "max_items": community.max_items,
            "max_interactions": community.max_interactions,
            "community_fields": community.fields,
            "selection_seed": community.seed,
            "min_history_retention": community.min_history_retention,
        },
        val_topk=val_topk,
        compile=compile,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=f"edurec-{dm.config.dataset.value}-{digest}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=settings.state["random_state"],
            n_startup_trials=10,
            multivariate=True,
            group=True,
        ),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=_HYPERBAND_MIN_RESOURCE,
            max_resource=trial_epochs,
            reduction_factor=_HYPERBAND_REDUCTION_FACTOR,
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
            trial_epochs,
            patience,
            val_topk=val_topk,
            verbose=verbose,
            compile=compile,
            limit_val_batches=limit_val_batches,
            log_mlflow=log_mlflow,
        ),
        n_trials=n_trials,
        gc_after_trial=True,
        callbacks=callbacks,
    )

    return study
