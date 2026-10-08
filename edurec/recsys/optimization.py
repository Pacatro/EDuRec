import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import lightning as L
import optuna
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch_geometric.data import HeteroData

from edurec import settings
from edurec.datasets import ElearningDataModule
from edurec.recsys.configs import ModelArch, ModelConfig, TrainConfig
from edurec.recsys.recsys import RecSys
from edurec.recsys.training import train_model

# Bump whenever the search space or the objective changes so old studies are
# not silently resumed with incompatible trials.
OPTIMIZER_VERSION = 9

# Hyperband successive halving: keep one trial per ``reduction_factor`` at each
# resource level, starting from a single epoch.
_HYPERBAND_MIN_RESOURCE = 1
_HYPERBAND_REDUCTION_FACTOR = 3


def _optim_digest(
    base_config: ModelConfig,
    base_train_config: TrainConfig,
    cache_params: Mapping[str, Any] | None,
    *,
    val_topk: int = settings.TOP_K,
    compile: bool = settings.COMPILE_MODEL,
) -> str:
    """Namespace a study by base config, processed data and optimizer version."""
    payload = {
        "optimizer_version": OPTIMIZER_VERSION,
        "model": asdict(base_config),
        "train": asdict(base_train_config),
        "cache_params": cache_params,
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
    """Sample only parameters used by the selected architecture."""
    arch = ModelArch(base_config.arch)
    is_sasrec = arch == ModelArch.SASREC_TEXT
    emb_dim = trial.suggest_categorical(
        "emb_dim", sorted({64, base_config.emb_dim, 256, 512})
    )

    scorer_shape = trial.suggest_categorical(
        "scorer_shape", ["linear", "single", "funnel"]
    )
    hidden_dims = {
        "linear": [],
        "single": [2 * emb_dim],
        "funnel": [2 * emb_dim, emb_dim],
    }[scorer_shape]

    # An ID embedding can activate the profile even without static user features.
    profile_overrides: dict[str, Any] = {}
    if base_config.use_user_features:
        profile_overrides["use_user_id_embedding"] = trial.suggest_categorical(
            "use_user_id_embedding", [False, True]
        )

    # Architecture-design switches (attention pooling, interaction and time
    # features) keep their configured defaults; the ablation command covers them.
    sequence_overrides: dict[str, Any] = {}
    if arch == ModelArch.KG_RNN:
        sequence_overrides.update(
            rnn_type=trial.suggest_categorical("rnn_type", ["gru", "lstm"]),
            rnn_hidden_dim=trial.suggest_categorical(
                "rnn_hidden_dim",
                sorted({64, base_config.rnn_hidden_dim, 128, 256, 512, 1024}),
            ),
            rnn_layers=trial.suggest_categorical(
                "rnn_layers", sorted({1, base_config.rnn_layers, 2, 3})
            ),
        )
    else:
        # Fixed distributions across trials; every width is divisible by every head count.
        heads = sorted({1, 2, 4, 8, base_config.transformer_heads})
        if any(head <= 0 for head in heads):
            raise ValueError("Transformer head counts must be positive.")
        widths = sorted({64, 128, 256, 512, base_config.transformer_hidden_dim})
        widths = [width for width in widths if all(width % head == 0 for head in heads)]
        if not widths:
            raise ValueError(
                "No Transformer width is compatible with the head search space."
            )
        sequence_overrides.update(
            transformer_hidden_dim=trial.suggest_categorical(
                "transformer_hidden_dim", widths
            ),
            transformer_heads=trial.suggest_categorical("transformer_heads", heads),
            transformer_layers=trial.suggest_categorical(
                "transformer_layers",
                sorted({1, base_config.transformer_layers, 2, 3, 4}),
            ),
        )

    graph_overrides: dict[str, Any] = {}
    if not is_sasrec and base_config.graph_mode == "kg":
        graph_overrides["gnn_layers"] = trial.suggest_categorical(
            "gnn_layers", sorted({1, base_config.gnn_layers, 2, 3, 4})
        )
        head_space = [head for head in (1, 2, 4, 8) if emb_dim % head == 0]
        graph_overrides["gnn_heads"] = trial.suggest_categorical(
            "gnn_heads", head_space
        )

    # GCL is part of the KGSeq model. Tune its strength over positive values so
    # the reference configuration always carries the regularizer; the ablation
    # command measures it by disabling ``use_gcl`` instead.
    gcl_overrides: dict[str, Any] = {}
    if arch == ModelArch.KG_RNN and base_config.use_gcl:
        gcl_overrides["gcl_weight"] = trial.suggest_categorical(
            "gcl_weight", [0.05, 0.1, 0.2, 0.5, 1.0]
        )

    config = replace(
        base_config,
        emb_dim=emb_dim,
        **graph_overrides,
        **sequence_overrides,
        # Scorer
        hidden_dims=hidden_dims,
        # User profile initial state
        **profile_overrides,
        # Regularization
        dropout=trial.suggest_categorical(
            "dropout", sorted({0.0, 0.1, base_config.dropout, 0.3, 0.5})
        ),
        **gcl_overrides,
        # Item bias
        use_item_bias=False
        if is_sasrec
        else trial.suggest_categorical("use_item_bias", [True, False]),
    )

    train_config = replace(
        base_train_config,
        # Optimizer
        lr=trial.suggest_categorical(
            "lr", sorted({1e-4, base_train_config.lr, 5e-4, 1e-3})
        ),
        weight_decay=trial.suggest_categorical(
            "weight_decay", sorted({0.0, 1e-5, base_train_config.weight_decay, 1e-3})
        ),
    )

    return config, train_config


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

    model = RecSys(
        cfg=config,
        knowledge_graph=knowledge_graph,
        i_static_feats=datamodule.i_static_feats,
        u_static_feats=datamodule.u_static_feats,
        u_cat_feats=datamodule.u_cat_feats,
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
        )

    if not isinstance(trainer.checkpoint_callback, ModelCheckpoint):
        raise TypeError("Training did not provide a ModelCheckpoint callback.")

    score = trainer.checkpoint_callback.best_model_score

    if score is None:
        raise RuntimeError(f"Metric {model.monitor!r} was not recorded.")

    value = float(score.item())
    if not math.isfinite(value):
        raise RuntimeError(f"Metric {model.monitor!r} is not finite: {value}.")
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
) -> optuna.Study:
    if not dm.is_processed:
        raise ValueError("Data must be processed before optimizing the model.")
    if n_trials < 1 or epochs < 1 or patience < 1 or val_topk < 1:
        raise ValueError("n_trials, epochs, patience and val_topk must be positive.")
    ModelArch(base_config.arch)
    if not base_config.has_history:
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

    knowledge_graph = dm.knowledge_graph
    storage = None
    callbacks = None

    if results_path is not None:
        results_path.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{results_path / 'study.db'}"
        callbacks = [_save_trials_callback(results_path / "trials.csv")]

    digest = _optim_digest(
        base_config,
        base_train_config,
        dm.cache_params,
        val_topk=val_topk,
        compile=compile,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=f"edurec-{dm.dataset_name.value}-{digest}",
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
        ),
        n_trials=n_trials,
        gc_after_trial=True,
        callbacks=callbacks,
    )

    return study
