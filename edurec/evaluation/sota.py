from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, cast

import pandas as pd
import torch
import torch.distributed as dist
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.model.abstract_recommender import AbstractRecommender
from recbole.utils import (
    InputType,
    ModelType,
    get_model,
    get_trainer,
    init_logger,
    init_seed,
)
from torch import nn

from .. import settings
from ..datasets import ElearningDataModule, RecSysDataset, RecSysQuery
from ..recsys.ranking import build_ranking_metrics, update_ranking_metrics

BENCHMARK_SPLITS = ("train", "valid", "test")
SEQUENTIAL_BENCHMARK_SPLITS = ("train_seq", "valid_seq", "test_seq")


@contextmanager
def _recbole_checkpoints() -> Generator[None]:
    """Allow torch.load to unpickle RecBole checkpoints during the run.

    RecBole 1.2.0 saves checkpoints containing custom pickled objects and
    loads them without weights_only, which fails with PyTorch >= 2.6. The
    checkpoints are produced by this same run, so they are trusted.
    """
    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = trusted_load
    try:
        yield
    finally:
        torch.load = original_load


def eval_sota_models(
    models: list[str],
    dm: ElearningDataModule,
    cfg_path: Path | None = None,
    epochs: int = settings.EPOCHS,
    lr: float = settings.LR,
    batch_size: int = settings.BATCH_SIZE,
    patience: int = settings.PATIENCE,
    topks: list[int] | None = None,
    adaptive_k: bool = settings.ADAPTIVE_K,
    results_path: Path | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    dataset_name = dm.data_variant
    atomic_dataset_dir = dm.atomic_folder
    dm.setup(stage="test")

    if dm.has_temporal_order and any(_is_sequential_model(model) for model in models):
        _save_sequential_benchmark_files(dm)

    base_config = _build_config_dict(
        data_root=atomic_dataset_dir.parent,
        atomic_dataset_dir=atomic_dataset_dir,
        dataset_name=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        topks=topks or settings.TOP_KS,
        show_progress=show_progress,
    )

    results: list[dict[str, Any]] = []

    for model in models:
        if not dm.has_temporal_order and _is_sequential_model(model):
            print(
                f"[EVAL] Skipping {model}: it is a sequential model and "
                f"{dataset_name} has no temporal order."
            )
            continue

        print(f"[EVAL] Evaluating {model}...")

        metrics, training_time, inference_time = _run_model(
            model=model,
            dataset_name=dataset_name,
            cfg_path=cfg_path,
            config_dict=_config_for_model(model, base_config),
            dm=dm,
            adaptive_k=adaptive_k,
        )

        result = {
            "model": model,
            "seed": settings.state["random_state"],
            **metrics,
            "training_time_s": training_time,
            "inference_time_s": inference_time,
        }

        if results_path is not None:
            _save_model_result(result, model, results_path)

        results.append(result)

    return pd.DataFrame(results)


def _save_model_result(result: dict[str, Any], model: str, results_path: Path) -> None:
    model_root = results_path / model / f"seed_{result['seed']}"
    model_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(
        model_root / settings.METRICS_FILENAME,
        index=False,
    )


def _run_model(
    model: str,
    dataset_name: str,
    cfg_path: Path | None,
    config_dict: dict[str, object],
    dm: ElearningDataModule,
    adaptive_k: bool,
) -> tuple[dict[str, Any], float, float]:
    """Run RecBole's training and evaluation stages.

    Args:
        model: Name of the RecBole model.
        dataset_name: Name of the atomic dataset.
        cfg_path: Optional path to an additional RecBole configuration file.
        config_dict: Programmatic RecBole configuration.

    Returns:
        Test metrics, training time and inference time in seconds.
    """
    with TemporaryDirectory(
        prefix=f"edurec-recbole-{model.lower()}-"
    ) as checkpoint_dir:
        run_config = {**config_dict, "checkpoint_dir": checkpoint_dir}
        return _fit_and_evaluate_model(
            model,
            dataset_name,
            cfg_path,
            run_config,
            dm,
            adaptive_k,
        )


def _fit_and_evaluate_model(
    model: str,
    dataset_name: str,
    cfg_path: Path | None,
    config_dict: dict[str, object],
    dm: ElearningDataModule,
    adaptive_k: bool,
) -> tuple[dict[str, Any], float, float]:
    config = Config(
        model=model,
        dataset=dataset_name,
        config_file_list=[str(cfg_path)] if cfg_path is not None else None,
        config_dict=config_dict,
    )

    # Config.__getitem__ can theoretically return None, so static type
    # checkers cannot infer the actual types initialized by RecBole.
    seed = cast(int, config["seed"])
    local_rank = cast(int, config["local_rank"])
    reproducibility = cast(bool, config["reproducibility"])
    device = cast(torch.device, config["device"])
    model_name = cast(str, config["model"])
    model_type = cast(ModelType, config["MODEL_TYPE"])
    show_progress = cast(bool, config["show_progress"])
    single_spec = cast(bool, config["single_spec"])

    init_seed(seed, reproducibility)
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)

    init_seed(seed + local_rank, reproducibility)

    model_class = get_model(model_name)
    recbole_model = model_class(config, train_data._dataset).to(device)

    trainer_class = get_trainer(model_type, model_name)
    trainer = trainer_class(config, recbole_model)

    _synchronize_device(device)
    training_started_at = perf_counter()

    with _recbole_checkpoints():
        trainer.fit(
            train_data,
            valid_data,
            saved=True,
            show_progress=show_progress,
        )

        _synchronize_device(device)
        training_time = perf_counter() - training_started_at

        _synchronize_device(device)
        inference_started_at = perf_counter()

        _load_best_model(recbole_model, trainer.saved_model_file, device)

        test_result = _evaluate_common_queries(
            model=recbole_model,
            trainer=trainer,
            dataset=train_data._dataset,
            dm=dm,
            config=config,
            adaptive_k=adaptive_k,
        )

    _synchronize_device(device)
    inference_time = perf_counter() - inference_started_at

    if not single_spec and dist.is_initialized():
        dist.destroy_process_group()

    return test_result, training_time, inference_time


def _load_best_model(
    model: Any,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    model.eval()


@torch.no_grad()
def _evaluate_common_queries(
    model: nn.Module,
    trainer: Any,
    dataset: Any,
    dm: ElearningDataModule,
    config: Config,
    adaptive_k: bool,
) -> dict[str, float]:
    """Evaluate RecBole scores with EDuRec's one-target-per-query protocol."""
    device = cast(torch.device, config["device"])
    topks = list(cast(list[int], config["topk"]))
    metrics = build_ranking_metrics(topks, adaptive_k=adaptive_k).to(device)

    user_lookup = _token_lookup(
        dataset,
        field=settings.USER_COL,
        size=dm.num_users,
        device=device,
    )
    item_lookup = _token_lookup(
        dataset,
        field=settings.ITEM_COL,
        size=dm.num_items,
        device=device,
    )
    is_sequential = cast(ModelType, config["MODEL_TYPE"]) == ModelType.SEQUENTIAL

    for batch in dm.test_dataloader():
        interaction = _query_interaction(
            batch=batch,
            dataset=dataset,
            user_lookup=user_lookup,
            item_lookup=item_lookup,
            is_sequential=is_sequential,
            device=device,
        )
        recbole_scores = _full_sort_scores(
            model=model,  # type: ignore
            trainer=trainer,
            dataset=dataset,
            interaction=interaction,
            device=device,
        )
        scores = recbole_scores.index_select(1, item_lookup)

        update_ranking_metrics(
            metrics=metrics,
            scores=scores,
            target_item_ids=batch.target_item_id.to(device),
            query_ids=batch.query_id.to(device),
            history_items=batch.history_items.to(device),
            history_mask=batch.history_valid_mask.to(device),
            max_k=max(topks),
        )

    return {
        name: float(value.detach().cpu().item())
        for name, value in metrics.compute().items()
    }


def _token_lookup(
    dataset: Any,
    field: str,
    size: int,
    device: torch.device,
) -> torch.Tensor:
    """Map EDuRec's contiguous IDs to RecBole's internal token IDs."""
    tokens = [str(idx) for idx in range(size)]
    try:
        internal_ids = dataset.token2id(field, tokens)
    except ValueError as error:
        raise RuntimeError(
            f"RecBole dataset does not contain the full {field} catalog."
        ) from error
    return torch.as_tensor(internal_ids, dtype=torch.long, device=device)


def _query_interaction(
    batch: RecSysQuery,
    dataset: Any,
    user_lookup: torch.Tensor,
    item_lookup: torch.Tensor,
    is_sequential: bool,
    device: torch.device,
) -> Interaction:
    user_ids = user_lookup[batch.user_id.to(device).long()]
    values: dict[str, torch.Tensor] = {settings.USER_COL: user_ids}

    if is_sequential:
        history_mask = batch.history_valid_mask.to(device).bool()
        external_history = batch.history_items.to(device).long() - 1
        internal_history = torch.zeros_like(external_history)
        internal_history[history_mask] = item_lookup[external_history[history_mask]]
        values[f"{settings.ITEM_COL}_list"] = internal_history
        values["item_length"] = history_mask.sum(dim=1).long()

    return dataset.join(Interaction(values))


def _full_sort_scores(
    model: AbstractRecommender,
    trainer: Any,
    dataset: Any,
    interaction: Interaction,
    device: torch.device,
) -> torch.Tensor:
    """Get full-catalog scores, including models without full_sort_predict."""
    interaction = interaction.to(device)
    assert hasattr(model, "full_sort_predict") or hasattr(model, "predict")
    assert not isinstance(model, torch.Tensor)

    try:
        scores = model.full_sort_predict(interaction)
    except NotImplementedError:
        item_count = dataset.item_num
        repeated = interaction.repeat_interleave(item_count)
        repeated.update(dataset.get_item_feature().to(device).repeat(len(interaction)))
        if len(repeated) <= trainer.test_batch_size:
            scores = model.predict(repeated)
        else:
            scores = trainer._spilt_predict(repeated, len(repeated))

    return scores.view(len(interaction), dataset.item_num)


def _save_sequential_benchmark_files(dm: ElearningDataModule) -> None:
    """Export split-preserving sequence rows for RecBole sequential models."""
    split_datasets = {
        "train_seq": dm.train_ds,
        "valid_seq": dm.val_ds,
        "test_seq": dm.test_ds,
    }
    for split_name, split_dataset in split_datasets.items():
        frame = _sequential_benchmark_frame(split_name, split_dataset)
        path = dm.atomic_folder / f"{dm.data_variant}.{split_name}.inter"
        frame.to_csv(path, sep="\t", index=False)


def _sequential_benchmark_frame(
    split_name: str,
    split_dataset: RecSysDataset,
) -> pd.DataFrame:
    history_mask = split_dataset.history_valid_mask.bool()
    user_ids = split_dataset.user_ids
    target_item_ids = split_dataset.target_item_ids
    history_items = split_dataset.history_items

    if split_name == "train_seq":
        keep = history_mask.any(dim=1)
        keep_array = keep.cpu().numpy()
        user_ids = user_ids[keep_array]
        target_item_ids = target_item_ids[keep_array]
        history_items = history_items[keep]
        history_mask = history_mask[keep]

    if len(user_ids) == 0:
        raise RuntimeError(
            f"Sequential benchmark split {split_name!r} has no usable queries."
        )

    histories = [
        " ".join(str(int(item_id) - 1) for item_id in items[mask].tolist())
        for items, mask in zip(history_items, history_mask, strict=True)
    ]
    return pd.DataFrame(
        {
            f"{settings.USER_COL}:token": [str(int(value)) for value in user_ids],
            f"{settings.ITEM_COL}:token": [
                str(int(value)) for value in target_item_ids
            ],
            f"{settings.ITEM_COL}_list:token_seq": histories,
        }
    )


def _synchronize_device(device: torch.device) -> None:
    """Wait until all pending device operations have completed."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _is_sequential_model(model: str) -> bool:
    """Whether the RecBole model requires a time-ordered interaction list."""
    model_class = get_model(model)
    return getattr(model_class, "type", None) == ModelType.SEQUENTIAL


def _config_for_model(
    model: str,
    base_config: dict[str, object],
) -> dict[str, object]:
    """Adapt the base configuration to a specific RecBole model."""
    config = dict(base_config)

    model_class = get_model(model)
    model_type = getattr(model_class, "type", None)
    input_type = getattr(model_class, "input_type", None)

    if model_type == ModelType.SEQUENTIAL:
        item_list_field = f"{settings.ITEM_COL}_list"
        config.update(
            {
                "benchmark_filename": list(SEQUENTIAL_BENCHMARK_SPLITS),
                "load_col": {
                    "inter": [settings.USER_COL, settings.ITEM_COL, item_list_field],
                    "user": [settings.USER_COL],
                    "item": [settings.ITEM_COL],
                },
                "ITEM_LIST_LENGTH_FIELD": "item_length",
                "LIST_SUFFIX": "_list",
                "MAX_ITEM_LIST_LENGTH": settings.MAX_HISTORY_LEN,
                "alias_of_item_id": [item_list_field],
                "train_neg_sample_args": None,
            }
        )

        return config

    config["train_neg_sample_args"] = _negative_sampling_config(input_type)
    return config


def _negative_sampling_config(input_type: InputType | None) -> dict[str, object] | None:
    if input_type not in {InputType.PAIRWISE, InputType.POINTWISE}:
        return None
    return {
        "distribution": "uniform",
        "sample_num": 1,
        "alpha": 1.0,
        "dynamic": False,
        "candidate_num": 0,
    }


def _build_config_dict(
    data_root: Path,
    atomic_dataset_dir: Path,
    dataset_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    topks: list[int],
    show_progress: bool = False,
) -> dict[str, object]:
    """Build the common RecBole configuration."""
    load_col: dict[str, list[str]] = {
        "inter": [
            settings.USER_COL,
            settings.ITEM_COL,
        ],
        "user": _field_names(atomic_dataset_dir / f"{dataset_name}.user"),
        "item": _field_names(atomic_dataset_dir / f"{dataset_name}.item"),
    }

    return {
        "data_path": str(data_root),
        "benchmark_filename": list(BENCHMARK_SPLITS),
        "USER_ID_FIELD": settings.USER_COL,
        "ITEM_ID_FIELD": settings.ITEM_COL,
        "load_col": load_col,
        "seed": settings.state["random_state"],
        "reproducibility": True,
        "gpu_id": 0,
        "use_gpu": settings.state["device"] != "cpu",
        "epochs": epochs,
        "train_batch_size": batch_size,
        "eval_batch_size": batch_size,
        "learner": "adam",
        "learning_rate": lr,
        "stopping_step": patience,
        "eval_step": 1,
        "save_dataset": False,
        "save_dataloaders": False,
        "checkpoint_dir": "saved/recbole",
        "eval_args": {
            "group_by": "user",
            "order": "TO",
            "mode": "full",
        },
        "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision", "MAP"],
        "topk": topks,
        "valid_metric": f"NDCG@{max(topks)}",
        "metric_decimal_place": 4,
        "embedding_size": settings.EMB_DIM,
        "show_progress": show_progress,
        "state": "info" if show_progress else "error",
        "log_wandb": False,
    }


def _field_names(path: Path) -> list[str]:
    """Read field names from a RecBole atomic file."""
    columns = pd.read_csv(path, sep="\t", nrows=0).columns
    return [column.split(":", maxsplit=1)[0] for column in columns]
