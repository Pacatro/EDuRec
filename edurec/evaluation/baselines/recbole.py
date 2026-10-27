"""RecBole-backed baselines evaluated with the shared retrieval protocol."""

import os
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Self, cast

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

from edurec import settings
from edurec.datasets import ElearningDataModule
from edurec.datasets.adapter import DataAdapter
from edurec.datasets.cache import ProcessedData, replace_folder
from edurec.datasets.queries import QueryConfig
from edurec.datasets.torch_data import RecSysDataset, RecSysQuery
from edurec.recsys.ranking import build_ranking_metrics, update_ranking_metrics

#: Benchmark split names read by RecBole for the general interaction protocol.
BENCHMARK_SPLITS = ("train", "valid", "test")

#: Benchmark split names read by RecBole for prebuilt sequential histories.
SEQUENTIAL_BENCHMARK_SPLITS = ("train_seq", "valid_seq", "test_seq")

#: Exported sequential file -> processed split that supplies its rows.
SEQUENTIAL_SPLIT_SOURCES = {
    "train_seq": "train",
    "valid_seq": "val",
    "test_seq": "test",
}


def _is_sequential_model(model: str) -> bool:
    """Whether the RecBole model requires a time-ordered interaction list."""
    model_class = get_model(model)
    return getattr(model_class, "type", None) == ModelType.SEQUENTIAL


class RecBoleAdapter(DataAdapter[RecSysQuery, Interaction]):
    """Default general/sequential inputs; subclass for baseline-specific fields.

    ``bind_catalog`` must be called after RecBole prepares its internal IDs.
    A separate adapter instance is used for each evaluated model.
    """

    def __init__(
        self,
        data: ProcessedData,
        config: QueryConfig | None = None,
        sequential: bool = False,
    ):
        self.requires_temporal_order = sequential
        super().__init__(data, config)
        self.sequential = sequential
        self.dataset_name = self.data.dataset_name
        self._catalog: tuple[Any, torch.Tensor, torch.Tensor, torch.device] | None = (
            None
        )

    @classmethod
    def for_model(
        cls,
        model: str,
        data: ProcessedData,
        config: QueryConfig | None = None,
    ) -> Self:
        """Build the default adapter for a RecBole model and input protocol."""
        return cls(data, config, sequential=_is_sequential_model(model))

    def dataset(self, split: str) -> RecSysDataset:
        return self.evaluation_queries(split)

    def export(
        self, folder: Path, *, dataset_name: str | None = None
    ) -> dict[str, Path]:
        self.dataset_name = dataset_name or self.data.dataset_name
        paths = replace_folder(
            folder,
            self.export_files,
        )
        return {name: folder / path.name for name, path in paths.items()}

    def export_files(
        self, output_dir: Path, *, dataset_name: str | None = None
    ) -> dict[str, Path]:
        """Build the model's atomic files; override to add baseline-specific files."""
        name = dataset_name or self.dataset_name
        if not name:
            raise ValueError(
                "dataset_name is required when the processed data has no manifest."
            )
        files = _export_atomic_files(self.data, name, output_dir)
        if self.sequential:
            queries = {
                export_name: self.get_queries(split)
                for export_name, split in SEQUENTIAL_SPLIT_SOURCES.items()
            }
            files.update(_export_sequential_benchmark(queries, name, output_dir))
        return files

    def configure(self, model: str, base: dict[str, object]) -> dict[str, object]:
        """Override in a model adapter to load extra fields or atomic files."""
        return _config_for_model(model, base, self.config.max_history, self.sequential)

    def bind_catalog(self, dataset: Any, device: torch.device) -> None:
        user_lookup = _token_lookup(
            dataset, settings.USER_COL, self.data.num_users, device
        )
        item_lookup = _token_lookup(
            dataset, settings.ITEM_COL, self.data.num_items, device
        )
        self._catalog = dataset, user_lookup, item_lookup, device

    def evaluation_input(self, query: RecSysQuery) -> Interaction:
        if self._catalog is None:
            raise RuntimeError("Call bind_catalog() before converting RecBole queries.")
        dataset, users, items, device = self._catalog
        return _query_interaction(query, dataset, users, items, self.sequential, device)

    def canonical_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Drop RecBole padding and restore canonical item-ID column order."""
        if self._catalog is None:
            raise RuntimeError("Call bind_catalog() before converting RecBole scores.")
        return scores.index_select(1, self._catalog[2])


def _token_lookup(
    dataset: Any,
    field: str,
    size: int,
    device: torch.device,
) -> torch.Tensor:
    """Map KGSeq's contiguous IDs to RecBole's internal token IDs."""
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


def _config_for_model(
    model: str,
    base_config: dict[str, object],
    max_history: int,
    sequential: bool,
) -> dict[str, object]:
    """Adapt the base configuration to a specific RecBole model."""
    config = dict(base_config)

    model_class = get_model(model)
    input_type = getattr(model_class, "input_type", None)

    if sequential:
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
                "MAX_ITEM_LIST_LENGTH": max_history,
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


def _export_atomic_files(
    data: ProcessedData,
    dataset_name: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Write the RecBole atomic files used by the RecBole baselines.

    Only ``user_id`` and ``item_id`` are exported. The RecBole configurations
    load identifier fields exclusively (see ``_build_config_dict``), so
    user/item side features are intentionally omitted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    positives = data.positive_splits()
    split_frames = {
        "train": positives["train"],
        "valid": positives["val"],
        "test": positives["test"],
    }

    atomic_files: dict[str, Path] = {}

    combined = pd.concat(split_frames.values(), axis=0, ignore_index=True)

    atomic_files["inter"] = output_dir / f"{dataset_name}.inter"
    _inter_id_frame(combined).to_csv(atomic_files["inter"], sep="\t", index=False)

    for split_name, split_df in split_frames.items():
        path = output_dir / f"{dataset_name}.{split_name}.inter"
        atomic_files[f"{split_name}.inter"] = path
        _inter_id_frame(split_df).to_csv(path, sep="\t", index=False)

    atomic_files["user"] = output_dir / f"{dataset_name}.user"
    _id_frame(data.user_features, settings.USER_COL).to_csv(
        atomic_files["user"], sep="\t", index=False
    )

    atomic_files["item"] = output_dir / f"{dataset_name}.item"
    _id_frame(data.item_features, settings.ITEM_COL).to_csv(
        atomic_files["item"], sep="\t", index=False
    )

    return atomic_files


def _export_sequential_benchmark(
    views: Mapping[str, RecSysDataset],
    dataset_name: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Write split-preserving sequence rows for RecBole sequential models.

    ``views`` maps each exported split name (``train_seq``, ``valid_seq``,
    ``test_seq``) to its query dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_files: dict[str, Path] = {}

    for split_name in SEQUENTIAL_BENCHMARK_SPLITS:
        view = views.get(split_name)
        if view is None:
            raise RuntimeError(f"Sequential benchmark split {split_name!r} is missing.")
        path = output_dir / f"{dataset_name}.{split_name}.inter"
        _build_sequential_frame(split_name, view).to_csv(path, sep="\t", index=False)
        atomic_files[split_name] = path

    return atomic_files


def _build_sequential_frame(split_name: str, view: RecSysDataset) -> pd.DataFrame:
    """RecBole rows with the split's target item and its strictly-past items."""
    history_mask = view.history_valid_mask.bool()
    user_ids = view.user_ids
    target_item_ids = view.target_item_ids
    history_items = view.history_items

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


def _inter_id_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction identifiers as RecBole tokens, one row per interaction."""
    missing = {settings.USER_COL, settings.ITEM_COL}.difference(df.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise RuntimeError(f"Interactions are missing identifier columns: {names}.")

    return pd.DataFrame(
        {
            f"{settings.USER_COL}:token": _token_column(
                cast(pd.Series, df[settings.USER_COL])
            ),
            f"{settings.ITEM_COL}:token": _token_column(
                cast(pd.Series, df[settings.ITEM_COL])
            ),
        }
    )


def _id_frame(
    frame: pd.DataFrame | None,
    id_col: str,
) -> pd.DataFrame:
    if frame is None:
        raise RuntimeError(f"{id_col} features are not available.")

    ids = frame[id_col].astype("int64").sort_values()
    return pd.DataFrame({f"{id_col}:token": ids.astype(str).to_numpy()})


def _token_column(values: pd.Series) -> pd.Series:
    return values.astype("int64").astype(str)


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


def eval_recbole_baselines(
    models: list[str],
    dm: ElearningDataModule,
    epochs: int = settings.EPOCHS,
    lr: float = settings.LR,
    batch_size: int = settings.BATCH_SIZE,
    patience: int = settings.PATIENCE,
    topks: list[int] | None = None,
    adaptive_k: bool = settings.ADAPTIVE_K,
    show_progress: bool = False,
    adapter_factory: Callable[[str], RecBoleAdapter] | None = None,
) -> pd.DataFrame:
    """Train and evaluate RecBole baselines on the shared query protocol.

    ``adapter_factory`` receives each model name and may return a subclass of
    ``RecBoleAdapter``; the default resolves the protocol from RecBole.
    """
    dm.validate_topks(topks if topks is not None else settings.TOP_KS)
    dataset_name = dm.data_variant
    atomic_dataset_dir = dm.atomic_folder
    dm.setup(stage="test")

    make_adapter: Callable[[str], RecBoleAdapter]
    if adapter_factory is None:
        make_adapter = lambda model: RecBoleAdapter.for_model(
            model, dm.data, dm.query_config
        )
    else:
        make_adapter = adapter_factory

    base_config = _build_config_dict(
        data_root=atomic_dataset_dir.parent,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        topks=topks or settings.TOP_KS,
        show_progress=show_progress,
    )

    results: list[dict[str, Any]] = []

    for model in models:
        if not dm.data.has_temporal_order and _is_sequential_model(model):
            print(
                f"[EVAL] Skipping {model}: it is a sequential model and "
                f"{dataset_name} has no temporal order."
            )
            continue

        adapter = make_adapter(model)
        if not isinstance(adapter, RecBoleAdapter):
            raise TypeError("RecBole model adapters must extend RecBoleAdapter.")
        adapter.prepare("fit")
        adapter.export(atomic_dataset_dir, dataset_name=dataset_name)

        print(f"[EVAL] Evaluating {model}...")

        metrics, training_time, inference_time = _run_model(
            model=model,
            dataset_name=dataset_name,
            config_dict=adapter.configure(model, base_config),
            dm=dm,
            adapter=adapter,
            adaptive_k=adaptive_k,
        )

        result = {
            "model": model,
            "seed": settings.state["random_state"],
            **metrics,
            "training_time_s": training_time,
            "inference_time_s": inference_time,
        }

        results.append(result)

    return pd.DataFrame(results)


def _run_model(
    model: str,
    dataset_name: str,
    config_dict: dict[str, object],
    dm: ElearningDataModule,
    adapter: RecBoleAdapter,
    adaptive_k: bool,
) -> tuple[dict[str, Any], float, float]:
    """Run RecBole's training and evaluation stages.

    Args:
        model: Name of the RecBole model.
        dataset_name: Name of the atomic dataset.
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
            run_config,
            dm,
            adapter,
            adaptive_k,
        )


def _fit_and_evaluate_model(
    model: str,
    dataset_name: str,
    config_dict: dict[str, object],
    dm: ElearningDataModule,
    adapter: RecBoleAdapter,
    adaptive_k: bool,
) -> tuple[dict[str, Any], float, float]:
    config = Config(
        model=model,
        dataset=dataset_name,
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

    if device.type == "cuda":
        print(f"[EVAL] {model_name} running on GPU ({device})")
    else:
        print(f"[EVAL] {model_name} running on CPU")

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
        trainer.fit(train_data, valid_data, saved=True, show_progress=show_progress)

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
            adapter=adapter,
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
    adapter: RecBoleAdapter,
    config: Config,
    adaptive_k: bool,
) -> dict[str, float]:
    """Evaluate RecBole scores with KGSeq's one-target-per-query protocol."""
    device = cast(torch.device, config["device"])
    topks = list(cast(list[int], config["topk"]))
    metrics = build_ranking_metrics(topks, adaptive_k=adaptive_k).to(device)

    adapter.prepare("test")
    adapter.bind_catalog(dataset, device)

    for batch in dm.evaluation_dataloader():
        interaction = adapter.evaluation_input(batch)
        recbole_scores = _full_sort_scores(
            model=model,  # type: ignore
            trainer=trainer,
            dataset=dataset,
            interaction=interaction,
            device=device,
        )
        scores = adapter.canonical_scores(recbole_scores)

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


def _synchronize_device(device: torch.device) -> None:
    """Wait until all pending device operations have completed."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _recbole_gpu_id() -> str:
    """RecBole GPU selector, written to ``CUDA_VISIBLE_DEVICES``.

    RecBole overwrites ``CUDA_VISIBLE_DEVICES`` with this value and derives the
    device from ``torch.cuda.is_available()``, so an out-of-range index hides
    every GPU and silently forces CPU. An explicit ``CUDA_VISIBLE_DEVICES``
    restriction is honoured; otherwise the first GPU is selected. ``""`` forces
    CPU execution.
    """
    if settings.state["device"] == "cpu":
        return ""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return visible.strip()

    return "0" if torch.cuda.is_available() else ""


def _build_config_dict(
    data_root: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    topks: list[int],
    show_progress: bool = False,
) -> dict[str, object]:
    """Build the common RecBole configuration.

    Only identifier fields are loaded because none of the supported RecBole
    baselines consume user or item side features. Sequential models replace the
    interaction columns with the item history list.
    """
    load_col: dict[str, list[str]] = {
        "inter": [settings.USER_COL, settings.ITEM_COL],
        "user": [settings.USER_COL],
        "item": [settings.ITEM_COL],
    }

    gpu_id = _recbole_gpu_id()

    return {
        "data_path": str(data_root),
        "benchmark_filename": list(BENCHMARK_SPLITS),
        "USER_ID_FIELD": settings.USER_COL,
        "ITEM_ID_FIELD": settings.ITEM_COL,
        "load_col": load_col,
        "seed": settings.state["random_state"],
        "reproducibility": True,
        # RecBole derives the device from gpu_id (it ignores use_gpu), so the
        # selector must point at a device this machine actually exposes.
        "gpu_id": gpu_id,
        "use_gpu": bool(gpu_id),
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
