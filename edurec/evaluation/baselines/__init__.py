"""Registry for native (non-RecBole) recommendation baselines.

These baselines are implemented with PyTorch Lightning and scored with the same
shared retrieval evaluator as the proposed EDuRec models and the RecBole
baselines (see ``recbole.py``), so their numbers are directly comparable.

Adding a new baseline
---------------------
1. Create a module in this subpackage, for example
   ``edurec/evaluation/baselines/my_model.py``.
2. Implement a runner with the standard keyword signature and register it::

    from edurec.evaluation.baselines import register_baseline

    @register_baseline("MyModel")
    def eval_my_model(
        dm: ElearningDataModule,
        train_cfg: TrainConfig,
        val_topk: int = settings.TOP_K,
        compile: bool = settings.COMPILE_MODEL,
        verbose: bool = False,
        run_name: str | None = None,
        experiment_name: str = settings.EXPERIMENT_NAME,
    ) -> pd.DataFrame:
        ...  # return a one-row frame whose "model" column is "MyModel"

3. That is all. The module is auto-discovered on import and the model becomes
   available through ``edurec eval --baseline-model MyModel`` (or
   ``--baseline-model all``). The returned ``DataFrame`` must use the shared
   metric names (``ndcg@k``, ``recall@k``, ``mrr@k``, ...) so the results line
   up with EDuRec and the RecBole baselines. The runner may accept extra optional
   keyword arguments; the CLI only passes the standard ones listed above.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Protocol

import pandas as pd

from edurec.datasets import ElearningDataModule
from edurec.recsys import TrainConfig


class BaselineRunner(Protocol):
    """Standard entry point every native baseline must implement."""

    def __call__(
        self,
        dm: ElearningDataModule,
        train_cfg: TrainConfig | None = ...,
        val_topk: int = ...,
        compile: bool = ...,
        verbose: bool = ...,
        run_name: str | None = ...,
        experiment_name: str = ...,
    ) -> pd.DataFrame: ...


_REGISTRY: dict[str, BaselineRunner] = {}
_TRAIN_DEFAULTS: dict[str, Callable[[], TrainConfig]] = {}


def register_baseline[Runner: BaselineRunner](
    label: str,
    *,
    train_defaults: Callable[[], TrainConfig] | None = None,
) -> Callable[[Runner], Runner]:
    """Register a baseline runner under ``label`` (used in the result files)."""

    def decorator(runner: Runner) -> Runner:
        if label in _REGISTRY:
            raise ValueError(f"Baseline {label!r} is already registered.")
        _REGISTRY[label] = runner
        if train_defaults is not None:
            _TRAIN_DEFAULTS[label] = train_defaults
        return runner

    return decorator


def baseline_train_defaults(label: str) -> TrainConfig | None:
    """Fresh training defaults from a baseline's reference implementation."""
    factory = _TRAIN_DEFAULTS.get(label)
    return factory() if factory is not None else None


def baseline_labels() -> list[str]:
    """Labels of every registered native baseline, in registration order."""
    return list(_REGISTRY)


def get_baseline(label: str) -> BaselineRunner | None:
    """Return the runner registered under ``label``, or ``None`` if unknown."""
    return _REGISTRY.get(label)


def discover_baselines() -> None:
    """Import every module in this subpackage.

    Modules that call :func:`register_baseline` become available to the CLI;
    modules that provide other baseline runners (such as ``recbole.py``) are
    imported without registering a native runner. Modules whose name starts
    with an underscore are skipped. Import errors surface immediately so a
    broken baseline fails fast instead of being silently omitted.
    """
    package = importlib.import_module(__name__)
    for info in pkgutil.iter_modules(list(package.__path__)):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{info.name}")


__all__ = [
    "baseline_labels",
    "baseline_train_defaults",
    "discover_baselines",
    "get_baseline",
    "register_baseline",
]
