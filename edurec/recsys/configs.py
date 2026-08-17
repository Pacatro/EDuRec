from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Self

import yaml

from .. import settings


@dataclass
class TrainConfig:
    """Training hyperparameters, independent from the model architecture."""

    epochs: int = settings.EPOCHS
    lr: float = settings.LR
    batch_size: int = settings.BATCH_SIZE
    patience: int = settings.PATIENCE
    weight_decay: float = settings.WEIGHT_DECAY
    topks: list[int] = field(default_factory=lambda: list(settings.TOP_KS))
    alpha: float = settings.LOSS_ALPHA
    adaptive_k: bool = settings.ADAPTIVE_K

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)

    @classmethod
    def load(cls, path: Path | str) -> Self:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            return cls(**yaml.safe_load(f))


def resolve_train_config(
    cli: Mapping[str, Any] | None = None,
    saved_path: Path | str | None = None,
    defaults: TrainConfig | None = None,
) -> TrainConfig:
    """Resolve the effective training config for a run.

    Precedence: explicit CLI values win over the saved config file, which
    wins over the provided defaults (global or per-dataset).
    """
    resolved = defaults if defaults is not None else TrainConfig()
    if saved_path is not None and Path(saved_path).exists():
        resolved = replace(resolved, **asdict(TrainConfig.load(saved_path)))
    if cli:
        resolved = replace(
            resolved,
            **{name: value for name, value in cli.items() if value is not None},
        )
    return resolved


def monitor_topk(top_k: int | None, train_cfg: TrainConfig) -> int:
    """The cutoff that drives early stopping and checkpointing."""
    return top_k if top_k is not None else max(train_cfg.topks)
