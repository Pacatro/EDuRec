from typing import TYPE_CHECKING

from .base import BaseRecArch
from .kg_rnn import KGRNN
from .kg_seq import KGSeq

if TYPE_CHECKING:
    from ..configs import ModelConfig

ARCHS: dict[str, type[BaseRecArch]] = {
    "kg_rnn": KGRNN,
    "kg_seq": KGSeq,
}

ARCH_LABELS: dict[str, str] = {
    "kg_rnn": "KGRNN",
    "kg_seq": "KGSeq",
}


def build_model(cfg: "ModelConfig") -> BaseRecArch:
    """Instantiate the architecture selected by ``cfg.arch``.

    Raises a clear error for unknown architectures and for configurations that
    the serial ``kg_seq`` architecture cannot support (no sequential history).
    """
    arch = str(cfg.arch)
    try:
        arch_cls = ARCHS[arch]
    except KeyError as exc:
        choices = ", ".join(ARCHS)
        raise ValueError(
            f"Unknown architecture {arch!r}. Available architectures: {choices}."
        ) from exc

    if arch == "kg_seq" and not cfg.uses_sequence:
        raise ValueError(
            "The 'kg_seq' architecture requires sequential history: the dataset "
            "must provide a chronological timestamp and use_seq_encoder must be "
            "True."
        )

    return arch_cls(cfg)


def arch_applicable(cfg: "ModelConfig") -> bool:
    """Whether the selected architecture can be built for this configuration."""
    return not (str(cfg.arch) == "kg_seq" and not cfg.uses_sequence)


def arch_label(cfg: "ModelConfig") -> str:
    return ARCH_LABELS.get(str(cfg.arch), "KGRNN")


__all__ = [
    "ARCHS",
    "ARCH_LABELS",
    "KGRNN",
    "BaseRecArch",
    "KGSeq",
    "arch_applicable",
    "arch_label",
    "build_model",
]
