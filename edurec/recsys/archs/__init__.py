from edurec.recsys.archs.base import BaseRecArch
from edurec.recsys.archs.kgseq import KGSeq
from edurec.recsys.archs.kgtransformer import KGTransformer
from edurec.recsys.configs import ModelConfig

ARCHS: dict[str, type[BaseRecArch]] = {
    "kg_rnn": KGSeq,
    "kg_transformer": KGTransformer,
}

ARCH_LABELS: dict[str, str] = {
    "kg_rnn": "KGSeq",
    "kg_transformer": "KGTransformer",
}


def build_model(cfg: ModelConfig) -> BaseRecArch:
    """Instantiate the architecture selected by ``cfg.arch``.

    Raises a clear error for unknown architectures and for datasets without a
    chronological history, which is required because users are represented
    solely by their interaction sequence.
    """
    arch = str(cfg.arch)
    try:
        arch_cls = ARCHS[arch]
    except KeyError as exc:
        choices = ", ".join(ARCHS)
        raise ValueError(
            f"Unknown architecture {arch!r}. Available architectures: {choices}."
        ) from exc

    if not cfg.has_history:
        raise ValueError(
            "This architecture requires sequential history: the dataset must "
            "provide a chronological timestamp."
        )

    return arch_cls(cfg)


def arch_label(cfg: ModelConfig) -> str:
    return ARCH_LABELS[str(cfg.arch)]


__all__ = [
    "ARCHS",
    "ARCH_LABELS",
    "BaseRecArch",
    "KGSeq",
    "KGTransformer",
    "arch_label",
    "build_model",
]
