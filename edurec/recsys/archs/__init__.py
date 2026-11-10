from edurec import settings
from edurec.recsys.archs.base import BaseRecArch
from edurec.recsys.archs.kgseq import KGSeq
from edurec.recsys.archs.sasrec_text import SASRecText
from edurec.recsys.configs import ModelArch, ModelConfig

ARCHS: dict[ModelArch, tuple[type[BaseRecArch], str]] = {
    ModelArch.KG_RNN: (KGSeq, "KGSeq"),
    ModelArch.SASREC_TEXT: (SASRecText, "SASRecText"),
}


def build_model(
    cfg: ModelConfig,
    max_history_len: int = settings.MAX_HISTORY_LEN,
) -> BaseRecArch:
    """Instantiate the architecture selected by ``cfg.arch``.

    ``max_history_len`` sizes the positional embeddings of sequence-only
    architectures. Raises a clear error for unknown architectures or SASRec
    without history. KGSeq can instead represent users with their identifiers
    and static profiles.
    """
    arch = ModelArch(cfg.arch)
    try:
        arch_cls, _ = ARCHS[arch]
    except KeyError as exc:
        choices = ", ".join(item.value for item in ARCHS)
        raise ValueError(
            f"Unknown architecture {arch!r}. Available architectures: {choices}."
        ) from exc

    if not cfg.has_history and arch is ModelArch.SASREC_TEXT:
        raise ValueError(
            "This architecture requires sequential history: the dataset must "
            "provide a chronological timestamp."
        )

    if arch is ModelArch.SASREC_TEXT:
        return SASRecText(cfg, max_history_len=max_history_len)

    return arch_cls(cfg)


def arch_label(arch: ModelArch | str) -> str:
    return ARCHS[ModelArch(arch)][1]


__all__ = [
    "BaseRecArch",
    "KGSeq",
    "SASRecText",
    "arch_label",
    "build_model",
]
