from dataclasses import replace
from typing import Any

from ..recsys.configs import ModelArch, ModelConfig

BASE_ABLATION: dict[str, Any] = {
    "graph_mode": "id",
    "use_text_features": False,
    "use_seq_encoder": False,
    "use_gcl": False,
    "use_item_bias": False,
    "scorer_type": "dot",
}

FULL_ABLATION: dict[str, Any] = {
    "graph_mode": "kg",
    "use_text_features": True,
    "use_seq_encoder": True,
    "use_gcl": True,
    "use_item_bias": True,
    "scorer_type": "mlp",
}


ABLATIONS: dict[str, dict[str, Any]] = {
    "base": dict(BASE_ABLATION),
    "full": dict(FULL_ABLATION),
    "no_graph": {**FULL_ABLATION, "graph_mode": "id"},
    "no_text": {**FULL_ABLATION, "use_text_features": False},
    "no_sequence": {
        **FULL_ABLATION,
        "use_seq_encoder": False,
    },
    "no_gcl": {**FULL_ABLATION, "use_gcl": False},
    "no_item_bias": {**FULL_ABLATION, "use_item_bias": False},
    "dot_product": {**FULL_ABLATION, "scorer_type": "dot"},
}


def get_ablation_config(base_cfg: ModelConfig, variant: str) -> ModelConfig:
    try:
        overrides = ABLATIONS[variant]
    except KeyError as exc:
        choices = ", ".join(ABLATIONS)
        raise ValueError(
            f"Unknown ablation variant {variant!r}. Available variants: {choices}."
        ) from exc
    return replace(base_cfg, **overrides)


def ablation_applicable(base_cfg: ModelConfig, variant: str) -> bool:
    """Whether a variant actually changes the model for this dataset.

    Variants that disable a module the dataset does not provide (e.g.
    ``no_sequence`` on a dataset without history) would otherwise be silently
    identical to ``full`` and report a meaningless zero importance.
    """
    candidate = get_ablation_config(base_cfg, variant)

    if candidate.arch == ModelArch.KG_SEQ and not candidate.uses_sequence:
        return False

    if variant in {"base", "full", "dot_product"}:
        return True

    full = get_ablation_config(base_cfg, "full")

    if variant == "no_graph":
        return full.graph_mode == "kg"
    if variant == "no_gcl":
        return full.use_gcl and full.graph_mode == "kg"
    if variant == "no_item_bias":
        return full.use_item_bias
    if variant == "no_text":
        return full.num_user_text_feats > 0 or full.num_item_text_feats > 0
    if variant == "no_sequence":
        return full.uses_sequence

    return full != candidate
