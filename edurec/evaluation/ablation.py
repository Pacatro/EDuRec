from dataclasses import replace
from typing import Any

from ..recsys.configs import ModelConfig

FULL_ABLATION: dict[str, Any] = {
    "graph_mode": "kg",
    "use_text_features": True,
    "use_user_features": True,
    "use_item_bias": True,
    "scorer_type": "mlp",
}


ABLATIONS: dict[str, dict[str, Any]] = {
    "full": dict(FULL_ABLATION),
    "no_graph": {**FULL_ABLATION, "graph_mode": "id"},
    "no_text": {**FULL_ABLATION, "use_text_features": False},
    "no_user": {**FULL_ABLATION, "use_user_features": False},
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
    ``no_text`` on a dataset without text features) would otherwise be silently
    identical to ``full`` and report a meaningless zero importance.
    """
    if variant == "full":
        return True

    candidate = get_ablation_config(base_cfg, variant)
    full = get_ablation_config(base_cfg, "full")

    if variant == "no_graph":
        return full.graph_mode == "kg"
    if variant == "no_item_bias":
        return full.use_item_bias
    if variant == "no_text":
        return full.num_item_text_feats > 0
    if variant == "no_user":
        return full.user_profile.is_active

    return full != candidate
