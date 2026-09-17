from dataclasses import replace
from typing import Any

from ..recsys.configs import ModelConfig

BASE_ABLATION: dict[str, Any] = {
    "graph_mode": "id",
    "use_user_features": False,
    "use_item_features": False,
    "use_text_features": False,
    "use_seq_encoder": False,
    "use_context": False,
    "use_gcl": False,
    "use_item_bias": False,
    "scorer_type": "dot",
    "fusion_type": "sum",
    "hidden_dims": [],
}

FULL_ABLATION: dict[str, Any] = {
    "graph_mode": "lightgcn",
    "use_user_features": True,
    "use_item_features": True,
    "use_text_features": True,
    "use_seq_encoder": True,
    "use_context": True,
    "use_gcl": True,
    "use_item_bias": True,
    "scorer_type": "mlp",
    "fusion_type": "masked_gated",
}


ABLATIONS: dict[str, dict[str, Any]] = {
    "base": dict(BASE_ABLATION),
    "full": dict(FULL_ABLATION),
    "no_graph": {
        **FULL_ABLATION,
        "graph_mode": "none",
        "use_gcl": False,
    },
    "no_features": {
        **FULL_ABLATION,
        "use_user_features": False,
        "use_item_features": False,
        "use_text_features": False,
    },
    "no_user_features": {**FULL_ABLATION, "use_user_features": False},
    "no_item_features": {**FULL_ABLATION, "use_item_features": False},
    "no_text": {**FULL_ABLATION, "use_text_features": False},
    "no_sequence": {
        **FULL_ABLATION,
        "use_seq_encoder": False,
    },
    "no_context": {**FULL_ABLATION, "use_context": False},
    "sum_fusion": {**FULL_ABLATION, "fusion_type": "sum"},
    "no_gcl": {**FULL_ABLATION, "use_gcl": False},
    "no_item_bias": {**FULL_ABLATION, "use_item_bias": False},
    "dot_product": {**FULL_ABLATION, "scorer_type": "dot", "hidden_dims": []},
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
    if variant in {"base", "full", "dot_product"}:
        return True

    full = get_ablation_config(base_cfg, "full")
    candidate = get_ablation_config(base_cfg, variant)

    if variant == "no_graph":
        return full.available_modules["graph"]
    if variant == "no_gcl":
        return full.use_gcl and full.graph_mode == "lightgcn"
    if variant == "sum_fusion":
        return sum(full.available_modules.values()) >= 2
    if variant == "no_item_bias":
        return full.use_item_bias
    if variant == "no_text":
        return full.num_user_text_feats > 0 or full.num_item_text_feats > 0
    if variant == "no_features":
        return full.has_user_features or full.has_item_features
    if variant == "no_user_features":
        return full.has_user_features
    if variant == "no_item_features":
        return full.has_item_features
    if variant == "no_sequence":
        return full.available_modules["sequence"]
    if variant == "no_context":
        return full.available_modules["context"]

    return full.available_modules != candidate.available_modules
