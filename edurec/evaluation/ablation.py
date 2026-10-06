from dataclasses import replace
from typing import Any

from edurec.recsys.configs import ModelArch, ModelConfig

FULL_ABLATION: dict[str, Any] = {
    "graph_mode": "kg",
    "use_text_features": True,
    "use_user_features": True,
    "use_item_bias": True,
    "scorer_type": "mlp",
    "use_attention_pooling": True,
}


ABLATIONS: dict[str, dict[str, Any]] = {
    "full": dict(FULL_ABLATION),
    "no_graph": {**FULL_ABLATION, "graph_mode": "id"},
    "no_text": {**FULL_ABLATION, "use_text_features": False},
    "no_user": {**FULL_ABLATION, "use_user_features": False},
    "no_item_bias": {**FULL_ABLATION, "use_item_bias": False},
    "no_attention_pooling": {**FULL_ABLATION, "use_attention_pooling": False},
    "dot_product": {**FULL_ABLATION, "scorer_type": "dot"},
    "candidate_attention": {
        **FULL_ABLATION,
        "scorer_type": "candidate_attention",
    },
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

    if base_cfg.arch == ModelArch.SASREC_TEXT:
        # Only content removal changes this fixed causal/dot architecture.
        return variant == "no_text" and base_cfg.num_item_text_feats > 0

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
