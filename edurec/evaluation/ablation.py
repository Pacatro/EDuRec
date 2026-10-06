from dataclasses import replace
from typing import Any

from edurec.recsys.configs import ModelArch, ModelConfig

# The default ("full") architecture: item KG + GRU, user-profile initial state,
# interaction context and time gaps, last valid hidden state (no attention).
FULL_ABLATION: dict[str, Any] = {
    "graph_mode": "kg",
    "use_text_features": True,
    "use_item_features": True,
    "use_user_features": True,
    "use_interaction_features": True,
    "use_time_features": True,
    "use_attention_pooling": False,
    "use_item_bias": True,
}

ITEM_KG_ONLY: dict[str, Any] = {
    **FULL_ABLATION,
    "use_user_features": False,
    "use_interaction_features": False,
    "use_time_features": False,
}


ABLATIONS: dict[str, dict[str, Any]] = {
    # A. Item KG + GRU
    "item_kg": dict(ITEM_KG_ONLY),
    # B. A + user profile initialization
    "item_kg_user": {**ITEM_KG_ONLY, "use_user_features": True},
    # C. A + interaction context
    "item_kg_context": {**ITEM_KG_ONLY, "use_interaction_features": True},
    # D. A + time gaps
    "item_kg_time": {**ITEM_KG_ONLY, "use_time_features": True},
    # E. A + context + time + user profile (default architecture)
    "full": dict(FULL_ABLATION),
    # F. E + attention pooling
    "attention_pooling": {**FULL_ABLATION, "use_attention_pooling": True},
    # Structure ablations.
    "no_graph": {**FULL_ABLATION, "graph_mode": "id"},
    "no_text": {**FULL_ABLATION, "use_text_features": False},
    "no_item_bias": {**FULL_ABLATION, "use_item_bias": False},
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

    full = get_ablation_config(base_cfg, "full")
    user_active = full.use_user_features and full.user_state.is_active
    context_active = (
        full.use_interaction_features and full.interaction_context.is_active
    )

    if variant == "no_graph":
        return full.graph_mode == "kg"
    if variant == "no_item_bias":
        return full.use_item_bias
    if variant == "no_text":
        return full.num_item_text_feats > 0
    if variant == "item_kg_user":
        return user_active
    if variant == "item_kg_context":
        return context_active
    if variant == "item_kg_time":
        return full.use_time_features
    if variant == "attention_pooling":
        return True

    return full != get_ablation_config(base_cfg, variant)
