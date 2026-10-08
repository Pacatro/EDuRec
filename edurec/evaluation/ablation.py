from dataclasses import replace
from typing import Any

from edurec.recsys.configs import ModelArch, ModelConfig

# The default ("full") architecture: item KG + GRU, user-profile initial state,
# interaction context and time gaps, last valid hidden state (no attention),
# item bias and GCL regularization (weight taken from the dataset config).
FULL_ABLATION: dict[str, Any] = {
    "graph_mode": "kg",
    "use_text_features": True,
    "use_item_features": True,
    "use_user_features": True,
    "use_interaction_features": True,
    "use_time_features": True,
    "use_attention_pooling": False,
    "use_item_bias": True,
    "use_gcl": True,
}

# Everything switched off: a plain recurrent model over item-ID embeddings.
SEQ_ONLY: dict[str, Any] = {
    **FULL_ABLATION,
    "graph_mode": "id",
    "use_text_features": False,
    "use_item_features": False,
    "use_user_features": False,
    "use_interaction_features": False,
    "use_time_features": False,
    "use_item_bias": False,
    "use_gcl": False,
}


ABLATIONS: dict[str, dict[str, Any]] = {
    # Reference: the complete architecture.
    "full": dict(FULL_ABLATION),
    # Lower bound: no KG, no content features, no profile/context/time.
    "seq_only": dict(SEQ_ONLY),
    # Leave-one-out variants: each removes exactly one module so its marginal
    # contribution can be read off directly against ``full``.
    "no_graph": {**FULL_ABLATION, "graph_mode": "id"},
    "no_gcl": {**FULL_ABLATION, "use_gcl": False},
    "no_context": {**FULL_ABLATION, "use_interaction_features": False},
    "no_time": {**FULL_ABLATION, "use_time_features": False},
    "no_user": {**FULL_ABLATION, "use_user_features": False},
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
    ``no_graph`` on a dataset whose knowledge graph has no relations, or
    ``no_gcl`` when GCL is disabled) would otherwise be silently identical to
    ``full`` and report a meaningless zero importance.
    """
    if variant == "full":
        return True

    if base_cfg.arch == ModelArch.SASREC_TEXT:
        # The current ablations target KGSeq modules that the fixed SASRec
        # architecture does not expose, so none of them change it.
        return False

    full = get_ablation_config(base_cfg, "full")
    user_active = full.use_user_features and full.user_state.is_active
    context_active = (
        full.use_interaction_features and full.interaction_context.is_active
    )

    if variant == "no_graph":
        return full.graph_mode == "kg" and bool(full.kg_edge_types)
    if variant == "no_gcl":
        return full.gcl_enabled
    if variant == "no_context":
        return context_active
    if variant == "no_time":
        return full.use_time_features
    if variant == "no_user":
        return user_active

    return full != get_ablation_config(base_cfg, variant)
