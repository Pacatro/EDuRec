from dataclasses import replace
from typing import Any

from ..recsys.architecture import EDuRecConfig


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


def get_ablation_config(base_cfg: EDuRecConfig, variant: str) -> EDuRecConfig:
    try:
        overrides = ABLATIONS[variant]
    except KeyError as exc:
        choices = ", ".join(ABLATIONS)
        raise ValueError(
            f"Unknown ablation variant {variant!r}. Available variants: {choices}."
        ) from exc
    return replace(base_cfg, **overrides)
