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
    "scorer_type": "mlp",
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
    "no_sequence": {
        **FULL_ABLATION,
        "use_seq_encoder": False,
        "use_context": False,
    },
    "no_context": {**FULL_ABLATION, "use_context": False},
    "no_gcl": {**FULL_ABLATION, "use_gcl": False},
    "dot_product": {**FULL_ABLATION, "scorer_type": "dot", "hidden_dims": []},
}

CONTENT_ABLATIONS: dict[str, dict[str, Any]] = {}


def get_ablation_config(base_cfg: EDuRecConfig, variant: str) -> EDuRecConfig:
    return replace(base_cfg, **ABLATIONS[variant])


def get_content_ablation_config(base_cfg: EDuRecConfig, variant: str) -> EDuRecConfig:
    full_cfg = get_ablation_config(base_cfg, "full")
    return replace(full_cfg, **CONTENT_ABLATIONS[variant])


def ablation_variants(include_content: bool = True) -> dict[str, dict[str, Any]]:
    variants = dict(ABLATIONS)
    if include_content:
        variants.update(CONTENT_ABLATIONS)
    return variants
