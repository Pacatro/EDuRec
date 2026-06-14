from dataclasses import replace
from typing import Any

from ..recsys.architecture import EDuRecConfig


ABLATIONS: dict[str, dict[str, Any]] = {
    "base": {
        "graph_mode": "id",
        "use_sasrec": False,
        "use_context": False,
        "use_gcl": False,
    },
    "base_graph": {
        "graph_mode": "lightgcn",
        "use_sasrec": False,
        "use_context": False,
        "use_gcl": False,
    },
    "base_sequence": {
        "graph_mode": "id",
        "use_sasrec": True,
        "use_context": True,
        "use_gcl": False,
    },
    "base_graph_sequence": {
        "graph_mode": "lightgcn",
        "use_sasrec": True,
        "use_context": True,
        "use_gcl": False,
    },
    "full": {
        "graph_mode": "lightgcn",
        "use_sasrec": True,
        "use_context": True,
        "use_gcl": True,
    },
}

CONTENT_ABLATIONS: dict[str, dict[str, Any]] = {
    "no_user_features": {"use_user_features": False},
    "no_item_features": {"use_item_features": False},
    "no_text_features": {"use_text_features": False},
    "no_context": {"use_context": False},
    "no_gcl": {"use_gcl": False},
    "dot_product": {"scorer_type": "dot"},
}


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
