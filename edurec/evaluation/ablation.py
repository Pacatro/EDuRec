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
    # The effective flags for this variant are resolved from dataset metadata in
    # get_ablation_config rather than assuming every optional input exists.
    "availability_aware": dict(FULL_ABLATION),
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

def get_ablation_config(base_cfg: EDuRecConfig, variant: str) -> EDuRecConfig:
    cfg = replace(base_cfg, **ABLATIONS[variant])
    if variant != "availability_aware":
        return cfg

    return replace(
        cfg,
        use_user_features=base_cfg.has_user_features,
        use_item_features=base_cfg.has_item_features,
        use_text_features=(
            base_cfg.num_user_text_feats > 0 or base_cfg.num_item_text_feats > 0
        ),
        use_seq_encoder=base_cfg.has_history,
        use_context=base_cfg.num_ctx_feats > 0,
    )
