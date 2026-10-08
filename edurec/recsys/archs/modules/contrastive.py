"""Training-only knowledge-graph views and item contrastive regularization."""

import torch
import torch.nn.functional as F

from edurec.recsys.archs.modules.kg_encoder import EdgeType


def drop_relation_edges(
    edge_index: dict[EdgeType, torch.Tensor], dropout: float
) -> dict[EdgeType, torch.Tensor]:
    """Drop edges together with their explicit ``rev_`` counterparts.

    Keep node IDs and all relation keys, including empty relations. Reverse
    edges are rebuilt from sampled forward edges: coalescing the original
    relations can give the two directions different column orders. Inputs
    are never modified. Sampling uses the device's seeded PyTorch RNG.
    """
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("Edge dropout must be between 0 and 1.")
    if dropout == 0.0:
        return dict(edge_index)

    view: dict[EdgeType, torch.Tensor] = {}
    for edge_type, edges in edge_index.items():
        source, relation, target = edge_type
        if relation.startswith("rev_"):
            forward_type = (target, relation.removeprefix("rev_"), source)
            if forward_type in edge_index:
                continue

        keep = torch.rand(edges.size(1), device=edges.device) >= dropout
        selected = edges[:, keep]
        view[edge_type] = selected
        reverse_type = (target, f"rev_{relation}", source)
        if reverse_type in edge_index:
            view[reverse_type] = selected.flip(0)

    return view


def item_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    item_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric cross-view InfoNCE for unique item IDs.

    ``first``/``second`` have shape ``[num_items, emb_dim]``; ``item_ids``
    selects the same items in both views. Other selected items are negatives.
    Compute normalized logits in float32 even under mixed precision.
    """
    if not temperature > 0.0:
        raise ValueError("Contrastive temperature must be positive.")
    ids = item_ids.unique()
    if ids.numel() < 2:
        return (first.sum() + second.sum()) * 0.0

    with torch.autocast(device_type=first.device.type, enabled=False):
        z1 = F.normalize(first[ids].float(), dim=-1)
        z2 = F.normalize(second[ids].float(), dim=-1)
        logits = z1 @ z2.T / temperature
        labels = torch.arange(ids.numel(), device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )
