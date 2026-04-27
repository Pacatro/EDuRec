from dataclasses import dataclass

import torch


@dataclass
class UserHistory:
    """
    Represents the prefix history of a user for next-item prediction.

    items: padded item ids (0 = padding)
    ctx: contextual features aligned with items
    valid_mask: mask indicating valid history positions
    """

    items: torch.Tensor
    ctx: torch.Tensor
    valid_mask: torch.Tensor
