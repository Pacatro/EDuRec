import torch
from torch import nn


class GatedFusion(nn.Module):
    """Fuse two representations through a learned per-dimension gate.

    The gate is computed from both inputs so the model decides, for every
    dimension, how much of the primary (interaction) representation to keep
    versus the side (profile) representation.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.gate = nn.Linear(emb_dim * 2, emb_dim)

    def forward(
        self,
        primary: torch.Tensor,
        side: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([primary, side], dim=-1)))
        return gate * primary + (1.0 - gate) * side
