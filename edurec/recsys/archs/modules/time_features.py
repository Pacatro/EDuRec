import torch
from torch import nn


class TimeEncoder(nn.Module):
    """Encode non-negative time gaps into a dense embedding.

    Raw gaps (in seconds) span many orders of magnitude, so the value is
    compressed with ``log1p`` before the linear projection.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.proj = nn.Linear(1, emb_dim)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """Project time gaps with shape ``[batch, seq_len]``.

        Returns a tensor with shape ``[batch, seq_len, emb_dim]``. Negative
        deltas are clamped to zero before the log transform.
        """
        return self.proj(torch.log1p(delta.clamp(min=0.0)).unsqueeze(-1))
