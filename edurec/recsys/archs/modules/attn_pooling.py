import torch
from torch import nn


class AttentionPooling(nn.Module):
    """Attention-weighted pooling over a padded sequence.

    A learnable scoring function assigns a weight to every step and the module
    returns the weighted sum of the sequence.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the pooled representation with shape ``[batch, dim]``.

        Args:
            sequence: Hidden states with shape ``[batch, seq_len, dim]``.
            mask: Optional boolean mask with shape ``[batch, seq_len]``. True
                values indicate valid steps; padded steps are ignored. Rows
                with no valid step receive a zero vector.
        """
        scores = self.attn(sequence).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = torch.nan_to_num(scores.softmax(dim=1))
        return torch.einsum("bl,blh->bh", weights, sequence)


def masked_mean_pool(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Uniform mean over valid steps with shape ``[batch, dim]``.

    Args:
        sequence: Hidden states with shape ``[batch, seq_len, dim]``.
        mask: Boolean mask with shape ``[batch, seq_len]``. True values indicate
            valid steps; padded steps are ignored. Rows with no valid step
            receive a zero vector.
    """
    weights = mask.bool().unsqueeze(-1).to(sequence.dtype)
    return (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
