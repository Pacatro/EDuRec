from enum import StrEnum

import torch
from torch import nn
from torch.nn import functional as F


class LossReduction(StrEnum):
    MEAN = "mean"
    SUM = "sum"


class InfoNCELoss(nn.Module):
    def __init__(
        self,
        tau: float = 0.1,
        reduction: LossReduction = LossReduction.MEAN,
        chunk_size: int = 1024,
    ):
        super().__init__()
        self.tau = tau
        self.reduction = reduction
        self.chunk_size = chunk_size

    def forward(
        self,
        u_emb1: torch.Tensor,
        i_emb1: torch.Tensor,
        u_emb2: torch.Tensor,
        i_emb2: torch.Tensor,
    ) -> torch.Tensor:
        loss_u = self._loss(u_emb1, u_emb2)
        loss_i = self._loss(i_emb1, i_emb2)

        if self.reduction == LossReduction.SUM:
            return loss_u + loss_i

        return 0.5 * (loss_u + loss_i)

    def _loss(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)

        loss_12 = self._directional_loss(h1, h2)
        loss_21 = self._directional_loss(h2, h1)

        return 0.5 * (loss_12 + loss_21)

    def _directional_loss(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
    ) -> torch.Tensor:
        if anchors.size(0) != positives.size(0):
            raise ValueError(
                "InfoNCE expects the same number of anchors and positives, got "
                f"{anchors.size(0)} and {positives.size(0)}."
            )

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}.")

        num_samples = anchors.size(0)
        total_loss = anchors.new_zeros(())

        for start in range(0, num_samples, self.chunk_size):
            end = min(start + self.chunk_size, num_samples)
            logits = anchors[start:end] @ positives.T / self.tau
            labels = torch.arange(start, end, device=anchors.device)
            total_loss = total_loss + F.cross_entropy(
                logits,
                labels,
                reduction="sum",
            )

        return total_loss / num_samples
