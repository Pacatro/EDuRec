from enum import StrEnum

import torch
from torch import nn
from torch.nn import functional as F


class LossReduction(StrEnum):
    MEAN = "mean"
    SUM = "sum"


class InfoNCELoss(nn.Module):
    def __init__(self, tau: float = 0.1, reduction: LossReduction = LossReduction.MEAN):
        super().__init__()
        self.tau = tau
        self.reduction = reduction

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

        logits_12 = h1 @ h2.T / self.tau
        logits_21 = h2 @ h1.T / self.tau

        labels = torch.arange(h1.size(0), device=h1.device)

        loss_12 = F.cross_entropy(logits_12, labels)
        loss_21 = F.cross_entropy(logits_21, labels)

        return 0.5 * (loss_12 + loss_21)
