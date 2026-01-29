import lightning.pytorch as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torchmetrics import Metric, MetricCollection
from torchmetrics.functional import mean_absolute_error
from torchmetrics.retrieval import (
    # RetrievalAUROC,
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)

from .. import config


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking Loss for implicit feedback.

    Args:
        lambda_reg (float): L2 regularization parameter
    """

    def __init__(self, lambda_reg: float = 1e-4):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(
        self, pos_scores: torch.Tensor, neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pos_scores: scores for positive interactions
            neg_scores: scores for negative interactions

        Returns:
            BPR loss
        """
        diff = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(diff)).mean()

        # L2 regularization
        reg_loss = self.lambda_reg * (pos_scores.pow(2).sum() + neg_scores.pow(2).sum())

        return loss + reg_loss


class RetrievalFBetaScore(Metric):
    def __init__(
        self, top_k: int = 10, beta: float = 1.0, adaptive_k: bool = True, **kwargs
    ):
        super().__init__(**kwargs)
        self.beta = beta
        self.top_k = top_k

        self.precision = RetrievalPrecision(top_k=top_k, adaptive_k=adaptive_k)
        self.recall = RetrievalRecall(top_k=top_k)

    def update(self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor):
        self.precision.update(preds, target, indexes=indexes)
        self.recall.update(preds, target, indexes=indexes)

    def compute(self):
        precision = self.precision.compute()
        recall = self.recall.compute()

        return ((1 + self.beta**2) * precision * recall) / (
            (self.beta**2 * precision) + recall
        )

    def reset(self):
        self.precision.reset()
        self.recall.reset()


class RecSys(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        threshold: float = 8.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        top_k: int = 10,
        loss_fn: nn.Module | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = loss_fn or nn.MSELoss()
        self.threshold = threshold
        self.lr = lr
        self.weight_decay = weight_decay

        ranking_metrics = MetricCollection(
            RetrievalPrecision(top_k=top_k, adaptive_k=True),
            RetrievalRecall(top_k=top_k),
            RetrievalFBetaScore(top_k=top_k, beta=1.0, adaptive_k=True),
            RetrievalNormalizedDCG(top_k=top_k),
            RetrievalHitRate(top_k=top_k),
            RetrievalMAP(top_k=top_k),
            RetrievalMRR(top_k=top_k),
            # RetrievalAUROC(top_k=top_k),
        )

        self.val_ranking_metrics = ranking_metrics.clone(prefix="val/")
        self.test_ranking_metrics = ranking_metrics.clone(prefix="test/")

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch)

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        prefix: str,
        ranking_metrics: MetricCollection | None = None,
    ):
        preds: torch.Tensor = self.model(batch)
        ratings = batch[config.RATING_COL].float().view(-1)

        loss = self.loss_fn(preds, ratings)
        mae = mean_absolute_error(preds, ratings)
        rmse = loss**0.5

        if ranking_metrics is not None and prefix in ["val", "test"]:
            user_ids = batch[config.USER_COL].long().view(-1)
            target = batch[config.RELEVANT_COL].int().view(-1)
            ranking_metrics.update(preds.detach(), target, indexes=user_ids)

        self.log(f"{prefix}/MSE", loss, prog_bar=False)
        self.log(f"{prefix}/RMSE", rmse, prog_bar=True)
        self.log(f"{prefix}/MAE", mae, prog_bar=False)

        return loss

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor]) -> None:
        self._step(
            batch,
            "val",
            ranking_metrics=self.val_ranking_metrics,
        )

    def test_step(self, batch: dict[str, torch.Tensor]) -> None:
        self._step(
            batch,
            "test",
            ranking_metrics=self.test_ranking_metrics,
        )

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_ranking_metrics.compute())
        self.val_ranking_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_ranking_metrics.compute())
        self.test_ranking_metrics.reset()

    def predict_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self(batch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/MSE"},
        }
