import lightning.pytorch as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torchmetrics import Metric, MetricCollection
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
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        top_k: int = 10,
        alpha: float = 0.5,
        monitor: str = config.MONITOR,
        rating_loss_fn: nn.Module | None = None,
        relevance_loss_fn: nn.Module | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["model", "relevance_loss_fn", "rating_loss_fn"]
        )
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.monitor = monitor
        self.alpha = alpha

        self.rating_loss_fn = rating_loss_fn or nn.MSELoss()
        self.relevance_loss_fn = relevance_loss_fn or nn.BCELoss()

        ranking_metrics = MetricCollection(
            {
                f"Precision@{top_k}": RetrievalPrecision(top_k=top_k, adaptive_k=True),
                f"Recall@{top_k}": RetrievalRecall(top_k=top_k),
                f"F1@{top_k}": RetrievalFBetaScore(
                    top_k=top_k, beta=1.0, adaptive_k=True
                ),
                f"NDCG@{top_k}": RetrievalNormalizedDCG(top_k=top_k),
                f"HitRate@{top_k}": RetrievalHitRate(top_k=top_k),
                f"MAP@{top_k}": RetrievalMAP(top_k=top_k),
                f"MRR@{top_k}": RetrievalMRR(top_k=top_k),
                # f"AUROC@{top_k}": RetrievalAUROC(top_k=top_k),
            }
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
        preds = self.model(batch)

        pred_ratings = preds["rating"].flatten()
        pred_relevance = preds["relevance"].flatten()

        true_ratings = batch[config.RATING_COL].float().flatten()
        true_relevance = batch[config.RELEVANT_COL].float().flatten()

        loss_rating = self.rating_loss_fn(pred_ratings, true_ratings)
        loss_relevance = self.relevance_loss_fn(pred_relevance, true_relevance)
        loss = (self.alpha * loss_rating) + ((1 - self.alpha) * loss_relevance)

        self.log(f"{prefix}/MTLoss", loss, prog_bar=True)
        self.log(
            f"{prefix}/{self.rating_loss_fn.__class__.__name__}",
            loss_rating,
            prog_bar=False,
            on_epoch=True,
        )
        self.log(
            f"{prefix}/{self.relevance_loss_fn.__class__.__name__}",
            loss_relevance,
            prog_bar=True,
            on_epoch=True,
        )

        if ranking_metrics is not None:
            user_ids = batch[config.USER_COL].long().flatten()
            target = batch[config.RELEVANT_COL].bool().flatten()
            ranking_metrics.update(pred_relevance.detach(), target, indexes=user_ids)

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

    def on_validation_epoch_start(self) -> None:
        self.val_ranking_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_ranking_metrics.compute())

    def on_test_epoch_start(self) -> None:
        self.test_ranking_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_ranking_metrics.compute())

    def predict_step(self, batch: dict[str, torch.Tensor]) -> dict[str, int | float]:
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
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }
