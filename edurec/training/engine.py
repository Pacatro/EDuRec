from typing import Any, Protocol

import lightning.pytorch as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import Metric, MetricCollection
from torchmetrics.retrieval import (
    RetrievalAUROC,
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)

from .. import config


# WARNING: This is a temporary solution until we find a better arquitecture for the model
class ModelProto(Protocol):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...

    def compute_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]: ...

    def compute_ranking_metrics(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        ranking_metrics: MetricCollection,
    ) -> None: ...


class RetrievalFBetaScore(Metric):
    def __init__(
        self, top_k: int = 10, beta: float = 1.0, adaptive_k: bool = False, **kwargs
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
        model: ModelProto,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        top_k: int = 10,
        alpha: float = 0.5,
        monitor: str = config.MONITOR,
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
        self.top_k = top_k

        ranking_metrics = MetricCollection(
            {
                f"Precision@{top_k}": RetrievalPrecision(top_k=top_k, adaptive_k=True),
                f"Recall@{top_k}": RetrievalRecall(top_k=top_k),
                # f"F1@{top_k}": RetrievalFBetaScore(
                #     top_k=top_k, beta=1.0, adaptive_k=True
                # ),
                f"Ndcg@{top_k}": RetrievalNormalizedDCG(top_k=top_k),
                f"Hit@{top_k}": RetrievalHitRate(top_k=top_k),
                f"Map@{top_k}": RetrievalMAP(top_k=top_k),
                f"Mrr@{top_k}": RetrievalMRR(top_k=top_k),
                f"AUROC@{top_k}": RetrievalAUROC(top_k=top_k),
            }
        )

        self.val_ranking_metrics = ranking_metrics.clone(prefix="val/")
        self.test_ranking_metrics = ranking_metrics.clone(prefix="test/")

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model.forward(batch)

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        prefix: str,
        ranking_metrics: MetricCollection | None = None,
    ):
        preds = self.model.forward(batch)
        loss, logs = self.model.compute_loss(preds, batch, self.alpha)

        self.log(f"{prefix}/Loss", loss, prog_bar=True)

        if logs is not None:
            for k, v in logs.items():
                self.log(f"{prefix}/{k}", v, on_epoch=True)

        if ranking_metrics is not None:
            self.model.compute_ranking_metrics(preds, batch, ranking_metrics)

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
            optimizer,
            mode="min" if "Loss" in self.monitor else "max",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }
