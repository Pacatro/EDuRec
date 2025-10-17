import torch
from torch import nn
import lightning.pytorch as L
from typing import Optional
from torchmetrics import MetricCollection, Metric
from torchmetrics.retrieval import (
    RetrievalPrecision,
    RetrievalRecall,
    RetrievalNormalizedDCG,
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
)


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
        loss_fn: Optional[nn.Module] = None,
        val_metrics_path: str = "val_metrics.png",
        train_metrics_path: str = "train_metrics.png",
        train_losses_path: str = "train_losses.png",
        encoders: Optional[dict] = None,
        plot: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = nn.MSELoss() if loss_fn is None else loss_fn
        self.threshold = threshold
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_metrics_path = val_metrics_path
        self.train_metrics_path = train_metrics_path
        self.train_losses_path = train_losses_path
        self.plot = plot
        self.encoders = encoders

        ranking_metrics = MetricCollection(
            {
                f"Precision_{top_k}": RetrievalPrecision(top_k=top_k, adaptive_k=True),
                f"Recall_{top_k}": RetrievalRecall(top_k=top_k),
                f"F1_{top_k}": RetrievalFBetaScore(
                    top_k=top_k, beta=1.0, adaptive_k=True
                ),
                f"NDCG_{top_k}": RetrievalNormalizedDCG(top_k=top_k),
                f"HR_{top_k}": RetrievalHitRate(top_k=top_k),
                f"MAP_{top_k}": RetrievalMAP(top_k=top_k),
                f"MRR_{top_k}": RetrievalMRR(top_k=top_k),
            }
        )

        self.val_ranking_metrics = ranking_metrics.clone(prefix="val/")
        self.test_ranking_metrics = ranking_metrics.clone(prefix="test/")

    def forward(self, batch):
        score = self.model(batch)

        if self.encoders:
            user_id_tensor = batch["user_id"]
            item_id_tensor = batch["item_id"]

            user_id_array = user_id_tensor.detach().cpu().numpy()
            item_id_array = item_id_tensor.detach().cpu().numpy()

            user_id = self.encoders["user_id"].inverse_transform(
                user_id_array.reshape(-1, 1)
            )
            item_id = self.encoders["item_id"].inverse_transform(
                item_id_array.reshape(-1, 1)
            )
        else:
            user_id = batch["user_id"].long()
            item_id = batch["item_id"].long()

        return {
            "user_id": user_id.ravel(),
            "item_id": item_id.ravel(),
            "prediction": score.detach(),
            "relevant": score.detach() > self.threshold,
        }

    def _step(
        self,
        batch: dict,
        prefix: str,
        ranking_metrics: MetricCollection | None = None,
    ):
        ratings = batch["rating"]
        user_ids = batch["user_id"].long()

        preds = self.model(batch)
        loss = self.loss_fn(preds, ratings.float())

        if ranking_metrics is not None:
            target = (ratings >= self.threshold).int()
            ranking_metrics.update(
                preds,
                target,
                indexes=user_ids,
            )

        self.log(f"{prefix}/MSE", loss, prog_bar=True)
        self.log(f"{prefix}/RMSE", (loss**0.5), prog_bar=True)

        return loss

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        self._step(
            batch,
            "val",
            ranking_metrics=self.val_ranking_metrics,
        )

    def test_step(self, batch):
        self._step(
            batch,
            "test",
            ranking_metrics=self.test_ranking_metrics,
        )

    def on_validation_epoch_start(self):
        self.val_ranking_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_ranking_metrics.compute())

    def on_test_epoch_start(self):
        self.test_ranking_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_ranking_metrics.compute())

    def predict_step(self, batch):
        return self.forward(batch)

    def configure_optimizers(self):
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
