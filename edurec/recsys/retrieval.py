import lightning.pytorch as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import MetricCollection
from torchmetrics.retrieval import (
    RetrievalHitRate,
    RetrievalNormalizedDCG,
    RetrievalRecall,
)

from .. import settings
from ..datasets import RetrievalQuery
from .architecture import RetrievalConfig, TwoTowerRetrieval


class Retrieval(L.LightningModule):
    def __init__(
        self,
        cfg: RetrievalConfig,
        u_static_feats: torch.Tensor,
        i_static_feats: torch.Tensor,
        lr: float = settings.RETRIEVAL_LR,
        weight_decay: float = settings.RETRIEVAL_WEIGHT_DECAY,
        top_k: int = settings.RETRIEVAL_TOP_K,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["u_static_feats", "i_static_feats"])
        self.cfg = cfg
        self.lr = lr
        self.weight_decay = weight_decay
        self.top_k = top_k

        self.register_buffer("u_static_feats", u_static_feats, persistent=False)
        self.register_buffer("i_static_feats", i_static_feats, persistent=False)

        self.model = TwoTowerRetrieval(cfg)
        self.model_name = self.model.__class__.__name__

        ranking_metrics = MetricCollection(
            {
                f"Hit@{top_k}": RetrievalHitRate(top_k=top_k),
                f"Recall@{top_k}": RetrievalRecall(top_k=top_k),
                f"NDCG@{top_k}": RetrievalNormalizedDCG(top_k=top_k),
            }
        )

        self.val_ranking_metrics = ranking_metrics.clone(prefix="val/")
        self.test_ranking_metrics = ranking_metrics.clone(prefix="test/")

    def forward(self, batch: RetrievalQuery) -> torch.Tensor:
        query_emb = self.encode_query(
            user_ids=batch.user_id,
            history_items=batch.history_items,
            history_ctx=batch.history_ctx,
            history_valid_mask=batch.history_valid_mask,
        )
        item_emb = self.encode_items(batch.positive_item_id)
        return (query_emb @ item_emb.T) / self.cfg.temperature

    def encode_query(
        self,
        user_ids: torch.Tensor,
        history_items: torch.Tensor,
        history_ctx: torch.Tensor,
        history_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert isinstance(self.u_static_feats, torch.Tensor)
        assert isinstance(self.i_static_feats, torch.Tensor)

        return self.model.encode_query(
            user_ids=user_ids,
            history_items=history_items,
            history_ctx=history_ctx,
            history_valid_mask=history_valid_mask,
            u_static_feats=self.u_static_feats,
            i_static_feats=self.i_static_feats,
        )

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.i_static_feats, torch.Tensor)

        return self.model.encode_items(
            item_ids=item_ids,
            i_static_feats=self.i_static_feats,
        )

    def training_step(self, batch: RetrievalQuery) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: RetrievalQuery):
        self._step(batch, "val", ranking_metrics=self.val_ranking_metrics)

    def test_step(self, batch: RetrievalQuery):
        self._step(batch, "test", ranking_metrics=self.test_ranking_metrics)

    def _step(
        self,
        batch: RetrievalQuery,
        prefix: str,
        ranking_metrics: MetricCollection | None = None,
    ) -> torch.Tensor:
        logits = self(batch)
        targets = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, targets)

        batch_size = batch.user_id.size(0)
        self.log(
            f"{prefix}/Loss",
            loss,
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=(prefix != "test"),
            batch_size=batch_size,
        )

        if ranking_metrics is not None:
            num_candidates = logits.size(1)
            preds = logits.flatten()
            target = torch.eye(num_candidates, device=logits.device).flatten().long()
            indexes = batch.query_id.long().repeat_interleave(num_candidates)
            ranking_metrics.update(preds=preds, target=target, indexes=indexes)

        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.val_ranking_metrics.compute())
        self.val_ranking_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_ranking_metrics.compute())
        self.test_ranking_metrics.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
