import lightning.pytorch as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import MetricCollection
from torchmetrics.retrieval import RetrievalHitRate, RetrievalRecall

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
        self.register_buffer(
            "all_item_ids",
            torch.arange(cfg.num_items, dtype=torch.long),
            persistent=False,
        )

        self.model = TwoTowerRetrieval(cfg)
        self.model_name = self.model.__class__.__name__

        ranking_metrics = MetricCollection(
            {
                f"Hit@{top_k}": RetrievalHitRate(top_k=top_k),
                f"Recall@{top_k}": RetrievalRecall(top_k=top_k),
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
        candidate_item_ids = torch.cat(
            [batch.positive_item_id.unsqueeze(1), batch.negative_item_ids],
            dim=1,
        )
        return self._score_candidates(query_emb, candidate_item_ids)

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
        if prefix == "train":
            logits = self(batch)
            targets = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
        else:
            query_emb = self.encode_query(
                user_ids=batch.user_id,
                history_items=batch.history_items,
                history_ctx=batch.history_ctx,
                history_valid_mask=batch.history_valid_mask,
            )
            logits = self._score_candidates(query_emb, self.all_item_ids)
            targets = batch.positive_item_id.long()

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
            target = torch.zeros_like(logits, dtype=torch.long)
            target.scatter_(1, batch.positive_item_id.unsqueeze(1), 1)
            indexes = batch.query_id.long().repeat_interleave(num_candidates)
            ranking_metrics.update(
                preds=preds,
                target=target.flatten(),
                indexes=indexes,
            )

        return loss

    def _score_candidates(
        self,
        query_emb: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        item_emb = self.encode_items(item_ids)

        if item_ids.ndim == 1:
            return (query_emb @ item_emb.T) / self.cfg.temperature

        return (
            (query_emb.unsqueeze(1) * item_emb).sum(dim=-1) / self.cfg.temperature
        )

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
