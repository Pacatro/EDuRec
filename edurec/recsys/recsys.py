import lightning.pytorch as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch_geometric.utils import dropout_edge
from torch_geometric.data import Data
from torchmetrics import MetricCollection
from torchmetrics.retrieval import (
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)

from .. import settings
from ..datasets import RecSysQuery
from .architecture import EDuRec, EDuRecConfig
from .losses import InfoNCELoss, LossReduction


class RecSys(L.LightningModule):
    def __init__(
        self,
        cfg: EDuRecConfig,
        inter_graph: Data,
        u_static_feats: torch.Tensor,
        i_static_feats: torch.Tensor,
        val_topk: int = settings.TOP_K,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "inter_graph",
                "u_static_feats",
                "i_static_feats",
            ]
        )
        self.cfg = cfg
        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay
        self.alpha = cfg.alpha
        self.val_topk = val_topk
        self.topks = cfg.topks if cfg.topks else [settings.TOP_K]
        self.monitor = f"val/ndcg@{val_topk}"

        self.register_buffer("edge_index", inter_graph.edge_index, persistent=False)
        self.register_buffer("u_static_feats", u_static_feats, persistent=False)
        self.register_buffer("i_static_feats", i_static_feats, persistent=False)

        self.gcl_loss = InfoNCELoss(
            tau=cfg.temperature,
            reduction=LossReduction(cfg.loss_reduction),
        )

        self.val_ranking_metrics = MetricCollection(
            {
                f"ndcg@{self.val_topk}": RetrievalNormalizedDCG(
                    top_k=self.val_topk, empty_target_action="neg"
                )
            },
            prefix="val/",
        )

        if self.topks:
            metrics = {
                "precision": (RetrievalPrecision, {"adaptive_k": cfg.adaptive_k}),
                "recall": (RetrievalRecall, {}),
                "ndcg": (RetrievalNormalizedDCG, {}),
                "hit": (RetrievalHitRate, {}),
                "map": (RetrievalMAP, {}),
                "mrr": (RetrievalMRR, {}),
            }

            self.test_ranking_metrics = MetricCollection(
                {
                    f"{name}@{k}": cls(top_k=k, empty_target_action="neg", **kwargs)
                    for k in cfg.topks
                    for name, (cls, kwargs) in metrics.items()
                },
                prefix="test/",
            )

        self.model = EDuRec(cfg)
        self.model_name = self.__class__.__name__

    def forward(self, batch: RecSysQuery) -> torch.Tensor:
        scores = self.model(
            u_ids=batch.user_id,
            h_ids=batch.history_items,
            h_ctx=batch.history_ctx,
            h_mask=batch.history_valid_mask,
            edge_index=self.edge_index,
            u_static_feats=self.u_static_feats,
            i_static_feats=self.i_static_feats,
        )

        if scores.ndim == 3:
            if scores.size(-1) != 1:
                raise RuntimeError("RecSys must return a single logit per candidate.")
            return scores.squeeze(-1)

        return scores

    def training_step(self, batch: RecSysQuery) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: RecSysQuery):
        self._step(
            batch,
            "val",
            ranking_metrics=self.val_ranking_metrics,
        )

    def test_step(self, batch: RecSysQuery):
        self._step(
            batch,
            "test",
            ranking_metrics=self.test_ranking_metrics,
        )

    def _step(
        self,
        batch: RecSysQuery,
        prefix: str,
        ranking_metrics: MetricCollection | None = None,
    ) -> torch.Tensor:
        scores = self(batch)
        query_ids = batch.query_id
        target_item_ids = batch.target_item_id.long()

        rank_loss = self._compute_rec_loss(
            scores=scores,
            target_item_ids=target_item_ids,
            negative_item_ids=(
                batch.negative_item_ids if prefix == "train" else None
            ),
        )
        use_gcl = (
            prefix == "train" and self.cfg.use_gcl and self.cfg.graph_mode == "lightgcn"
        )

        gcl_loss = self._compute_gcl_loss(batch) if use_gcl else rank_loss.new_zeros(())

        loss = rank_loss + self.alpha * gcl_loss

        if prefix == "train":
            self.log(
                f"{prefix}/RankLoss",
                rank_loss.detach(),
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                f"{prefix}/GclLoss",
                gcl_loss.detach(),
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                f"{prefix}/Loss",
                loss.detach(),
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        if ranking_metrics is not None:
            targets = torch.zeros_like(scores, dtype=torch.bool)
            targets.scatter_(1, target_item_ids.unsqueeze(1), True)
            preds = scores.reshape(-1).float()
            target = targets.reshape(-1)

            num_items = scores.size(1)
            indexes = query_ids.reshape(-1).long().repeat_interleave(num_items)

            if preds.numel() != target.numel() or preds.numel() != indexes.numel():
                raise RuntimeError(
                    f"preds, target and indexes must have same length: "
                    f"{preds.numel()}, {target.numel()}, {indexes.numel()}"
                )

            ranking_metrics.update(preds=preds, target=target, indexes=indexes)

        return loss

    def _compute_rec_loss(
        self,
        scores: torch.Tensor,
        target_item_ids: torch.Tensor,
        negative_item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scores.ndim != 2:
            raise RuntimeError(
                f"RecSys must return [batch, num_items] scores, got {scores.shape}."
            )

        target_item_ids = target_item_ids.reshape(-1).long()

        if target_item_ids.numel() != scores.size(0):
            raise RuntimeError(
                "target_item_ids must have one target item per batch row."
            )

        if negative_item_ids is None or negative_item_ids.size(1) == 0:
            return F.cross_entropy(scores, target_item_ids)

        candidate_ids = torch.cat(
            [target_item_ids.unsqueeze(1), negative_item_ids.long()],
            dim=1,
        )
        candidate_scores = scores.gather(1, candidate_ids)
        positive_labels = torch.zeros(
            scores.size(0),
            dtype=torch.long,
            device=scores.device,
        )
        return F.cross_entropy(candidate_scores, positive_labels)

    def _compute_gcl_loss(self, batch: RecSysQuery) -> torch.Tensor:
        if self.model.gnn is None:
            raise RuntimeError("GCL requires an active graph encoder.")

        p = self.cfg.edge_dropout

        assert isinstance(self.edge_index, torch.Tensor)
        edge_index_1, _ = dropout_edge(self.edge_index, p=p, force_undirected=True)
        edge_index_2, _ = dropout_edge(self.edge_index, p=p, force_undirected=True)

        u_emb1, i_emb1 = self.model.gnn(edge_index_1)
        u_emb2, i_emb2 = self.model.gnn(edge_index_2)

        user_ids, item_ids = self._contrastive_batch_ids(batch)
        gcl_loss = self.gcl_loss(
            u_emb1[user_ids],
            i_emb1[item_ids],
            u_emb2[user_ids],
            i_emb2[item_ids],
        )

        return gcl_loss

    @staticmethod
    def _contrastive_batch_ids(batch: RecSysQuery) -> tuple[torch.Tensor, torch.Tensor]:
        user_ids = batch.user_id.reshape(-1).long().unique()
        target_item_ids = batch.target_item_id.reshape(-1).long()

        history_item_ids = (
            batch.history_items[batch.history_valid_mask].reshape(-1).long() - 1
        )
        item_ids = torch.cat([target_item_ids, history_item_ids]).unique()

        return user_ids, item_ids

    def on_validation_epoch_start(self):
        self.val_ranking_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_ranking_metrics.compute(), sync_dist=True)

    def on_test_epoch_start(self):
        if self.test_ranking_metrics:
            self.test_ranking_metrics.reset()

    def on_test_epoch_end(self):
        if self.test_ranking_metrics:
            self.log_dict(self.test_ranking_metrics.compute(), sync_dist=True)

    def predict_step(self, batch: RecSysQuery) -> torch.Tensor:
        return self(batch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }
