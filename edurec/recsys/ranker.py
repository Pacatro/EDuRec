import lightning.pytorch as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge
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
from ..datasets import RankingQuery
from .ghost import Ghost, GhostConfig
from .losses import InfoNCELoss


class Ranker(L.LightningModule):
    def __init__(
        self,
        cfg: GhostConfig,
        inter_graph: Data,
        u_static_feats: torch.Tensor,
        i_static_feats: torch.Tensor,
        val_topk: int = settings.RANKER_TOP_K,
        lr: float = settings.RANKER_LR,
        weight_decay: float = settings.RANKER_WEIGHT_DECAY,
        top_ks: list[int] = [10, 20, 50],
        alpha: float = settings.LOSS_ALPHA,
        adaptive_k: bool = settings.ADAPTIVE_K,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["inter_graph", "u_static_feats", "i_static_feats"]
        )
        self.cfg = cfg
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.val_topk = val_topk
        self.top_ks = top_ks if top_ks else [settings.RANKER_TOP_K]
        self.monitor = f"val/NDCG@{val_topk}"

        self.register_buffer("edge_index", inter_graph.edge_index)
        self.register_buffer("u_static_feats", u_static_feats, persistent=False)
        self.register_buffer("i_static_feats", i_static_feats, persistent=False)

        self.gcl_loss = InfoNCELoss(tau=cfg.gnn.tau, reduction=cfg.gnn.loss_reduc)

        self.val_ranking_metrics = MetricCollection(
            {f"NDCG@{self.val_topk}": RetrievalNormalizedDCG(top_k=self.val_topk)},
            prefix="val/",
        )

        if self.top_ks:
            metrics = {
                "Precision": (RetrievalPrecision, {"adaptive_k": adaptive_k}),
                "Recall": (RetrievalRecall, {}),
                "NDCG": (RetrievalNormalizedDCG, {}),
                "Hit": (RetrievalHitRate, {}),
                "MAP": (RetrievalMAP, {}),
                "MRR": (RetrievalMRR, {}),
            }

            self.test_ranking_metrics = MetricCollection(
                {
                    f"{name}@{k}": cls(top_k=k, **kwargs)
                    for k in top_ks
                    for name, (cls, kwargs) in metrics.items()
                },
                prefix="test/",
            )

        self.model = Ghost(cfg)
        self.model_name = self.__class__.__name__

    def forward(self, batch: RankingQuery) -> torch.Tensor:
        scores = self.model(
            u_ids=batch.user_id,
            h_ids=batch.history_items,
            h_ctx=batch.history_ctx,
            h_mask=batch.history_valid_mask,
            c_ids=batch.candidate_ids,
            edge_index=self.edge_index,
            u_static_feats=self.u_static_feats,
            i_static_feats=self.i_static_feats,
        )

        if scores.ndim == 3:
            if scores.size(-1) != 1:
                raise RuntimeError("Ranker must return a single logit per candidate.")
            return scores.squeeze(-1)

        return scores

    def training_step(self, batch: RankingQuery) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: RankingQuery):
        self._step(
            batch,
            "val",
            ranking_metrics=self.val_ranking_metrics,
        )

    def test_step(self, batch: RankingQuery):
        self._step(
            batch,
            "test",
            ranking_metrics=self.test_ranking_metrics,
        )

    def _step(
        self,
        batch: RankingQuery,
        prefix: str,
        ranking_metrics: MetricCollection | None = None,
    ) -> torch.Tensor:
        scores = self(batch)
        targets = batch.candidate_labels
        query_ids = batch.query_id
        positive_position = batch.positive_position

        if scores.numel() == 0 or targets.numel() == 0:
            return scores.new_zeros(())

        rank_loss = self._compute_rank_loss(scores, targets, positive_position)
        gcl_loss = (
            self._compute_gcl_loss() if prefix == "train" else rank_loss.new_zeros(())
        )

        loss = rank_loss + self.alpha * gcl_loss

        self.log(
            f"{prefix}/RankLoss",
            rank_loss,
            prog_bar=(prefix == "train"),
            logger=(prefix == "train"),
        )
        self.log(
            f"{prefix}/GclLoss",
            gcl_loss,
            prog_bar=(prefix == "train"),
            logger=(prefix == "train"),
        )
        self.log(
            f"{prefix}/Loss",
            loss,
            on_step=(prefix == "train"),
            prog_bar=True,
            logger=True,
        )

        if ranking_metrics is not None:
            if scores.ndim == 3 and scores.size(-1) == 1:
                scores = scores.squeeze(-1)

            if targets.ndim == 1:
                scores = scores.reshape(-1, 1)
                targets = targets.reshape(-1, 1)

            if scores.shape != targets.shape:
                raise RuntimeError(
                    f"scores and targets must have the same shape, "
                    f"got {scores.shape} and {targets.shape}"
                )

            preds = scores.reshape(-1).float()
            target = targets.reshape(-1) > 0

            num_candidates = targets.size(1)
            indexes = query_ids.reshape(-1).long().repeat_interleave(num_candidates)

            if preds.numel() != target.numel() or preds.numel() != indexes.numel():
                raise RuntimeError(
                    f"preds, target and indexes must have same length: "
                    f"{preds.numel()}, {target.numel()}, {indexes.numel()}"
                )

            ranking_metrics.update(preds=preds, target=target, indexes=indexes)

        return loss

    def _compute_rank_loss(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        positive_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scores.ndim == 2 and targets.ndim == 2:
            if positive_position is None:
                positive_position = targets.argmax(dim=1)

            return F.cross_entropy(scores, positive_position.long())

        scores = scores.reshape_as(targets)
        return F.binary_cross_entropy_with_logits(scores, targets.float())

    def _compute_gcl_loss(self) -> torch.Tensor:
        p = self.cfg.edge_dropout

        assert isinstance(self.edge_index, torch.Tensor)
        edge_index_1, _ = dropout_edge(self.edge_index, p=p, force_undirected=True)
        edge_index_2, _ = dropout_edge(self.edge_index, p=p, force_undirected=True)

        u_emb1, i_emb1 = self.model.gnn(
            edge_index_1,
            self.u_static_feats,
            self.i_static_feats,
        )

        u_emb2, i_emb2 = self.model.gnn(
            edge_index_2,
            self.u_static_feats,
            self.i_static_feats,
        )

        gcl_loss = self.gcl_loss(u_emb1, i_emb1, u_emb2, i_emb2)

        return gcl_loss

    def on_validation_epoch_start(self):
        self.val_ranking_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_ranking_metrics.compute())

    def on_test_epoch_start(self):
        if self.test_ranking_metrics:
            self.test_ranking_metrics.reset()

    def on_test_epoch_end(self):
        if self.test_ranking_metrics:
            self.log_dict(self.test_ranking_metrics.compute())

    def predict_step(self, batch: RankingQuery) -> torch.Tensor:
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
