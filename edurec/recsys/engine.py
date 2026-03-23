import lightning.pytorch as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge
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
from .arquitecture import Ghost, GhostConfig, InfoNCELoss


class RecSys(L.LightningModule):
    def __init__(
        self,
        cfg: GhostConfig,
        inter_graph: Data,
        u_static: torch.Tensor,
        i_static: torch.Tensor,
        lr: float = config.LR,
        weight_decay: float = config.WEIGHT_DECAY,
        top_k: int = config.TOP_K,
        alpha: float = config.ALPHA,
        adaptive_k: bool = config.ADAPTIVE_K,
        monitor: str = config.MONITOR,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.cfg = cfg
        self.inter_graph = inter_graph.to(
            "cuda" if config.state["device"] == "auto" else config.state["device"]
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.monitor = monitor
        self.alpha = alpha
        self.top_k = top_k

        self.register_buffer("u_static", u_static)
        self.register_buffer("i_static", i_static)

        self.ranker_loss = nn.BCEWithLogitsLoss()
        self.gcl_loss = InfoNCELoss(
            tau=cfg.gcl.tau,
            max_samples_u=cfg.gcl.max_samples_u,
            max_samples_i=cfg.gcl.max_samples_i,
            reduction=cfg.gcl.loss_reduc,
        )

        self.val_ranking_metrics = MetricCollection(
            {f"Ndcg@{top_k}": RetrievalNormalizedDCG(top_k=top_k)}, prefix="val/"
        )

        self.test_ranking_metrics = MetricCollection(
            {
                f"Precision@{top_k}": RetrievalPrecision(top_k=top_k, adaptive_k=adaptive_k),
                f"Recall@{top_k}": RetrievalRecall(top_k=top_k),
                f"Ndcg@{top_k}": RetrievalNormalizedDCG(top_k=top_k),
                f"Hit@{top_k}": RetrievalHitRate(top_k=top_k),
                f"Map@{top_k}": RetrievalMAP(top_k=top_k),
                f"Mrr@{top_k}": RetrievalMRR(top_k=top_k),
                f"AUROC@{top_k}": RetrievalAUROC(top_k=top_k),
                f"F1@{top_k}": RetrievalFBetaScore(
                    top_k=top_k, beta=1.0, adaptive_k=adaptive_k
                ),
            },
            prefix="test/",
        )

        self.model = Ghost(cfg)
        self.model_name = self.model.__class__.__name__

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.model(
            u_id=batch["user_id"],
            h_ids=batch["history_items"],
            h_ctx=batch["history_ctx"],
            c_ids=batch["candidates"],
            inter_graph=self.inter_graph,
            u_static_global=self.u_static,
            i_static_global=self.i_static,
            hist_mask=batch["mask"],
        )

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

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        prefix: str,
        ranking_metrics: MetricCollection | None = None,
    ) -> torch.Tensor:
        scores = self(batch)
        targets = batch["target"].float()

        rank_loss = self._compute_rank_loss(scores, targets)
        gcl_loss = self._compute_gcl_loss() if prefix == "train" else rank_loss.new_zeros(())
        loss = self._compute_loss(rank_loss, gcl_loss)

        self.log(
            f"{prefix}/RankLoss",
            rank_loss,
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=(prefix != "train"),
            batch_size=targets.size(0),
        )
        self.log(
            f"{prefix}/GclLoss",
            gcl_loss,
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=(prefix != "train"),
            batch_size=targets.size(0),
        )
        self.log(
            f"{prefix}/Loss",
            loss,
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=targets.size(0),
        )

        if ranking_metrics is not None:
            preds = scores.flatten()
            target = targets.flatten().long()

            num_candidates = targets.size(1) if targets.ndim > 1 else 1

            if targets.ndim == 1:
                indexes = batch["query_id"].reshape(-1).long()
            else:
                num_candidates = targets.size(1)
                indexes = (
                    batch["query_id"]
                    .reshape(-1)
                    .long()
                    .repeat_interleave(num_candidates)
                )

            ranking_metrics.update(preds=preds, target=target, indexes=indexes)

        return loss

    def _compute_rank_loss(
        self, scores: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if scores.ndim == 2 and targets.ndim == 2:
            pos_idx = targets.argmax(dim=1)  # [B]
            return self.ranker_loss(scores, pos_idx)

        return nn.functional.binary_cross_entropy_with_logits(scores, targets.float())

    def _compute_loss(
        self, rank_loss: torch.Tensor, gcl_loss: torch.Tensor
    ) -> torch.Tensor:
        return rank_loss + self.alpha * gcl_loss

    def _compute_gcl_loss(self) -> torch.Tensor:
        assert self.inter_graph.edge_index is not None
        edge_index_1 = self._create_graph_view(self.inter_graph.edge_index)
        edge_index_2 = self._create_graph_view(self.inter_graph.edge_index)

        z1 = self.model.gcl.encode(self.inter_graph, edge_index_1)
        z2 = self.model.gcl.encode(self.inter_graph, edge_index_2)

        num_users = self.inter_graph.u_x.shape[0]
        num_items = self.inter_graph.i_x.shape[0]

        return self.gcl_loss(z1, z2, num_users, num_items)

    def _create_graph_view(self, edge_index: torch.Tensor) -> torch.Tensor:
        if self.cfg.edge_dropout <= 0:
            return edge_index

        num_edges = edge_index.size(1)

        # The interaction graph is stored as [u->i | i->u], preserving one
        # reverse edge for each interaction in the same position.
        if num_edges % 2 != 0:
            edge_index_view, _ = dropout_edge(
                edge_index, p=self.cfg.edge_dropout, force_undirected=True
            )
            return edge_index_view

        half_edges = num_edges // 2
        if half_edges == 0:
            return edge_index

        keep_mask = torch.rand(half_edges, device=edge_index.device) >= self.cfg.edge_dropout
        if not torch.any(keep_mask):
            keep_mask[torch.randint(half_edges, (1,), device=edge_index.device)] = True

        edge_index_u2i = edge_index[:, :half_edges][:, keep_mask]
        edge_index_i2u = edge_index[:, half_edges:][:, keep_mask]
        return torch.cat([edge_index_u2i, edge_index_i2u], dim=1).contiguous()

    def on_validation_epoch_start(self) -> None:
        self.val_ranking_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_ranking_metrics.compute())

    def on_test_epoch_start(self) -> None:
        self.test_ranking_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_ranking_metrics.compute())

    def predict_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
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
