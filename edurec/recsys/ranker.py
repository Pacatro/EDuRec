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


from .. import config
from .ghost import Ghost, GhostConfig
from .losses import InfoNCELoss
from ..datasets import RankerBatch


class Ranker(L.LightningModule):
    def __init__(
        self,
        cfg: GhostConfig,
        inter_graph: Data,
        u_static_feats: torch.Tensor,
        i_static_feats: torch.Tensor,
        lr: float = config.RANKER_LR,
        weight_decay: float = config.RANKER_WEIGHT_DECAY,
        top_k: int = config.RANKER_TOP_K,
        alpha: float = config.ALPHA,
        adaptive_k: bool = config.ADAPTIVE_K,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["inter_graph", "u_static_feats", "i_static_feats"]
        )
        self.cfg = cfg
        self.inter_graph = inter_graph.to(self._resolve_graph_device())
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.top_k = top_k
        self.monitor = f"val/NDCG@{top_k}"

        self.register_buffer("u_static_feats", u_static_feats, persistent=False)
        self.register_buffer("i_static_feats", i_static_feats, persistent=False)

        self.gcl_loss = InfoNCELoss(tau=cfg.gnn.tau, reduction=cfg.gnn.loss_reduc)

        self.val_ranking_metrics = MetricCollection(
            {f"NDCG@{top_k}": RetrievalNormalizedDCG(top_k=top_k)}, prefix="val/"
        )

        self.test_ranking_metrics = MetricCollection(
            {
                f"Precision@{top_k}": RetrievalPrecision(
                    top_k=top_k, adaptive_k=adaptive_k
                ),
                f"Recall@{top_k}": RetrievalRecall(top_k=top_k),
                f"NDCG@{top_k}": RetrievalNormalizedDCG(top_k=top_k),
                f"Hit@{top_k}": RetrievalHitRate(top_k=top_k),
                f"MAP@{top_k}": RetrievalMAP(top_k=top_k),
                f"MRR@{top_k}": RetrievalMRR(top_k=top_k),
            },
            prefix="test/",
        )

        self.model = Ghost(cfg)
        self.model_name = self.__class__.__name__

    def on_load_checkpoint(self, checkpoint: dict):
        state_dict = checkpoint.get("state_dict")
        if isinstance(state_dict, dict):
            state_dict.pop("model.edge_index", None)

    def _resolve_graph_device(self) -> str:
        if config.state["device"] != "auto":
            return config.state["device"]

        return "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, batch: RankerBatch) -> torch.Tensor:
        scores = self.model(
            u_ids=batch.user_id,
            h_ids=batch.history_items,
            h_ctx=batch.history_ctx,
            h_mask=batch.history_valid_mask,
            c_ids=batch.candidate_ids,
            inter_graph=self.inter_graph,
            u_static_feats=self.u_static_feats,
            i_static_feats=self.i_static_feats,
        )

        if scores.ndim == 3:
            if scores.size(-1) != 1:
                raise RuntimeError("Ranker must return a single logit per candidate.")
            return scores.squeeze(-1)

        return scores

    def training_step(self, batch: RankerBatch) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: RankerBatch):
        self._step(
            batch,
            "val",
            ranking_metrics=self.val_ranking_metrics,
        )

    def test_step(self, batch: RankerBatch):
        self._step(
            batch,
            "test",
            ranking_metrics=self.test_ranking_metrics,
        )

    def _step(
        self,
        batch: RankerBatch,
        prefix: str,
        ranking_metrics: MetricCollection | None = None,
    ) -> torch.Tensor:
        scores = self(batch)
        targets = batch.candidate_labels.float()

        scores, targets, positive_position, query_ids = self._select_valid_queries(
            scores=scores,
            targets=targets,
            positive_position=batch.positive_position,
            query_ids=batch.query_id,
        )

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
            preds = scores.flatten()
            target = targets.flatten().long()

            num_candidates = targets.size(1) if targets.ndim > 1 else 1

            if targets.ndim == 1:
                indexes = query_ids.reshape(-1).long()
            else:
                num_candidates = targets.size(1)
                indexes = query_ids.reshape(-1).long().repeat_interleave(num_candidates)

            ranking_metrics.update(preds=preds, target=target, indexes=indexes)

        return loss

    def _select_valid_queries(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        positive_position: torch.Tensor | None,
        query_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if scores.ndim == 1 or targets.ndim == 1:
            return scores, targets, positive_position, query_ids

        valid_mask = targets.sum(dim=1) == 1
        scores = scores[valid_mask]
        targets = targets[valid_mask]
        query_ids = query_ids[valid_mask]

        if positive_position is not None:
            positive_position = positive_position[valid_mask]

        return scores, targets, positive_position, query_ids

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
        assert self.inter_graph.edge_index is not None
        edge_index = self.inter_graph.edge_index
        num_nodes = self.inter_graph.num_nodes
        p = self.cfg.edge_dropout

        edge_index_1, _ = dropout_edge(edge_index, p=p, force_undirected=True)
        edge_index_2, _ = dropout_edge(edge_index, p=p, force_undirected=True)

        graph_view_1 = Data(edge_index=edge_index_1, num_nodes=num_nodes)
        graph_view_2 = Data(edge_index=edge_index_2, num_nodes=num_nodes)

        u_emb1, i_emb1 = self.model.gnn(
            graph_view_1,
            self.u_static_feats,
            self.i_static_feats,
        )

        u_emb2, i_emb2 = self.model.gnn(
            graph_view_2,
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
        self.test_ranking_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_ranking_metrics.compute())

    def predict_step(self, batch: RankerBatch) -> torch.Tensor:
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
