from dataclasses import replace
from functools import cached_property

import lightning.pytorch as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch_geometric.data import HeteroData
from torchmetrics import MetricCollection

from edurec import settings
from edurec.datasets import DataAdapter, RecSysDataset, RecSysQuery, graphs
from edurec.recsys.archs import build_model
from edurec.recsys.configs import ModelConfig, TrainConfig, monitor_mode
from edurec.recsys.ranking import build_ranking_metrics, update_ranking_metrics


class EdurecDataAdapter(DataAdapter[RecSysQuery, RecSysQuery]):
    """Canonical queries plus the knowledge graph used by the proposed model."""

    @cached_property
    def knowledge_graph(self) -> HeteroData:
        """Static item knowledge graph, built once and treated as read-only."""
        return graphs.build_knowledge_graph(
            self.data.item_features,
            self.data.i_static_feats,
            self.data.schema["items"],
            self.data.graph_node_features,
        )

    def dataset(self, split: str) -> RecSysDataset:
        return self.get_queries(split)

    def evaluation_input(self, query: RecSysQuery) -> RecSysQuery:
        return query


class RecSys(L.LightningModule):
    """LightningModule that trains and evaluates an EDuRec architecture.

    It owns the data buffers, losses, ranking metrics, optimizer and the
    train/val/test loops. The architecture selected by ``cfg.arch`` is built
    internally so checkpoints can be reloaded from the saved config alone.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        knowledge_graph: HeteroData,
        i_static_feats: torch.Tensor,
        u_static_feats: torch.Tensor,
        u_cat_feats: torch.Tensor,
        train_cfg: TrainConfig | None = None,
        val_topk: int = settings.TOP_K,
    ) -> None:
        super().__init__()
        graph_node_feats = {
            node: knowledge_graph[node].x
            for node in knowledge_graph.node_types
            if node != "item" and "x" in knowledge_graph[node]
        }
        # Node feature widths are derived from the graph at runtime. Copy the
        # config instead of mutating the object owned by the caller.
        cfg = replace(
            cfg,
            kg_node_feat_dims={
                node: features.size(1) for node, features in graph_node_feats.items()
            },
        )
        self.save_hyperparameters(
            ignore=[
                "knowledge_graph",
                "i_static_feats",
                "u_static_feats",
                "u_cat_feats",
            ]
        )

        self.cfg = cfg
        self.train_cfg = train_cfg or TrainConfig()
        self.val_topk = int(val_topk)
        self.topks = sorted(set(self.train_cfg.topks or [settings.TOP_K]))
        self.monitor = f"val/ndcg@{self.val_topk}"

        self.val_ranking_metrics = build_ranking_metrics([self.val_topk], "val")
        self.test_ranking_metrics = build_ranking_metrics(
            self.topks,
            "test",
            adaptive_k=self.train_cfg.adaptive_k,
        )

        self.model = build_model(cfg)
        self.model.register_static(
            edge_index={
                edge_type: knowledge_graph[edge_type].edge_index
                for edge_type in knowledge_graph.edge_types
            },
            i_static_feats=i_static_feats,
            u_static_feats=u_static_feats,
            u_cat_feats=u_cat_feats,
            graph_node_feats=graph_node_feats,
        )

    def forward(
        self,
        batch: RecSysQuery,
        candidate_item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score a batch, optionally restricting scoring to candidate items.

        Candidate scoring ``[batch, 1 + num_negatives]`` avoids materializing
        scores for the full catalog during training. When ``candidate_item_ids``
        is ``None`` the full ``[batch, num_items]`` score matrix is returned.
        """
        return self.model(
            h_ids=batch.history_items,
            h_mask=batch.history_valid_mask,
            user_ids=batch.user_id,
            h_dense=batch.history_dense_features,
            h_cat=batch.history_cat_features,
            h_delta=batch.history_delta_times,
            candidate_item_ids=candidate_item_ids,
        )

    def training_step(self, batch: RecSysQuery) -> torch.Tensor:
        _, rec_loss = self._scores_and_loss(batch, use_negatives=True)
        gcl_loss = self.model.auxiliary_loss(
            batch.history_items,
            batch.history_valid_mask,
            batch.target_item_id,
        )

        loss = rec_loss
        logged = {"train/RecLoss": rec_loss.detach()}
        if gcl_loss is not None:
            loss = rec_loss + self.cfg.gcl_weight * gcl_loss
            logged["train/GCLloss"] = gcl_loss.detach()

        self.log_dict(
            logged,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/Loss",
            loss.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: RecSysQuery) -> None:
        self._eval_step(batch, "val", self.val_ranking_metrics, [self.val_topk])

    def test_step(self, batch: RecSysQuery) -> None:
        self._eval_step(batch, "test", self.test_ranking_metrics, self.topks)

    def _scores_and_loss(
        self,
        batch: RecSysQuery,
        use_negatives: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a batch and return its recommendation loss.

        With sampled negatives, the cross-entropy is a single softmax over the
        target plus its negatives. Otherwise it runs over the full catalog.
        """
        negatives = batch.negative_item_ids if use_negatives else None
        if negatives is not None and negatives.numel() > 0:
            candidate_item_ids = torch.cat(
                [batch.target_item_id.reshape(-1, 1).long(), negatives],
                dim=1,
            )
            scores = self(batch, candidate_item_ids=candidate_item_ids)
            labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        else:
            scores = self(batch)
            labels = batch.target_item_id.reshape(-1).long()

        return scores, F.cross_entropy(scores, labels)

    def _eval_step(
        self,
        batch: RecSysQuery,
        prefix: str,
        ranking_metrics: MetricCollection,
        metric_topks: list[int],
    ) -> None:
        scores, loss = self._scores_and_loss(batch, use_negatives=False)

        self.log(
            f"{prefix}/Loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=scores.size(0),
        )

        update_ranking_metrics(
            metrics=ranking_metrics,
            scores=scores,
            target_item_ids=batch.target_item_id,
            query_ids=batch.query_id,
            history_items=batch.history_items,
            history_mask=batch.history_valid_mask,
            max_k=max(metric_topks),
        )

        self.log_dict(
            ranking_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=scores.size(0),
        )

    def predict_step(self, batch: RecSysQuery) -> torch.Tensor:
        return self(batch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=monitor_mode(self.monitor),
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }
