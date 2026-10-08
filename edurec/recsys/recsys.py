import lightning.pytorch as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch_geometric.data import HeteroData
from torchmetrics import MetricCollection
from torchmetrics.retrieval import RetrievalNormalizedDCG

from edurec import settings
from edurec.datasets import RecSysQuery
from edurec.recsys.archs import BaseRecArch, build_model
from edurec.recsys.configs import ModelConfig, TrainConfig
from edurec.recsys.ranking import build_ranking_metrics, update_ranking_metrics


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
        self.lr = self.train_cfg.lr
        self.weight_decay = self.train_cfg.weight_decay
        self.val_topk = int(val_topk)
        self.topks = sorted(set(self.train_cfg.topks or [settings.TOP_K]))
        self.monitor = f"val/ndcg@{self.val_topk}"

        self.val_ranking_metrics = MetricCollection(
            {
                f"ndcg@{self.val_topk}": RetrievalNormalizedDCG(
                    top_k=self.val_topk,
                    empty_target_action="neg",
                    aggregation="mean",
                )
            },
            prefix="val/",
        )
        self.test_ranking_metrics = build_ranking_metrics(
            self.topks,
            "test/",
            adaptive_k=self.train_cfg.adaptive_k,
        )

        self.model: BaseRecArch = build_model(cfg)
        self.model.register_static(
            edge_index={
                edge_type: knowledge_graph[edge_type].edge_index
                for edge_type in knowledge_graph.edge_types
            },
            i_static_feats=i_static_feats,
            u_static_feats=u_static_feats,
            u_cat_feats=u_cat_feats,
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
        return self._step(batch, prefix="train")

    def validation_step(self, batch: RecSysQuery) -> None:
        self._step(
            batch,
            prefix="val",
            ranking_metrics=self.val_ranking_metrics,
            metric_topks=[self.val_topk],
        )

    def test_step(self, batch: RecSysQuery) -> None:
        self._step(
            batch,
            prefix="test",
            ranking_metrics=self.test_ranking_metrics,
            metric_topks=self.topks,
        )

    def _step(
        self,
        batch: RecSysQuery,
        prefix: str,
        ranking_metrics: MetricCollection | None = None,
        metric_topks: list[int] | None = None,
    ) -> torch.Tensor:
        negative_item_ids = batch.negative_item_ids if prefix == "train" else None

        if negative_item_ids is not None and negative_item_ids.size(1) > 0:
            # One softmax over the target plus its sampled negatives.
            candidate_item_ids = torch.cat(
                [batch.target_item_id.reshape(-1, 1).long(), negative_item_ids],
                dim=1,
            )
            scores = self(batch, candidate_item_ids=candidate_item_ids)
            labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            per_example_loss = F.cross_entropy(scores, labels, reduction="none")
        else:
            scores = self(batch)
            per_example_loss = F.cross_entropy(
                scores, batch.target_item_id.reshape(-1).long(), reduction="none"
            )

        # Weights use one training-fitted scale, not batch normalization, so
        # rating strength remains meaningful even in single-example batches.
        rec_loss = (per_example_loss * batch.sample_weight).mean()
        loss = rec_loss

        if prefix == "train":
            gcl_loss = self.model.auxiliary_loss(
                batch.history_items,
                batch.history_valid_mask,
                batch.target_item_id,
            )
            if gcl_loss is not None:
                loss = rec_loss + self.cfg.gcl_weight * gcl_loss
                self.log_dict(
                    {
                        "train/RecLoss": rec_loss.detach(),
                        "train/GCLloss": gcl_loss.detach(),
                    },
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
        else:
            self.log(
                f"{prefix}/Loss",
                rec_loss.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=scores.size(0),
            )

        if ranking_metrics is not None and metric_topks:
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

        return loss

    def predict_step(self, batch: RecSysQuery) -> torch.Tensor:
        return self(batch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if self.monitor.lower().endswith("loss") else "max",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }
