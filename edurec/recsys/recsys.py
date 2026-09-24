import lightning.pytorch as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_edge
from torchmetrics import MetricCollection
from torchmetrics.retrieval import RetrievalNormalizedDCG

from .. import settings
from ..datasets import RecSysQuery
from .archs import BaseRecArch, build_model
from .configs import ModelConfig, TrainConfig
from .losses import InfoNCELoss, LossReduction
from .ranking import build_ranking_metrics, update_ranking_metrics


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
        u_static_feats: torch.Tensor,
        i_static_feats: torch.Tensor,
        train_cfg: TrainConfig | None = None,
        val_topk: int = settings.TOP_K,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "knowledge_graph",
                "u_static_feats",
                "i_static_feats",
            ]
        )

        self.cfg = cfg
        self.train_cfg = train_cfg or TrainConfig()
        self.lr = self.train_cfg.lr
        self.weight_decay = self.train_cfg.weight_decay
        self.alpha = self.train_cfg.alpha
        self.val_topk = int(val_topk)
        self.topks = sorted(set(self.train_cfg.topks or [settings.TOP_K]))
        self.monitor = f"val/ndcg@{self.val_topk}"

        self._edge_types = list(knowledge_graph.edge_types)
        for idx, edge_type in enumerate(self._edge_types):
            self.register_buffer(
                f"edge_index_{idx}",
                knowledge_graph[edge_type].edge_index,
                persistent=False,
            )
        self.register_buffer("u_static_feats", u_static_feats, persistent=False)
        self.register_buffer("i_static_feats", i_static_feats, persistent=False)

        self.gcl_loss = InfoNCELoss(
            tau=cfg.temperature,
            reduction=LossReduction(cfg.loss_reduction),
        )

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
            u_ids=batch.user_id,
            h_ids=batch.history_items,
            h_mask=batch.history_valid_mask,
            edge_index=self._edge_index_dict(),
            u_static_feats=self.get_buffer("u_static_feats"),
            i_static_feats=self.get_buffer("i_static_feats"),
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
            rank_loss = F.cross_entropy(scores, labels)
        else:
            scores = self(batch)
            rank_loss = F.cross_entropy(scores, batch.target_item_id.reshape(-1).long())

        use_gcl = prefix == "train" and self.cfg.use_gcl and self.cfg.graph_mode == "kg"
        use_gcl = False
        gcl_loss = self._compute_gcl_loss(batch) if use_gcl else rank_loss.new_zeros(())
        loss = rank_loss + self.alpha * gcl_loss

        if prefix == "train":
            self.log(
                "train/RankLoss",
                rank_loss.detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "train/GclLoss",
                gcl_loss.detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
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
                rank_loss.detach(),
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

    def _edge_index_dict(self) -> dict[tuple[str, str, str], torch.Tensor]:
        return {
            edge_type: self.get_buffer(f"edge_index_{idx}")
            for idx, edge_type in enumerate(self._edge_types)
        }

    def _compute_gcl_loss(self, batch: RecSysQuery) -> torch.Tensor:
        p = self.cfg.edge_dropout
        edge_index = self._edge_index_dict()

        edge_index_1 = {
            edge_type: dropout_edge(edges, p=p)[0]
            for edge_type, edges in edge_index.items()
        }
        edge_index_2 = {
            edge_type: dropout_edge(edges, p=p)[0]
            for edge_type, edges in edge_index.items()
        }

        user_feats = self.get_buffer("u_static_feats")[
            :, : self.cfg.effective_user_dense_feats
        ]
        item_feats = self.get_buffer("i_static_feats")[
            :, : self.cfg.effective_item_dense_feats
        ]

        u_emb1, i_emb1 = self.model.kg(edge_index_1, user_feats, item_feats)
        u_emb2, i_emb2 = self.model.kg(edge_index_2, user_feats, item_feats)

        user_ids, item_ids = self._contrastive_batch_ids(batch)
        return self.gcl_loss(
            u_emb1[user_ids],
            i_emb1[item_ids],
            u_emb2[user_ids],
            i_emb2[item_ids],
        )

    @staticmethod
    def _contrastive_batch_ids(batch: RecSysQuery) -> tuple[torch.Tensor, torch.Tensor]:
        user_ids = batch.user_id.reshape(-1).long().unique()
        target_item_ids = batch.target_item_id.reshape(-1).long()

        history_item_ids = (
            batch.history_items[batch.history_valid_mask].reshape(-1).long() - 1
        )
        item_ids = torch.cat([target_item_ids, history_item_ids]).unique()

        return user_ids, item_ids

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
            mode="max",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }
