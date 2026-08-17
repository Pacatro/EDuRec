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
from ..datasets import RecSysQuery
from .architecture import EDuRec, EDuRecConfig
from .configs import TrainConfig
from .losses import InfoNCELoss, LossReduction


class RecSys(L.LightningModule):
    def __init__(
        self,
        cfg: EDuRecConfig,
        inter_graph: Data,
        u_static_feats: torch.Tensor,
        i_static_feats: torch.Tensor,
        train_cfg: TrainConfig | None = None,
        val_topk: int = settings.TOP_K,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "inter_graph",
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

        self._validate_topks()

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
                    top_k=self.val_topk,
                    empty_target_action="neg",
                    aggregation="mean",
                )
            },
            prefix="val/",
        )
        self.test_ranking_metrics = self._build_test_metrics()

        self.model = EDuRec(cfg)
        self.model_name = self.__class__.__name__

    def forward(self, batch: RecSysQuery) -> torch.Tensor:
        scores = self.model(
            u_ids=batch.user_id,
            h_ids=batch.history_items,
            h_mask=batch.history_valid_mask,
            edge_index=self.edge_index,
            u_static_feats=self.u_static_feats,
            i_static_feats=self.i_static_feats,
            context=batch.context,
        )

        if scores.ndim == 3:
            if scores.size(-1) != 1:
                raise RuntimeError("RecSys must return a single logit per candidate.")
            scores = scores.squeeze(-1)

        if scores.ndim != 2:
            raise RuntimeError(
                "RecSys must return scores with shape [batch, num_items], "
                f"got {tuple(scores.shape)}."
            )

        return scores

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
        scores = self(batch)
        target_item_ids = batch.target_item_id.reshape(-1).long()

        rank_loss = self._compute_rec_loss(
            scores=scores,
            target_item_ids=target_item_ids,
            negative_item_ids=(batch.negative_item_ids if prefix == "train" else None),
        )

        use_gcl = (
            prefix == "train" and self.cfg.use_gcl and self.cfg.graph_mode == "lightgcn"
        )
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

        if ranking_metrics is not None:
            if not metric_topks:
                raise ValueError("metric_topks must be provided with ranking_metrics.")

            self._update_ranking_metrics(
                metrics=ranking_metrics,
                scores=scores,
                batch=batch,
                max_k=max(metric_topks),
            )

            self.log_dict(
                ranking_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

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
                "target_item_ids must have exactly one target item per batch row."
            )

        self._validate_target_ids(target_item_ids, scores.size(1))

        if negative_item_ids is None or negative_item_ids.size(1) == 0:
            return F.cross_entropy(scores, target_item_ids)

        if negative_item_ids.ndim != 2 or negative_item_ids.size(0) != scores.size(0):
            raise RuntimeError(
                "negative_item_ids must have shape [batch, num_negatives]."
            )

        negative_item_ids = negative_item_ids.long()
        if negative_item_ids.numel() > 0:
            self._validate_target_ids(negative_item_ids.reshape(-1), scores.size(1))

        candidate_ids = torch.cat(
            [target_item_ids.unsqueeze(1), negative_item_ids],
            dim=1,
        )
        candidate_scores = scores.gather(1, candidate_ids)
        positive_labels = torch.zeros(
            scores.size(0),
            dtype=torch.long,
            device=scores.device,
        )
        return F.cross_entropy(candidate_scores, positive_labels)

    @torch.no_grad()
    def _update_ranking_metrics(
        self,
        metrics: MetricCollection,
        scores: torch.Tensor,
        batch: RecSysQuery,
        max_k: int,
    ) -> None:
        """Update full-catalog ranking metrics using only the top ``max_k``.

        The dataset contains one relevant target per query. Therefore, keeping
        only the highest-scoring ``max_k`` candidates is sufficient for every
        metric computed at K <= ``max_k``. A target outside this candidate set
        becomes an empty-target query and ``empty_target_action='neg'`` assigns
        the correct value of zero.
        """
        if scores.ndim != 2:
            raise ValueError(
                f"scores must have shape [batch, num_items], got {scores.shape}."
            )

        if not torch.isfinite(scores).all():
            raise FloatingPointError(
                "Model scores contain NaN or infinite values before evaluation."
            )

        batch_size, num_items = scores.shape
        if max_k <= 0 or max_k > num_items:
            raise ValueError(f"max_k must be in [1, {num_items}], got {max_k}.")

        target_item_ids = batch.target_item_id.reshape(-1).long()
        query_ids = batch.query_id.reshape(-1).long()

        if target_item_ids.numel() != batch_size:
            raise ValueError("Expected exactly one target item per query.")
        if query_ids.numel() != batch_size:
            raise ValueError("Expected exactly one query ID per query.")
        if query_ids.unique().numel() != query_ids.numel():
            raise ValueError("query_id values must be unique inside each batch.")

        self._validate_target_ids(target_item_ids, num_items)

        eval_scores = self._mask_seen_items(
            scores=scores.detach().float(),
            history_items=batch.history_items,
            history_mask=batch.history_valid_mask,
            target_item_ids=target_item_ids,
        )

        top_scores, top_item_ids = torch.topk(
            eval_scores,
            k=max_k,
            dim=1,
            largest=True,
            sorted=True,
        )

        top_targets = top_item_ids.eq(target_item_ids.unsqueeze(1))
        indexes = query_ids.unsqueeze(1).expand_as(top_item_ids)

        metrics.update(
            preds=top_scores.reshape(-1),
            target=top_targets.reshape(-1),
            indexes=indexes.reshape(-1),
        )

    @staticmethod
    def _mask_seen_items(
        scores: torch.Tensor,
        history_items: torch.Tensor,
        history_mask: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Exclude previous interactions from the evaluation candidate set.

        ``history_items`` uses zero as padding and stores real item IDs shifted
        by one. The current target is restored after masking so repeated-item
        next-event prediction remains evaluable.
        """
        masked_scores = scores.clone()
        batch_size, num_items = masked_scores.shape

        if history_items.shape != history_mask.shape:
            raise ValueError("history_items and history_mask must have the same shape.")
        if history_items.size(0) != batch_size:
            raise ValueError("History batch size must match the score batch size.")

        history_ids = history_items.long() - 1
        valid_history = (
            history_mask.bool() & history_ids.ge(0) & history_ids.lt(num_items)
        )

        batch_indexes = (
            torch.arange(batch_size, device=masked_scores.device)
            .unsqueeze(1)
            .expand_as(history_ids)
        )

        target_scores = masked_scores.gather(1, target_item_ids.unsqueeze(1))

        masked_scores[
            batch_indexes[valid_history], history_ids[valid_history]
        ] = -torch.inf

        # Restore the target in case it occurred previously in the sequence.
        masked_scores.scatter_(1, target_item_ids.unsqueeze(1), target_scores)

        return masked_scores

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

    def _build_test_metrics(self) -> MetricCollection:
        metrics = {}

        for k in self.topks:
            common = {
                "top_k": k,
                "empty_target_action": "neg",
                "aggregation": "mean",
            }

            metrics[f"precision@{k}"] = RetrievalPrecision(
                **common,
                adaptive_k=self.train_cfg.adaptive_k,
            )
            metrics[f"recall@{k}"] = RetrievalRecall(**common)
            metrics[f"ndcg@{k}"] = RetrievalNormalizedDCG(**common)
            metrics[f"hit@{k}"] = RetrievalHitRate(**common)
            metrics[f"map@{k}"] = RetrievalMAP(**common)
            metrics[f"mrr@{k}"] = RetrievalMRR(**common)

        return MetricCollection(metrics, prefix="test/")

    def _validate_topks(self) -> None:
        requested_topks = [self.val_topk, *self.topks]

        if any(k <= 0 for k in requested_topks):
            raise ValueError(
                f"All top-k values must be positive, got {requested_topks}."
            )

        invalid = [k for k in requested_topks if k > self.cfg.num_items]
        if invalid:
            raise ValueError(
                "Top-k values cannot exceed the item catalog size "
                f"({self.cfg.num_items}), got {invalid}."
            )

    @staticmethod
    def _validate_target_ids(item_ids: torch.Tensor, num_items: int) -> None:
        if item_ids.numel() == 0:
            return

        min_id = int(item_ids.min().item())
        max_id = int(item_ids.max().item())
        if min_id < 0 or max_id >= num_items:
            raise ValueError(
                "Item IDs must be in the range "
                f"[0, {num_items - 1}], got [{min_id}, {max_id}]."
            )

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
