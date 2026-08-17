from pathlib import Path

import lightning.pytorch as L
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import MetricCollection
from torchmetrics.retrieval import RetrievalNormalizedDCG

from .. import settings
from ..datasets import ElearningDataModule, RecSysQuery
from ..recsys.recsys import build_ranking_metrics
from ..recsys.training import train_model
from .upgpr_graph import KnowledgeGraphData, build_knowledge_graph
from .upgpr_model import UPGPR, UPGPRConfig


class UPGPRRecSys(L.LightningModule):
    """Lightning integration for joint TransE and UPGPR policy training."""

    def __init__(
        self,
        cfg: UPGPRConfig,
        knowledge_graph: KnowledgeGraphData,
        val_topk: int = settings.TOP_K,
    ):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay
        self.val_topk = val_topk
        self.topks = cfg.topks or [settings.TOP_K]
        self.monitor = f"val/ndcg@{val_topk}"
        self.model = UPGPR(cfg, knowledge_graph)
        self.model_name = "UPGPR"

        self.val_ranking_metrics = MetricCollection(
            {
                f"ndcg@{val_topk}": RetrievalNormalizedDCG(
                    top_k=val_topk, empty_target_action="neg"
                )
            },
            prefix="val/",
        )
        self.test_ranking_metrics = build_ranking_metrics(
            self.topks, "test/", adaptive_k=cfg.adaptive_k
        )

    def forward(self, batch: RecSysQuery) -> torch.Tensor:
        # Beam search is useful for ranking/evaluation. During training the
        # policy receives its unbiased actor-critic objective separately.
        scores = self.model(batch.user_id, use_paths=not self.training)
        # PGPR evaluation removes courses already taken by the learner. Keep a
        # repeated target score intact so cross-entropy remains well-defined.
        rows, history_cols = torch.nonzero(
            batch.history_valid_mask, as_tuple=True
        )
        if rows.numel():
            history_items = batch.history_items[rows, history_cols].long() - 1
            targets = batch.target_item_id[rows].long()
            keep = history_items != targets
            scores[rows[keep], history_items[keep]] = torch.finfo(scores.dtype).min
        return scores

    def training_step(self, batch: RecSysQuery) -> torch.Tensor:
        scores = self(batch)
        rank_loss = F.cross_entropy(scores, batch.target_item_id.long())
        kg_loss = self.model.kg_loss()

        policy_loss, reward = self.model.policy_loss(batch.user_id)
        loss = rank_loss + self.cfg.kg_loss_weight * kg_loss + policy_loss
        self.log_dict(
            {
                "train/Loss": loss.detach(),
                "train/RankLoss": rank_loss.detach(),
                "train/KGLoss": kg_loss.detach(),
                "train/PolicyLoss": policy_loss.detach(),
                "train/Reward": reward.detach(),
            },
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: RecSysQuery) -> torch.Tensor:
        return self._ranking_step(batch, "val", self.val_ranking_metrics)

    def test_step(self, batch: RecSysQuery) -> torch.Tensor:
        return self._ranking_step(batch, "test", self.test_ranking_metrics)

    def _ranking_step(
        self, batch: RecSysQuery, prefix: str, metrics: MetricCollection
    ) -> torch.Tensor:
        scores = self(batch)
        target_item_ids = batch.target_item_id.reshape(-1).long()
        loss = F.cross_entropy(scores, target_item_ids)
        targets = torch.zeros_like(scores, dtype=torch.bool)
        targets.scatter_(1, target_item_ids[:, None], True)
        num_items = scores.size(1)
        indexes = batch.query_id.reshape(-1).long().repeat_interleave(num_items)
        metrics.update(
            preds=scores.reshape(-1).float(),
            target=targets.reshape(-1),
            indexes=indexes,
        )
        self.log(f"{prefix}/Loss", loss, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_ranking_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_ranking_metrics.compute(), sync_dist=True)

    def on_test_epoch_start(self) -> None:
        self.test_ranking_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_ranking_metrics.compute(), sync_dist=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }


def eval_upgpr(
    dm: ElearningDataModule,
    epochs: int,
    lr: float,
    val_topk: int,
    topks: list[int],
    patience: int,
    adaptive_k: bool,
    results_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    cfg = UPGPRConfig(
        num_users=dm.num_users,
        num_items=dm.num_items,
        lr=lr,
        topks=topks,
        adaptive_k=adaptive_k,
    )
    model = UPGPRRecSys(
        cfg,
        build_knowledge_graph(dm),
        val_topk,
    )
    trainer, _, timer = train_model(
        model=model,
        dm=dm,
        debug=False,
        epochs=epochs,
        patience=patience,
        monitor=model.monitor,
        compile=False,
        verbose=verbose,
    )
    metrics = trainer.test(ckpt_path="best", datamodule=dm, weights_only=False)[0]
    results = pd.DataFrame(
        [
            {
                "model": "UPGPR",
                **{k.removeprefix("test/"): v for k, v in metrics.items()},
                "training_time_s": timer.time_elapsed("train"),
                "inference_time_s": timer.time_elapsed("test"),
            }
        ]
    )
    if results_path is not None:
        results.to_csv(results_path / "UPGPR.csv", index=True)
    return results
