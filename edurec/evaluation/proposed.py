from typing import cast

from ..datasets import ElearningDataModule
from ..recsys import GhostConfig, Ranker, train_model


def eval_proposed_model(
    dm: ElearningDataModule,
    cfg: GhostConfig,
    epochs: int,
    lr: float,
    top_ks: list[int],
    val_topk: int,
    patience: int,
    adaptive_k: bool,
) -> dict[str, float]:
    assert dm.u_static_feats is not None
    assert dm.i_static_feats is not None

    ranker = Ranker(
        cfg=cfg,
        inter_graph=dm.build_inter_graph(),
        u_static_feats=dm.u_static_feats,
        i_static_feats=dm.i_static_feats,
        lr=lr,
        val_topk=val_topk,
        top_ks=top_ks,
        adaptive_k=adaptive_k,
    )

    trainer, _ = train_model(
        model=ranker,
        dm=dm,
        top_k=val_topk,
        debug=False,
        use_logger=False,
        epochs=epochs,
        patience=patience,
        monitor=f"val/ndcg@{val_topk}",
    )

    metrics = trainer.test(ckpt_path="best", datamodule=dm, weights_only=False)[0]
    return _normalize_metric_names(cast(dict[str, float], metrics))


def _normalize_metric_names(metrics: dict[str, float]) -> dict[str, float]:
    return {
        name.removeprefix("test/"): value
        for name, value in metrics.items()
    }
