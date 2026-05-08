import pandas as pd

from ..datasets import ElearningDataModule, DatasetName, Phase
from ..recsys import GhostConfig, Ranker, train_model
from .. import settings


def eval_ghost(
    dataset: DatasetName,
    batch_size: int,
    val_size: float,
    test_size: float,
    use_processed_data: bool,
    remove_sparse: bool,
    min_interactions: int,
    top_ks: list[int],
) -> pd.DataFrame:
    dm = ElearningDataModule(
        dataset=dataset,
        batch_size=batch_size,
        test_ratio=test_size,
        val_ratio=val_size,
        use_processed_data=use_processed_data,
        random_state=settings.state["random_state"],
        min_interactions=min_interactions,
        remove_sparse=remove_sparse,
    )

    dm.setup(phase=Phase.RANKING)

    inter_graph = dm.create_inter_graph()
    assert dm.u_static_feats is not None and dm.i_static_feats is not None

    print(dm.train_ds[0])

    u_feats = dm.u_static_feats
    i_feats = dm.i_static_feats

    cfg = GhostConfig(
        num_users=dm.num_users,
        num_items=dm.num_items,
        num_ctx_feats=dm.train_ds.num_ctx_feats,
        num_user_dense_feats=dm.num_user_dense_feats,
        num_item_dense_feats=dm.num_item_dense_feats,
        user_cat_cardinalities=dm.user_cat_cardinalities,
        item_cat_cardinalities=dm.item_cat_cardinalities,
    )

    val_topk = settings.RANKER_TOP_K

    ranker = Ranker(
        cfg=cfg,
        inter_graph=inter_graph,
        u_static_feats=u_feats,
        i_static_feats=i_feats,
        top_ks=top_ks,
        val_topk=val_topk,
    )

    trainer, _ = train_model(
        ranker,
        dm,
        top_k=val_topk,
        debug=False,
        use_logger=False,
        epochs=settings.EPOCHS,
        patience=settings.RANKER_PATIENCE,
        monitor=f"val/NDCG@{val_topk}",
    )

    metrics = trainer.test(ckpt_path="best", datamodule=dm, weights_only=False)[0]

    rows = {}

    for key, value in metrics.items():
        if key.startswith("test/") and "@" in key:
            metric, k = key.removeprefix("test/").split("@")
            rows.setdefault(metric, {})[f"top-{k}"] = float(value)

    eval_metrics = pd.DataFrame(rows).T.sort_index(axis=1)

    return eval_metrics
