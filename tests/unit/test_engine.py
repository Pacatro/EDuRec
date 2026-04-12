import torch
from torch.utils.data import DataLoader

from edurec.datasets import DatasetName, ElearningDataModule
from edurec.recsys.model import GhostConfig
from edurec.recsys.engine import RecSys


def test_recsys_step_and_metrics():
    dm = ElearningDataModule(
        DatasetName.MARS, batch_size=4, test_ratio=0.2, val_ratio=0.2
    )
    dm.setup()

    batch = next(
        iter(DataLoader(dm.val_ds, batch_size=4, shuffle=False, num_workers=0))
    )

    cfg = GhostConfig(
        num_users=dm.num_users,
        num_items=dm.num_items,
        num_ctx_feats=dm.train_ds.num_ctx_feats,
        num_user_dense_feats=dm.num_user_dense_feats,
        num_item_dense_feats=dm.num_item_dense_feats,
        user_cat_cardinalities=dm.user_cat_cardinalities,
        item_cat_cardinalities=dm.item_cat_cardinalities,
    )

    assert dm.u_static_feats is not None
    assert dm.i_static_feats is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    recsys = RecSys(
        cfg=cfg,
        inter_graph=dm.create_inter_graph(),
        u_static_feats=dm.u_static_feats,
        i_static_feats=dm.i_static_feats,
    ).to(device)

    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

    loss = recsys._step(batch, "val", ranking_metrics=recsys.val_ranking_metrics)
    metrics = recsys.val_ranking_metrics.compute()

    assert loss.item() > 0
    assert f"val/Ndcg@{recsys.top_k}" in metrics
