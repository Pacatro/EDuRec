from edurec import settings

from ..datasets import ElearningDataModule
from ..recsys import EDuRecConfig, RecSys, train_model


def eval_model(
    dm: ElearningDataModule,
    cfg: EDuRecConfig,
    epochs: int,
    val_topk: int,
    patience: int,
    compile: bool = settings.COMPILE_MODEL,
    verbose: bool = False,
) -> dict[str, float | str]:
    recsys = RecSys(
        cfg=cfg,
        inter_graph=dm.build_inter_graph(),
        u_static_feats=dm.u_static_feats,
        i_static_feats=dm.i_static_feats,
        val_topk=val_topk,
    )

    trainer, _ = train_model(
        model=recsys,
        dm=dm,
        debug=False,
        use_logger=False,
        epochs=epochs,
        patience=patience,
        monitor=f"val/ndcg@{val_topk}",
        compile=compile,
        verbose=verbose,
    )

    metrics = trainer.test(ckpt_path="best", datamodule=dm, weights_only=False)[0]
    metrics = {name.removeprefix("test/"): value for name, value in metrics.items()}
    return {"model": "EDuRec", **metrics}
