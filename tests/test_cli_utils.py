from edurec.cli.utils import print_model_modules
from edurec.recsys.architecture import EDuRecConfig


def test_print_model_modules_shows_effective_status(capsys) -> None:
    cfg = EDuRecConfig(
        num_users=3,
        num_items=4,
        num_ctx_feats=0,
        num_user_dense_feats=0,
        num_item_dense_feats=0,
        num_user_text_feats=0,
        num_item_text_feats=0,
        user_cat_cardinalities=[],
        item_cat_cardinalities=[],
        has_history=False,
        graph_mode="id",
    )

    print_model_modules("TRAIN", cfg)

    assert capsys.readouterr().out == (
        "[TRAIN] Model modules: graph=ON, user_features=OFF, "
        "item_features=OFF, sequence=OFF, context=OFF\n"
    )
