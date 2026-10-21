from dataclasses import replace

import pytest

from edurec.evaluation.ablation import ablation_applicable, get_ablation_config
from edurec.recsys.configs import (
    ModelArch,
    ModelConfig,
    TrainConfig,
    resolve_train_config,
    save_run_config,
)


def test_run_config_round_trip_and_cli_precedence(tmp_path, model_config):
    path = tmp_path / "run.yaml"
    train = TrainConfig(epochs=7, lr=0.001, batch_size=16)
    save_run_config(path, model_config, train)
    assert ModelConfig.load(path) == model_config
    assert TrainConfig.load(path) == train
    resolved = resolve_train_config(
        {"epochs": 2, "lr": None}, path, TrainConfig(batch_size=8)
    )
    assert resolved.epochs == 2
    assert resolved.batch_size == 16
    assert resolved.lr == 0.001


def test_yaml_scientific_notation_and_old_fields(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text("train:\n  lr: 1e-5\n  obsolete_option: true\n")
    assert TrainConfig.load(path).lr == 1e-5


def test_feature_switches_disable_corresponding_modules(model_config):
    cfg = replace(
        model_config,
        use_text_features=False,
        use_user_features=False,
        use_interaction_features=False,
    )
    assert cfg.effective_item_dense_feats == 1
    assert not cfg.user_state.is_active
    assert not cfg.interaction_context.is_active
    assert replace(cfg, use_item_features=False).effective_item_dense_feats == 0


def test_ablation_preserves_base_and_skips_missing_signals(model_config):
    variant = get_ablation_config(model_config, "seq_only")
    assert variant.graph_mode == "id"
    assert not variant.use_item_features
    assert model_config.graph_mode == "kg"
    assert not ablation_applicable(model_config, "no_graph")
    assert not ablation_applicable(model_config, "no_gcl")
    assert ablation_applicable(replace(model_config, gcl_weight=0.1), "no_gcl")
    with pytest.raises(ValueError, match="Unknown ablation"):
        get_ablation_config(model_config, "unknown")


def test_sasrec_rejects_graph_contrastive_learning(model_config):
    with pytest.raises(ValueError, match="GCL"):
        replace(model_config, arch=ModelArch.SASREC_TEXT, gcl_weight=0.1)
