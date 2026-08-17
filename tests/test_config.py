from pathlib import Path

from edurec.recsys.configs import TrainConfig, monitor_topk, resolve_train_config


def test_resolve_prefers_cli_over_saved_over_defaults(tmp_path: Path) -> None:
    saved = tmp_path / "train.yaml"
    TrainConfig(
        epochs=10,
        lr=1e-3,
        batch_size=64,
        patience=3,
        adaptive_k=True,
    ).save(saved)

    resolved = resolve_train_config(
        cli={"epochs": 20, "batch_size": None, "adaptive_k": False},
        saved_path=saved,
        defaults=TrainConfig(epochs=5, lr=1e-4, batch_size=128, patience=7),
    )

    assert resolved.epochs == 20
    assert resolved.lr == 1e-3
    assert resolved.batch_size == 64
    assert resolved.patience == 3
    assert resolved.adaptive_k is False


def test_resolve_applies_defaults_when_no_saved_config() -> None:
    resolved = resolve_train_config(
        cli={"batch_size": 256},
        saved_path="/nonexistent/train.yaml",
        defaults=TrainConfig(epochs=42, batch_size=32),
    )

    assert resolved.epochs == 42
    assert resolved.batch_size == 256


def test_resolve_uses_global_defaults_without_cli_or_saved() -> None:
    assert resolve_train_config() == TrainConfig()


def test_resolve_ignores_none_cli_values() -> None:
    resolved = resolve_train_config(cli={"epochs": None, "lr": None})

    assert resolved.epochs == TrainConfig().epochs
    assert resolved.lr == TrainConfig().lr


def test_resolve_keeps_saved_fields_the_cli_does_not_override() -> None:
    saved = Path("/nonexistent/saved.yaml")
    resolved = resolve_train_config(
        cli={"epochs": 1},
        saved_path=saved,
        defaults=TrainConfig(weight_decay=1e-3, topks=[5, 10]),
    )

    assert resolved.weight_decay == 1e-3
    assert resolved.topks == [5, 10]


def test_monitor_topk_defaults_to_max_topks() -> None:
    cfg = TrainConfig(topks=[5, 10, 20])

    assert monitor_topk(None, cfg) == 20
    assert monitor_topk(10, cfg) == 10