import pandas as pd
import pytest

from edurec import settings
from edurec.datasets.sources.registry import RawData
from edurec.recsys.configs import ModelConfig


@pytest.fixture(autouse=True)
def isolate_data_folders(tmp_path, monkeypatch):
    """Keep every test away from the real data/ folders."""
    monkeypatch.setattr(settings, "PROCESSED_FOLDER", str(tmp_path / "processed"))
    monkeypatch.setattr(settings, "ATOMICFILES_FOLDER", str(tmp_path / "atomicfiles"))


@pytest.fixture
def make_raw():
    def build(histories, concepts):
        events = pd.DataFrame(
            [
                (user, item, time, 1.0)
                for user, items in histories.items()
                for time, item in enumerate(items)
            ],
            columns=[
                settings.USER_COL,
                settings.ITEM_COL,
                settings.TIME_COL,
                settings.RATING_COL,
            ],
        )
        return RawData(
            interactions=events,
            user_features=pd.DataFrame({settings.USER_COL: list(histories)}),
            item_features=pd.DataFrame(
                {settings.ITEM_COL: list(concepts), "concept": list(concepts.values())}
            ),
            schema={"items": {"graph_list": ["concept"]}},
        )

    return build


@pytest.fixture
def raw_communities(make_raw):
    histories = {user: [0, 1, 2, 3] for user in range(4)}
    histories.update({user: [4, 5, 6, 7] for user in range(4, 7)})
    return make_raw(histories, {item: ["a" if item < 4 else "b"] for item in range(8)})


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        num_users=2,
        num_items=4,
        num_item_dense_feats=2,
        num_item_text_feats=1,
        emb_dim=4,
        rnn_hidden_dim=4,
        rnn_layers=1,
        hidden_dims=[4],
        transformer_hidden_dim=4,
        transformer_heads=2,
        transformer_layers=1,
        dropout=0.0,
        rnn_dropout=0.0,
        gnn_dropout=0.0,
    )
