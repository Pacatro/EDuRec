import pytest

from edurec.recsys.configs import ModelConfig


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
