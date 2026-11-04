import torch

from edurec.recsys.archs.modules import (
    GraphEncoder,
    Scorer,
    TabularEncoder,
    TransformerSeqEncoder,
    UserStateEncoder,
)


def test_tabular_encoder_matches_between_profile_and_event_shapes():
    torch.manual_seed(0)
    encoder = TabularEncoder(emb_dim=4, dense_dim=2, cat_cardinalities=[3, 4])
    dense = torch.randn(2, 3, 2)
    cat = torch.stack(
        [torch.randint(-1, 3, (2, 3)), torch.randint(-1, 4, (2, 3))], dim=-1
    )

    sequence = encoder(dense, cat)
    row = encoder(dense[:, 1], cat[:, 1])

    assert sequence.shape == (2, 3, 4)
    torch.testing.assert_close(sequence[:, 1], row)


def test_user_state_encoder_falls_back_to_learned_state():
    encoder = UserStateEncoder(emb_dim=4, hidden_dim=5, num_layers=2)
    state = encoder(
        torch.empty(3, 0, dtype=torch.long),
        torch.empty(3, 0),
        torch.tensor([0, 1, 2]),
    )

    assert state.shape == (2, 3, 5)
    expected = encoder.learned_state.unsqueeze(1).expand(-1, 3, -1)
    torch.testing.assert_close(state, expected)


def test_transformer_encoder_ignores_padding_and_empty_histories():
    torch.manual_seed(0)
    encoder = TransformerSeqEncoder(
        emb_dim=4,
        hidden_dim=4,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_history_len=6,
        use_attention_pooling=True,
    )
    encoder.eval()
    history = torch.randn(2, 4, 4)
    mask = torch.tensor([[True, True, True, False], [True, False, False, False]])

    with torch.no_grad():
        base = encoder(history, mask)
        shifted = history.clone()
        shifted[0, 3] = 100.0
        shifted[1, 1:] = -100.0
        padded = encoder(shifted, mask)
        empty = encoder(torch.randn(1, 3, 4), torch.zeros(1, 3, dtype=torch.bool))

    torch.testing.assert_close(base, padded)
    torch.testing.assert_close(empty, torch.zeros(1, 4))


def test_graph_encoder_id_mode_returns_projected_item_table():
    encoder = GraphEncoder(num_items=5, emb_dim=4, item_feat_dim=2, graph_mode="id")
    feats = torch.randn(5, 2)

    expected = encoder.input_norm(encoder.item_emb.weight + encoder.item_proj(feats))
    torch.testing.assert_close(encoder({}, feats), expected)


def test_scorer_chunked_full_catalog_matches_candidate_scores():
    scorer = Scorer(emb_dim=4, hidden_dims=[4], dropout=0.0, max_pairs_per_chunk=4)
    scorer.eval()
    user = torch.randn(2, 4)
    items = torch.randn(7, 4)
    item_ids = torch.tensor([[0, 3], [5, 6]])

    with torch.no_grad():
        full = scorer(user, items)
        selected = scorer(user, items, item_ids=item_ids)

    assert full.shape == (2, 7)
    torch.testing.assert_close(selected, full.gather(1, item_ids))
