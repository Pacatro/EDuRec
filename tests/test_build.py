import pandas as pd

from edurec import settings
from edurec.datasets import DataConfig, DatasetName
from edurec.datasets.preprocessing.build import build_processed


def test_build_preserves_disconnected_groups(raw_communities):
    result = build_processed(
        raw_communities, DataConfig(dataset=DatasetName.COCO, random_state=42)
    )

    pd.testing.assert_frame_equal(result.raw.interactions, raw_communities.interactions)
    assert result.data.num_users == 7
    assert result.data.num_items == 8
    assert result.data.num_interactions == 28
    for user in result.raw.user_features[settings.USER_COL]:
        splits = [
            frame.loc[frame[settings.USER_COL] == user, settings.TIME_COL]
            for frame in result.data.splits().values()
        ]
        assert splits[0].max() < splits[1].min()
        assert splits[1].max() < splits[2].min()


def test_build_can_keep_sparse_items(make_raw):
    raw = make_raw(
        {user: [0, 1, 2, user + 3] for user in range(4)},
        {item: [str(item)] for item in range(7)},
    )
    result = build_processed(
        raw,
        DataConfig(dataset=DatasetName.COCO, remove_sparse=False, random_state=42),
    )

    pd.testing.assert_frame_equal(result.raw.interactions, raw.interactions)
    assert result.data.num_items == 7
    assert result.data.num_interactions == 16


def test_interaction_budget_prunes_the_user_catalog(make_raw):
    raw = make_raw(
        {user: [0, 1, 2, 3, 4] for user in range(10)},
        {item: [] for item in range(5)},
    )
    result = build_processed(
        raw,
        DataConfig(
            dataset=DatasetName.MOOCCUBEX,
            max_interactions=25,
            remove_sparse=False,
            random_state=42,
        ),
    )

    selected = set(result.raw.interactions[settings.USER_COL])
    assert result.data.num_users == 5
    assert result.data.num_interactions == 25
    assert len(result.data.id_maps["users"]) == 5
    assert len(result.raw.user_features) == 5
    assert set(result.raw.user_features[settings.USER_COL]) == selected
    assert result.data.reports["interaction_budget"]["users"] == 5
