import numpy as np
import pandas as pd
import pytest

from edurec import settings
from edurec.datasets.features.tabular import DataProcessor, FeatureCounts


def test_feature_counts_total_counts_text_once():
    counts = FeatureCounts(dense=5, categorical=2, text=3)

    assert counts.total == 7


@pytest.mark.parametrize("with_observed_column", [False, True])
def test_numeric_features_missing_in_training_keep_schema(with_observed_column):
    numeric_cols = ["year_of_birth"]
    users = pd.DataFrame({settings.USER_COL: [0, 1, 2], "year_of_birth": [np.nan] * 3})
    if with_observed_column:
        numeric_cols.append("score")
        users["score"] = [2.0, np.nan, 6.0]
    items = pd.DataFrame({settings.ITEM_COL: [0]})
    interactions = pd.DataFrame(
        {settings.USER_COL: [0, 1, 2], settings.ITEM_COL: [0, 0, 0]}
    )
    processor = DataProcessor(schema={"users": {"num": numeric_cols}})
    processor.fit(users, items, interactions)

    transformed = processor.transform(users=users).users
    assert transformed is not None
    assert transformed["num__year_of_birth"].tolist() == [0.0] * 3
    assert processor.pipeline("users").metadata.numeric_cols == [
        f"num__{col}" for col in numeric_cols
    ]
    if with_observed_column:
        assert transformed["num__score"].tolist() == [0.0, 0.5, 1.0]

    held_out = users.iloc[:1].copy()
    held_out["year_of_birth"] = 1990.0
    result = processor.transform(users=held_out).users
    assert result is not None
    assert result.columns.tolist() == transformed.columns.tolist()
    assert np.isfinite(result.to_numpy()).all()
    preprocessor = processor.pipeline("users").preprocessor
    assert preprocessor is not None
    imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    assert imputer.statistics_[0] == 0.0
