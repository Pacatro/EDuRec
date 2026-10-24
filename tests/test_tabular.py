import numpy as np
import pandas as pd
import pytest
import torch

from edurec import settings
from edurec.datasets.features.ids import encode_ids
from edurec.datasets.features.lists import normalize_lists
from edurec.datasets.features.tabular import (
    categorical_cardinalities,
    categorical_tensor,
    dense_tensor,
    feature_columns,
    fit_tabular,
    resolve_schema,
    transform_features,
)


@pytest.mark.parametrize("with_observed_column", [False, True])
def test_numeric_features_missing_in_training_keep_schema(with_observed_column):
    numeric_cols = ["year_of_birth"]
    users = pd.DataFrame({settings.USER_COL: [0, 1, 2], "year_of_birth": [np.nan] * 3})
    if with_observed_column:
        numeric_cols.append("score")
        users["score"] = [2.0, np.nan, 6.0]
    schema = {"num": numeric_cols}
    transformer = fit_tabular(users, schema)

    transformed = transform_features(users, transformer, schema)
    assert transformed["num__year_of_birth"].tolist() == [0.0] * 3
    assert feature_columns(transformer, "num") == [
        f"num__{col}" for col in numeric_cols
    ]
    if with_observed_column:
        assert transformed["num__score"].tolist() == [0.0, 0.5, 1.0]

    held_out = users.iloc[:1].copy()
    held_out["year_of_birth"] = 1990.0
    result = transform_features(held_out, transformer, schema)
    assert result.columns.tolist() == transformed.columns.tolist()
    assert np.isfinite(result.to_numpy()).all()
    assert transformer is not None
    imputer = transformer.named_transformers_["num"].named_steps["imputer"]
    assert imputer.statistics_[0] == 0.0


def test_transform_uses_train_statistics_and_preserves_unknown_categories():
    train = pd.DataFrame({"score": [2.0, 6.0], "category": ["a", "b"]})
    held_out = pd.DataFrame({"score": [10.0, np.nan], "category": ["new", "a"]})
    schema = {"num": ["score"], "cat": ["category"]}
    transformer = fit_tabular(train, schema)

    result = transform_features(held_out, transformer, schema)

    assert result["num__score"].tolist() == [2.0, 0.5]
    assert result["cat__category"].tolist() == [-1, 0]
    assert categorical_cardinalities(transformer) == [3]
    assert categorical_tensor(result).dtype == torch.int64
    assert dense_tensor(result).dtype == torch.float32


@pytest.mark.parametrize("values", [[None, None], [np.nan, np.nan], ["", " "]])
def test_entirely_missing_categorical_column_keeps_shape(values):
    frame = pd.DataFrame({"category": values})
    schema = {"cat": ["category"]}
    transformer = fit_tabular(frame, schema)

    result = transform_features(frame, transformer, schema)
    unknown = transform_features(
        pd.DataFrame({"category": ["new"]}), transformer, schema
    )

    assert result["cat__category"].tolist() == [0, 0]
    assert unknown["cat__category"].tolist() == [-1]
    assert categorical_cardinalities(transformer) == [2]
    assert dense_tensor(result).shape == (2, 0)
    empty = transform_features(frame.iloc[:0], transformer, schema)
    assert categorical_tensor(empty).shape == (0, 1)


def test_ids_are_integer_and_aligned_with_entity_rows():
    users = pd.DataFrame({settings.USER_COL: ["z", "a"], "score": [9, 3]})
    items = pd.DataFrame({settings.ITEM_COL: [2**30 + 1, 2**30], "name": ["x", "y"]})
    events = pd.DataFrame(
        {settings.USER_COL: ["a", "z"], settings.ITEM_COL: [2**30, 2**30 + 1]}
    )

    mapped_users, mapped_items, splits, maps = encode_ids(
        users, items, {"train": events}
    )

    assert mapped_users[settings.USER_COL].tolist() == [0, 1]
    assert mapped_users["score"].tolist() == [9, 3]
    assert mapped_items["name"].tolist() == ["x", "y"]
    assert splits["train"][settings.USER_COL].tolist() == [1, 0]
    assert splits["train"][settings.ITEM_COL].tolist() == [1, 0]
    assert splits["train"][settings.ITEM_COL].dtype == np.int64
    assert maps["items"] == {2**30 + 1: 0, 2**30: 1}
    assert users[settings.USER_COL].tolist() == ["z", "a"]


@pytest.mark.parametrize("ids", [["a", "a"], ["a", None]])
def test_entity_ids_must_be_unique_and_present(ids):
    with pytest.raises(ValueError, match="non-null and unique"):
        encode_ids(
            pd.DataFrame({settings.USER_COL: ids}),
            pd.DataFrame({settings.ITEM_COL: ["x"]}),
            {},
        )


def test_interactions_cannot_reference_entities_outside_catalog():
    with pytest.raises(ValueError, match="outside the items catalog"):
        encode_ids(
            pd.DataFrame({settings.USER_COL: ["u"]}),
            pd.DataFrame({settings.ITEM_COL: ["i"]}),
            {
                "train": pd.DataFrame(
                    {settings.USER_COL: ["u"], settings.ITEM_COL: ["missing"]}
                )
            },
        )


def test_lists_keep_tokens_and_remove_missing_members_without_binarizing():
    frame = pd.DataFrame({"tags": [[" a ", None, np.nan, "a", ""], '["b", "b"]']})
    result = normalize_lists(frame, {"list": ["tags"]})

    assert result["tags"].tolist() == [["a"], ["b"]]
    assert result.columns.tolist() == ["tags"]
    assert frame.iloc[0]["tags"][1] is None


def test_resolved_schema_freezes_enabled_fields(monkeypatch):
    monkeypatch.setattr(settings, "PREPROCESS_FEATURE_TYPES", ("numeric", "time"))
    schema = {
        "items": {
            "num": ["score"],
            "cat": ["category"],
            "list": ["tags"],
            "graph_list": ["concept"],
            "graph_text": {"concept": ["name"]},
        }
    }
    frame = pd.DataFrame(
        {"score": [1], "category": ["a"], "tags": [[]], "concept": [["b"]]}
    )

    result = resolve_schema(schema, {"items": frame})["items"]

    assert result["num"] == ["score"]
    assert result["cat"] == result["list"] == []
    assert result["graph_list"] == ["concept"]
    assert result["graph_text"] == {}
    assert schema["items"]["cat"] == ["category"]


def test_source_columns_cannot_collide_with_encoded_feature_names():
    with pytest.raises(ValueError, match="reserved feature prefixes"):
        resolve_schema({}, {"items": pd.DataFrame({"num__score": [1]})})
