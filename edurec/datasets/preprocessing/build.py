from dataclasses import dataclass

import pandas as pd

from edurec import settings
from edurec.datasets.cache import ProcessedData
from edurec.datasets.config import DataConfig, SplitStrategy
from edurec.datasets.preprocessing.ids import encode_ids
from edurec.datasets.preprocessing.lists import normalize_lists
from edurec.datasets.preprocessing.splits import (
    add_relevance,
    get_relevance_threshold,
    split_data,
)
from edurec.datasets.preprocessing.tabular import (
    categorical_tensor,
    dense_tensor,
    fit_tabular,
    resolve_schema,
    transform_features,
)
from edurec.datasets.preprocessing.text import encode_graph_text
from edurec.datasets.sources.registry import RawData, Schema


@dataclass(frozen=True)
class BuildResult:
    """Processed artifacts plus the selected source tables used to build them."""

    data: ProcessedData
    raw: RawData


def split_strategy_for(interactions: pd.DataFrame) -> SplitStrategy:
    """Split strategy implied by the available interaction columns."""
    return "temporal" if settings.TIME_COL in interactions.columns else "random"


def build_processed(raw: RawData, config: DataConfig) -> BuildResult:
    """Split, label and encode tables already prepared by the source.

    Source preparation owns cleaning, filtering and sampling. This stage only
    fits interaction preprocessing on train and preserves preparation reports.
    """
    train, val, test = _split_with_relevance(raw.interactions, config)
    artifacts = preprocess(
        schema=raw.schema,
        users=raw.user_features,
        items=raw.item_features,
        train=train,
        val=val,
        test=test,
        graph_entities=raw.graph_entities,
    )

    artifacts.manifest = config.manifest(
        strategy=split_strategy_for(raw.interactions),
        num_users=len(raw.user_features),
        num_items=len(raw.item_features),
    )
    artifacts.reports = {**(raw.reports or {}), **_reports(artifacts, train)}
    return BuildResult(data=artifacts, raw=raw)


def _reports(
    artifacts: ProcessedData,
    train: pd.DataFrame,
) -> dict:
    train_items = train[settings.ITEM_COL].map(artifacts.id_maps["items"])

    return {
        "split_report": {
            name: {
                "interactions": len(frame),
                "users": frame[settings.USER_COL].nunique(),
                "courses": frame[settings.ITEM_COL].nunique(),
                "targets_unseen_in_train": int(
                    (~frame[settings.ITEM_COL].isin(train_items)).sum()
                ),
            }
            for name, frame in artifacts.splits().items()
        },
    }


def _split_with_relevance(
    interactions: pd.DataFrame,
    config: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, val, test = split_data(
        interactions,
        test_ratio=config.test_ratio,
        val_ratio=config.val_ratio,
        min_interactions=config.effective_min_interactions,
        random_state=config.random_state,
    )

    thresholds = get_relevance_threshold(train)

    return (
        add_relevance(train, thresholds),
        add_relevance(val, thresholds),
        add_relevance(test, thresholds),
    )


def preprocess(
    schema: Schema,
    users: pd.DataFrame,
    items: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    graph_entities: dict[str, pd.DataFrame] | None = None,
) -> ProcessedData:
    """Fit static catalog metadata and TRAIN context, then transform every split."""
    schema = resolve_schema(schema, {"users": users, "items": items, "inter": train})
    users, items, splits, id_maps = encode_ids(
        users, items, {"train": train, "val": val, "test": test}
    )
    users = normalize_lists(users, schema["users"])
    items = normalize_lists(items, schema["items"])
    splits = {
        name: normalize_lists(frame, schema["inter"]) for name, frame in splits.items()
    }
    transforms = {
        "users": fit_tabular(users, schema["users"]),
        "items": fit_tabular(items, schema["items"]),
        "inter": fit_tabular(splits["train"], schema["inter"]),
    }
    user_frame = transform_features(users, transforms["users"], schema["users"])
    item_frame = transform_features(items, transforms["items"], schema["items"])
    encoded_splits = {
        name: transform_features(frame, transforms["inter"], schema["inter"])
        for name, frame in splits.items()
    }

    return ProcessedData(
        train=encoded_splits["train"],
        val=encoded_splits["val"],
        test=encoded_splits["test"],
        u_static_feats=dense_tensor(user_frame),
        u_cat_feats=categorical_tensor(user_frame),
        i_static_feats=dense_tensor(item_frame),
        user_features=users,
        item_features=items,
        schema=schema,
        id_maps=id_maps,
        transforms=transforms,
        graph_node_features=encode_graph_text(items, graph_entities, schema["items"]),
    )
