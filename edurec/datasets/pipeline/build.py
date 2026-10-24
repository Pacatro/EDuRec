from dataclasses import dataclass

import pandas as pd

from edurec import settings
from edurec.datasets.cache import ProcessedData
from edurec.datasets.config import DataConfig, SplitStrategy
from edurec.datasets.features.ids import encode_ids
from edurec.datasets.features.lists import normalize_lists
from edurec.datasets.features.tabular import (
    categorical_tensor,
    dense_tensor,
    fit_tabular,
    resolve_schema,
    transform_features,
)
from edurec.datasets.features.text import encode_graph_text
from edurec.datasets.pipeline.prepare import (
    clean_cols,
    deduplicate_interactions,
    filter_sparse,
)
from edurec.datasets.pipeline.splits import (
    add_relevance,
    get_relevance_threshold,
    split_data,
)
from edurec.datasets.registry import RawData, Schema


@dataclass(frozen=True)
class BuildResult:
    """Processed artifacts plus the selected source tables used to build them."""

    data: ProcessedData
    raw: RawData


def split_strategy_for(interactions: pd.DataFrame) -> SplitStrategy:
    """Split strategy implied by the available interaction columns."""
    return "temporal" if settings.TIME_COL in interactions.columns else "random"


def build_processed(raw: RawData, config: DataConfig) -> BuildResult:
    """Turn raw source tables into split and encoded processed data.

    Order: column cleaning, one normalization/deduplication step, sparsity
    filtering, temporal or random split, relevance
    labelling, and preprocessing fitted on the training split only.
    """
    interactions = clean_cols(raw.interactions)
    users = clean_cols(raw.user_features)
    items = clean_cols(raw.item_features)

    if config.deduplicate or settings.TIME_COL not in interactions:
        interactions = deduplicate_interactions(interactions)

    if config.remove_sparse:
        users, items, interactions = filter_sparse(
            users,
            items,
            interactions,
            min_interactions=config.min_interactions,
        )

    prepared_raw = raw._replace(
        interactions=interactions,
        user_features=users,
        item_features=items,
    )

    train, val, test = _split_with_relevance(prepared_raw.interactions, config)
    artifacts = preprocess(
        schema=raw.schema,
        users=prepared_raw.user_features,
        items=prepared_raw.item_features,
        train=train,
        val=val,
        test=test,
        graph_entities=raw.graph_entities,
    )

    artifacts.manifest = config.manifest(
        strategy=split_strategy_for(prepared_raw.interactions),
        num_users=len(prepared_raw.user_features),
        num_items=len(prepared_raw.item_features),
    )
    artifacts.reports = _reports(raw, artifacts, train)
    return BuildResult(data=artifacts, raw=prepared_raw)


def _reports(
    raw: RawData,
    artifacts: ProcessedData,
    train: pd.DataFrame,
) -> dict:
    train_items = train[settings.ITEM_COL].map(artifacts.id_maps["items"])

    return {
        "source_report": raw.source_report,
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
        min_interactions=config.min_interactions,
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
