from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from edurec import settings
from edurec.datasets.cache import ProcessedData
from edurec.datasets.config import DataConfig, SplitStrategy
from edurec.datasets.features.tabular import DataProcessor
from edurec.datasets.pipeline.prepare import (
    SelectionResult,
    clean_cols,
    deduplicate_interactions,
    filter_sparse,
    select_dataset,
)
from edurec.datasets.pipeline.splits import (
    add_relevance,
    get_relevance_threshold,
    split_data,
)
from edurec.datasets.registry import RawData


@dataclass(frozen=True)
class BuildResult:
    """Processed artifacts plus the selected source tables used to build them."""

    data: ProcessedData
    raw: RawData


def split_strategy_for(interactions: pd.DataFrame) -> SplitStrategy:
    """Split strategy implied by the available interaction columns."""
    return "temporal" if settings.TIME_COL in interactions.columns else "random"


def build_processed(raw: RawData, config: DataConfig) -> BuildResult:
    """Turn raw source tables into split, encoded and selected processed data.

    Order: column cleaning, one normalization/deduplication step, sparsity
    filtering, community selection, temporal or random split, relevance
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

    selected = select_dataset(
        raw._replace(
            interactions=interactions,
            user_features=users,
            item_features=items,
        ),
        community=config.community,
        min_interactions=config.min_interactions,
    )
    selected_raw = selected.data

    train, val, test = _split_with_relevance(selected_raw.interactions, config)
    artifacts = preprocess(
        processor=DataProcessor(schema=raw.schema),
        users=selected_raw.user_features,
        items=selected_raw.item_features,
        train=train,
        val=val,
        test=test,
        graph_entities=raw.graph_entities,
    )

    artifacts.manifest = config.manifest(
        strategy=split_strategy_for(selected_raw.interactions),
        num_users=len(selected_raw.user_features),
        num_items=len(selected_raw.item_features),
    )
    artifacts.reports = _reports(raw, selected, artifacts, train)
    return BuildResult(data=artifacts, raw=selected_raw)


def _reports(
    raw: RawData,
    selected: SelectionResult,
    artifacts: ProcessedData,
    train: pd.DataFrame,
) -> dict:
    train_items = train[settings.ITEM_COL]

    return {
        "source_report": raw.source_report,
        "selection_report": selected.report,
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
    processor: DataProcessor,
    users: pd.DataFrame,
    items: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    graph_entities: dict[str, pd.DataFrame] | None = None,
) -> ProcessedData:
    processor.fit(users_train=users, items_train=items, interactions_train=train)
    processor.fit_graph_text_features(items, graph_entities)

    features = processor.transform(users=users, items=items)
    split_features = {
        "train": processor.transform(interactions=train),
        "val": processor.transform(interactions=val),
        "test": processor.transform(interactions=test),
    }

    if features.users is None or features.items is None:
        raise RuntimeError("User/item features were not processed.")

    user_frame = features.users
    item_frame = features.items
    if features.text_embeddings["users"] is not None:
        user_frame = pd.concat([user_frame, features.text_embeddings["users"]], axis=1)
    if features.text_embeddings["items"] is not None:
        item_frame = pd.concat([item_frame, features.text_embeddings["items"]], axis=1)

    split_dfs: dict[str, pd.DataFrame] = {}
    for name, processed in split_features.items():
        if processed.interactions is None:
            raise RuntimeError(f"{name} interactions were not processed.")
        split_dfs[name] = processed.interactions
        if processed.text_embeddings["inter"] is not None:
            split_dfs[name] = pd.concat(
                [split_dfs[name], processed.text_embeddings["inter"]],
                axis=1,
            )
        split_dfs[name] = split_dfs[name].reset_index(drop=True)

    static_feats = {}
    for name, df, prefix, id_col in (
        ("users", user_frame, "users", settings.USER_COL),
        ("items", item_frame, "items", settings.ITEM_COL),
    ):
        metadata = processor.pipeline(prefix).metadata
        cols = metadata.numeric_cols + metadata.text_embedding_cols
        static_feats[name] = torch.as_tensor(
            df.sort_values(id_col)[cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

    u_cat_feats = _categorical_codes(
        user_frame,
        processor.pipeline("users").metadata.categorical_cols,
        id_col=settings.USER_COL,
    )

    return ProcessedData(
        train=split_dfs["train"],
        val=split_dfs["val"],
        test=split_dfs["test"],
        u_static_feats=static_feats["users"],
        u_cat_feats=u_cat_feats,
        i_static_feats=static_feats["items"],
        user_features=users.reset_index(drop=True),
        item_features=items.reset_index(drop=True),
        data_processor=processor,
    )


def _categorical_codes(
    frame: pd.DataFrame,
    categorical_cols: list[str],
    id_col: str,
) -> torch.Tensor:
    """Integer codes for the encoded categorical columns, ordered by mapped id.

    The processor already ordinal-encodes categorical fields, so the values are
    reused as-is. ``-1`` marks unknown categories and is preserved for the model
    to map to a padding embedding.
    """
    if not categorical_cols:
        return torch.zeros((len(frame), 0), dtype=torch.long)

    values = frame.sort_values(id_col)[categorical_cols].to_numpy(dtype=np.float32)
    values = np.nan_to_num(values, nan=-1.0)
    return torch.as_tensor(values.astype(np.int64), dtype=torch.long)
