import pandas as pd
import torch

from .. import settings
from .dataprocessor import DataProcessor


def build_user_stats(
    users: pd.DataFrame,
    train: pd.DataFrame,
    processor: DataProcessor,
    num_users: int,
) -> torch.Tensor:
    """
    Per-user router statistics with columns:
    log interactions, normalized history length, missing attribute ratio,
    graph availability, attribute availability, history availability.
    """
    stats = torch.zeros((num_users, 6), dtype=torch.float32)
    user_ids = _valid_ids(_positive_interactions(train)[settings.USER_COL], num_users)
    counts = torch.bincount(user_ids, minlength=num_users).float()
    missing_ratio = _raw_missing_ratio(
        frame=users,
        id_col=settings.USER_COL,
        id_map=processor.user_id_map,
        processor=processor,
        prefix="users",
        size=num_users,
    )

    stats[:, 0] = torch.log1p(counts)
    stats[:, 1] = (counts / max(settings.MAX_HISTORY_LEN, 1)).clamp(max=1.0)
    stats[:, 2] = missing_ratio
    stats[:, 3] = (counts > 0).float()
    if _num_non_text_feats(processor, "users") > 0:
        stats[:, 4] = (missing_ratio < 1.0).float()
    stats[:, 5] = (counts > 0).float()
    return stats


def build_item_stats(
    items: pd.DataFrame,
    train: pd.DataFrame,
    processor: DataProcessor,
    num_items: int,
) -> torch.Tensor:
    """
    Per-item router statistics with columns:
    log distinct users, missing attribute ratio, normalized text length,
    graph availability, attribute availability, text availability.
    """
    stats = torch.zeros((num_items, 6), dtype=torch.float32)
    user_item_pairs = _positive_interactions(train)[
        [settings.USER_COL, settings.ITEM_COL]
    ].drop_duplicates()
    item_ids = _valid_ids(user_item_pairs[settings.ITEM_COL], num_items)
    user_counts = torch.bincount(item_ids, minlength=num_items).float()
    missing_ratio = _raw_missing_ratio(
        frame=items,
        id_col=settings.ITEM_COL,
        id_map=processor.item_id_map,
        processor=processor,
        prefix="items",
        size=num_items,
    )
    text_length = _raw_text_length_ratio(
        frame=items,
        id_col=settings.ITEM_COL,
        id_map=processor.item_id_map,
        processor=processor,
        prefix="items",
        size=num_items,
    )

    stats[:, 0] = torch.log1p(user_counts)
    stats[:, 1] = missing_ratio
    stats[:, 2] = text_length
    stats[:, 3] = (user_counts > 0).float()
    if _num_non_text_feats(processor, "items") > 0:
        stats[:, 4] = (missing_ratio < 1.0).float()
    stats[:, 5] = (text_length > 0).float()
    return stats


def _positive_interactions(df: pd.DataFrame) -> pd.DataFrame:
    if settings.RELEVANT_COL not in df.columns:
        return df.reset_index(drop=True)

    return df.loc[df[settings.RELEVANT_COL] > 0].reset_index(drop=True)


def _valid_ids(values: pd.Series, size: int) -> torch.Tensor:
    ids = torch.as_tensor(values.to_numpy(copy=True), dtype=torch.long)
    return ids[(ids >= 0) & (ids < size)]


def _raw_missing_ratio(
    frame: pd.DataFrame,
    id_col: str,
    id_map: dict[object, int],
    processor: DataProcessor,
    prefix: str,
    size: int,
) -> torch.Tensor:
    ratios = torch.zeros(size, dtype=torch.float32)
    if size == 0:
        return ratios

    groups = processor.column_groups.get(prefix, {})
    cols = list(
        dict.fromkeys(
            [
                *groups.get("numeric", []),
                *groups.get("categorical", []),
                *groups.get("list", []),
            ]
        )
    )
    cols = [col for col in cols if col in frame.columns]
    if not cols:
        return ratios

    missing = frame[cols].isna()
    for col in cols:
        if frame[col].dtype == object:
            missing[col] = frame[col].map(_is_missing_value)

    row_ratios = missing.mean(axis=1)
    mapped_ids = frame[id_col].map(id_map)
    valid = mapped_ids.notna()
    if not valid.any():
        return ratios

    ids = torch.as_tensor(mapped_ids[valid].to_numpy(dtype=int), dtype=torch.long)
    values = torch.as_tensor(
        row_ratios[valid].to_numpy(dtype="float32"), dtype=torch.float32
    )
    valid_ids = (ids >= 0) & (ids < size)
    ratios[ids[valid_ids]] = values[valid_ids]
    return ratios


def _raw_text_length_ratio(
    frame: pd.DataFrame,
    id_col: str,
    id_map: dict[object, int],
    processor: DataProcessor,
    prefix: str,
    size: int,
) -> torch.Tensor:
    ratios = torch.zeros(size, dtype=torch.float32)
    groups = processor.column_groups.get(prefix, {})
    text_cols = [col for col in groups.get("text", []) if col in frame.columns]
    if not text_cols:
        return ratios

    text_frame = frame[text_cols].fillna("").astype(str)
    lengths = text_frame.apply(
        lambda row: len(" ".join(value.strip() for value in row).split()),
        axis=1,
    )
    lengths = (lengths / max(processor.text_max_tokens, 1)).clip(upper=1.0)

    mapped_ids = frame[id_col].map(id_map)
    valid = mapped_ids.notna()
    if not valid.any():
        return ratios

    ids = torch.as_tensor(mapped_ids[valid].to_numpy(dtype=int), dtype=torch.long)
    values = torch.as_tensor(
        lengths[valid].to_numpy(dtype="float32"), dtype=torch.float32
    )
    valid_ids = (ids >= 0) & (ids < size)
    ratios[ids[valid_ids]] = values[valid_ids]
    return ratios


def _num_non_text_feats(processor: DataProcessor, prefix: str) -> int:
    metadata = processor.feature_metadata.get(prefix)
    if metadata is None:
        return 0
    return len(metadata.dense_cols) + len(metadata.categorical_cols)


def _is_missing_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set)):
        return not any(str(item).strip() for item in value)

    missing = pd.isna(value)  # type: ignore
    if getattr(missing, "ndim", 0) == 0:
        return bool(missing)
    return False
