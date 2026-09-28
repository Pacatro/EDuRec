from collections import defaultdict, deque
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch

from .. import settings


class History(NamedTuple):
    """Padded chronological history for one split.

    ``items`` holds item IDs shifted by one so that zero is padding, ``mask``
    marks the valid slots and ``times`` holds, for every valid slot, the elapsed
    time between that interaction and the prediction target (zero for padding
    and for datasets without a timestamp).
    """

    items: torch.Tensor
    mask: torch.Tensor
    times: torch.Tensor


def build_histories(
    splits: dict[str, pd.DataFrame],
    enabled: bool = True,
) -> dict[str, History]:
    """Build fixed-size, chronological histories for every split.

    History from an earlier split is available to later splits. When the split
    carries a timestamp, each valid slot also records how long before the target
    interaction it happened, which the sequence encoder turns into temporal
    features.
    """
    history_len = settings.MAX_HISTORY_LEN if enabled else 0
    histories: dict[str, History] = {}
    user_items: defaultdict[int, deque[int]] = defaultdict(
        lambda: deque(maxlen=history_len)
    )
    user_times: defaultdict[int, deque[float]] = defaultdict(
        lambda: deque(maxlen=history_len)
    )

    for split in ("train", "val", "test"):
        df = splits.get(split)
        if df is None:
            raise RuntimeError(f"Processed split {split} is not available.")

        has_time = enabled and settings.TIME_COL in df.columns
        items = torch.zeros((len(df), history_len), dtype=torch.long)
        valid_mask = torch.zeros((len(df), history_len), dtype=torch.bool)
        times = torch.zeros((len(df), history_len), dtype=torch.float32)

        if enabled:
            columns = [settings.USER_COL, settings.ITEM_COL]
            if has_time:
                columns.append(settings.TIME_COL)

            for row_idx, row in enumerate(df[columns].itertuples(index=False)):
                user_id = int(row[0])
                item_id = int(row[1])
                target_time = _as_float(row[2], 0.0) if has_time else 0.0

                past_items = user_items[user_id]
                past_times = user_times[user_id]
                if past_items:
                    length = len(past_items)
                    items[row_idx, :length] = torch.tensor(past_items).add_(1)
                    valid_mask[row_idx, :length] = True
                    if has_time:
                        deltas = [target_time - past_time for past_time in past_times]
                        times[row_idx, :length] = torch.as_tensor(
                            deltas, dtype=torch.float32
                        )

                past_items.append(item_id)
                if has_time:
                    past_times.append(target_time)

        histories[split] = History(items=items, mask=valid_mask, times=times)

    return histories


def _as_float(value: object, default: float) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if np.isnan(number) else number
