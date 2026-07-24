from collections import defaultdict, deque

import pandas as pd
import torch

from .. import settings


def build_histories(
    splits: dict[str, pd.DataFrame | None],
    enabled: bool = True,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Build fixed-size, chronological item histories for every split.

    History from an earlier split is available to later splits. Item IDs are
    shifted by one in the returned tensor so that zero remains padding.
    """
    history_len = settings.MAX_HISTORY_LEN if enabled else 0
    histories: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    user_items: defaultdict[int, deque[int]] = defaultdict(
        lambda: deque(maxlen=history_len)
    )

    for split in ("train", "val", "test"):
        df = splits.get(split)
        if df is None:
            raise RuntimeError(f"Processed split {split} is not available.")

        items = torch.zeros((len(df), history_len), dtype=torch.long)
        valid_mask = torch.zeros((len(df), history_len), dtype=torch.bool)
        if enabled:
            for row_idx, (user_id, item_id) in enumerate(
                df[[settings.USER_COL, settings.ITEM_COL]].itertuples(index=False)
            ):
                past = user_items[int(user_id)]
                if past:
                    items[row_idx, : len(past)] = torch.tensor(past).add_(1)
                    valid_mask[row_idx, : len(past)] = True
                past.append(int(item_id))

        histories[split] = items, valid_mask

    return histories
