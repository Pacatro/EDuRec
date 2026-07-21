from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .. import settings


@dataclass
class UserHistory:
    """
    Represents the prefix history of a user for next-item prediction.

    items: padded item ids (0 = padding)
    valid_mask: mask indicating valid history positions
    """

    items: torch.Tensor
    valid_mask: torch.Tensor


def build_histories(
    splits: dict[str, pd.DataFrame | None],
    enabled: bool = True,
) -> dict[str, UserHistory]:
    histories = {}
    user_state: dict[int, list[int]] = {}

    for split in ("train", "val", "test"):
        df = splits.get(split)
        if df is None:
            raise RuntimeError(f"Processed split {split} is not available.")

        split_len = len(df)
        history_len = settings.MAX_HISTORY_LEN if enabled else 0
        history = UserHistory(
            items=torch.zeros((split_len, history_len), dtype=torch.long),
            valid_mask=torch.zeros(
                (split_len, history_len),
                dtype=torch.bool,
            ),
        )
        if not enabled:
            histories[split] = history
            continue
        users = df[settings.USER_COL].to_numpy(dtype=np.int64)
        items = df[settings.ITEM_COL].to_numpy(dtype=np.int64)

        for row_idx, user_id in enumerate(users):
            past = user_state.get(int(user_id), [])[-settings.MAX_HISTORY_LEN :]
            if past:
                history.items[row_idx, : len(past)] = torch.tensor(
                    [item_id + 1 for item_id in past],
                    dtype=torch.long,
                )
                history.valid_mask[row_idx, : len(past)] = True

            user_state.setdefault(int(user_id), []).append(int(items[row_idx]))

        histories[split] = history

    return histories
