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
    ctx: contextual features aligned with items
    valid_mask: mask indicating valid history positions
    """

    items: torch.Tensor
    ctx: torch.Tensor
    valid_mask: torch.Tensor


def build_histories(
    splits: dict[str, pd.DataFrame | None], excluded_cols: set[str]
) -> dict[str, UserHistory]:
    histories = {}
    user_state: dict[int, list[tuple[int, list[float]]]] = {}

    for split in ("train", "val", "test"):
        df = splits.get(split)
        if df is None:
            raise RuntimeError(f"Processed split {split} is not available.")

        context_cols = [col for col in df.columns if col not in excluded_cols]
        history = UserHistory(
            items=torch.zeros((len(df), settings.MAX_HISTORY_LEN), dtype=torch.long),
            ctx=torch.zeros(
                (len(df), settings.MAX_HISTORY_LEN, len(context_cols)),
                dtype=torch.float32,
            ),
            valid_mask=torch.zeros(
                (len(df), settings.MAX_HISTORY_LEN),
                dtype=torch.bool,
            ),
        )
        users = df[settings.USER_COL].to_numpy(dtype=np.int64)
        items = df[settings.ITEM_COL].to_numpy(dtype=np.int64)
        context = (
            df[context_cols].to_numpy(dtype=np.float32)
            if context_cols
            else np.empty((len(df), 0), dtype=np.float32)
        )

        for row_idx, user_id in enumerate(users):
            past = user_state.get(int(user_id), [])[-settings.MAX_HISTORY_LEN :]
            if past:
                history.items[row_idx, : len(past)] = torch.tensor(
                    [item_id + 1 for item_id, _ in past],
                    dtype=torch.long,
                )
                history.valid_mask[row_idx, : len(past)] = True
                if context_cols:
                    history.ctx[row_idx, : len(past)] = torch.tensor(
                        [values for _, values in past],
                        dtype=torch.float32,
                    )

            user_state.setdefault(int(user_id), []).append(
                (int(items[row_idx]), context[row_idx].tolist())
            )

        histories[split] = history

    return histories
