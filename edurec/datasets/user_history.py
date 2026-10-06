from collections import defaultdict, deque

import pandas as pd
import torch

from edurec import settings

SPLIT_ORDER = ("train", "val", "test")


def build_histories(
    splits: dict[str, pd.DataFrame],
    max_history: int,
    enabled: bool = True,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build fixed-size, strictly chronological histories for every split.

    For each split the returned tuple is ``(items, valid_mask, context_index)``,
    each with shape ``[n, max_history]``:

    * ``items``: past item IDs shifted by one so zero remains padding.
    * ``valid_mask``: ``True`` where a real past event is stored.
    * ``context_index``: global interaction row index of each past event, or
      ``-1`` for padding. The global index is the row position in the
      concatenation of the splits in ``SPLIT_ORDER`` (they map to
      ``[0, len(train))``, then val rows, then test rows).

    The alignment is strict: ``items[r, t]``, ``valid_mask[r, t]`` and
    ``context_index[r, t]`` all refer to the SAME past event. A row's own event
    is never part of its own history, and only events strictly before the
    current row may appear, which prevents target leakage. Earlier splits feed
    into later ones: while iterating rows in order, each row first records the
    current per-user history and only then appends its own event.

    When ``enabled`` is ``False`` every split yields zero-width tensors of the
    appropriate dtypes instead of histories.
    """
    if max_history < 0:
        raise ValueError("max_history must be non-negative.")

    split_frames: list[tuple[str, pd.DataFrame]] = []
    for split in SPLIT_ORDER:
        df = splits.get(split)
        if df is None:
            raise RuntimeError(f"Processed split {split} is not available.")
        split_frames.append((split, df))

    histories: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    if not enabled:
        for split, df in split_frames:
            n_rows = len(df)
            histories[split] = (
                torch.zeros((n_rows, 0), dtype=torch.long),
                torch.zeros((n_rows, 0), dtype=torch.bool),
                torch.zeros((n_rows, 0), dtype=torch.long),
            )
        return histories

    global_items: list[int] = []
    for _, df in split_frames:
        global_items.extend(int(item) for item in df[settings.ITEM_COL].tolist())

    user_history: defaultdict[int, deque[int]] = defaultdict(
        lambda: deque(maxlen=max_history)
    )

    global_offset = 0
    for split, df in split_frames:
        n_rows = len(df)
        items = torch.zeros((n_rows, max_history), dtype=torch.long)
        valid_mask = torch.zeros((n_rows, max_history), dtype=torch.bool)
        context_index = torch.full((n_rows, max_history), -1, dtype=torch.long)

        users = df[settings.USER_COL].tolist()
        for row_idx in range(n_rows):
            past = user_history[int(users[row_idx])]
            if past:
                past_indices = list(past)
                count = len(past_indices)
                items[row_idx, :count] = torch.tensor(
                    [global_items[g] + 1 for g in past_indices], dtype=torch.long
                )
                valid_mask[row_idx, :count] = True
                context_index[row_idx, :count] = torch.tensor(
                    past_indices, dtype=torch.long
                )
            past.append(global_offset + row_idx)

        histories[split] = (items, valid_mask, context_index)
        global_offset += n_rows

    return histories
