import ast
import re
from html import unescape
from typing import Any

import numpy as np
import pandas as pd


def coerce_list_tokens(value: object) -> list[str]:
    if is_missing(value):
        return []

    # Feather restores list-valued columns as NumPy arrays.
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        for separator in (",", ";", "|"):
            if separator in stripped:
                return [
                    token.strip()
                    for token in stripped.split(separator)
                    if token.strip()
                ]
        return [stripped]

    return [str(value).strip()]


def single_token(value: object) -> list[str]:
    if is_missing(value):
        return []
    token = str(value).strip()
    return [token] if token else []


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_label(value: object) -> str:
    if is_missing(value):
        return ""
    text = str(value).strip().strip("\"'").strip().lower()
    return " ".join(text.split())


def clean_text(value: object) -> str:
    if is_missing(value):
        return ""

    text = unescape(str(value)).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def as_float_array(values: Any, n_rows: int) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    elif hasattr(values, "to_numpy"):
        values = values.to_numpy()

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.shape[0] != n_rows:
        raise RuntimeError(
            f"Expected {n_rows} rows after preprocessing, got {array.shape[0]}"
        )
    return array
