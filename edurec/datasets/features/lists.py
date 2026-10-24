"""List-valued metadata stays as tokens for graph membership relations."""

from typing import Any

import pandas as pd

from edurec.datasets.features.parsing import coerce_list_tokens


def normalize_lists(frame: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    for col in dict.fromkeys([*schema.get("list", []), *schema.get("graph_list", [])]):
        if col in result:
            result[col] = result[col].map(coerce_list_tokens)
    return result


def graph_field_kinds(schema: dict[str, Any]) -> dict[str, bool]:
    """Scalar categories and list attributes share the same graph/text contract."""
    return {
        **{col: False for col in [*schema.get("bin", []), *schema.get("cat", [])]},
        **{
            col: True
            for col in [*schema.get("list", []), *schema.get("graph_list", [])]
        },
    }
