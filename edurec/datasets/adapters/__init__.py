"""Model-specific adapters over canonical processed data."""

from edurec.datasets.adapters.base import (
    AdapterMetadata,
    ModelDataAdapter,
    create_adapter,
    register_adapter,
)
from edurec.datasets.adapters.proposed import ProposedAdapter

__all__ = [
    "AdapterMetadata",
    "ModelDataAdapter",
    "ProposedAdapter",
    "create_adapter",
    "register_adapter",
]
