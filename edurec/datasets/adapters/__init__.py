"""Model-specific adapters over canonical processed data."""

from edurec.datasets.adapters.base import AdapterMetadata, ModelDataAdapter
from edurec.datasets.adapters.proposed import ProposedAdapter
from edurec.datasets.adapters.registry import create_adapter, register_adapter

__all__ = [
    "AdapterMetadata",
    "ModelDataAdapter",
    "ProposedAdapter",
    "create_adapter",
    "register_adapter",
]
