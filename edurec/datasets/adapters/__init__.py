"""Read-only views shared by the proposed model, native baselines and RecBole.

Views are built from the canonical :class:`~edurec.datasets.cache.ProcessedData`
and never change interactions, labels or splits. See
:class:`~edurec.datasets.adapters.base.DataAdapter` for the public facade.
"""

from edurec.datasets.adapters.base import DataAdapter

__all__ = ["DataAdapter"]
