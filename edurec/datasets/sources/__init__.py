"""Per-dataset download, preparation, filtering and schema.

Importing this package registers every dataset spec with the registry.
"""

from edurec.datasets.sources import coco, doris, mars, mooccubex
from edurec.datasets.sources.registry import register_spec

for spec in (*mars.SPECS, doris.SPEC, mooccubex.SPEC, coco.SPEC):
    register_spec(spec)
