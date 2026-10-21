"""Per-dataset download, raw loading and schema.

Importing this package registers every dataset spec with the registry.
"""

from edurec.datasets.registry import register_spec
from edurec.datasets.sources import coco, doris, itm, mars, mooccubex

for spec in (*mars.SPECS, itm.SPEC, doris.SPEC, mooccubex.SPEC, coco.SPEC):
    register_spec(spec)
