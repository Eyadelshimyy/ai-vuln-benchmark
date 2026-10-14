"""Entity linking: resolve enrichment candidates to canonical reference entities.

- :mod:`womblex.link.normalise` — minimal name/address normalisation.
- :mod:`womblex.link.reference` — bundle-aware reference register loading.
- :mod:`womblex.link.matcher` — generic record-linkage matcher.
- :mod:`womblex.link.stage` — per-stage ``link_shards`` over a shard dir.
"""

from womblex.link.matcher import Candidate, Link, resolve
from womblex.link.reference import ReferenceEntity, ReferenceTable, load_reference
from womblex.link.stage import LinkStageResult, link_shards

__all__ = [
    "Candidate",
    "Link",
    "LinkStageResult",
    "ReferenceEntity",
    "ReferenceTable",
    "link_shards",
    "load_reference",
    "resolve",
]
