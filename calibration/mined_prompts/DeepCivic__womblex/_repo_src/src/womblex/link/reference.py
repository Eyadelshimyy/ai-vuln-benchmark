"""Bundle-aware reference register consumption for entity linking.

A reference register is a *logical* dataset that may be backed by a bundle
of partner files (the clean example is a shapefile: ``.shp`` + ``.shx`` +
``.dbf`` + ``.prj``; G-NAF is a set of PSVs). The :class:`ReferenceSource`
indirection keeps that seam open — physical loading is delegated to a
format handler that yields a normalised :class:`ReferenceTable` the matcher
consumes.

**v1 implements the CSV/single-file path only.** The interface does not
preclude a future geospatial / multi-file handler, but none is built here.
Which columns play which role (id, name, exact-match, fuzzy-match, parent)
is declared corpus-side via :class:`womblex.config.ReferenceConfig` — the
library knows nothing about specific registers.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from womblex.config import ReferenceConfig
from womblex.link.normalise import normalise_address, normalise_name


@dataclass
class ReferenceEntity:
    """One canonical reference entity with precomputed match keys."""

    entity_id: str
    name: str
    entity_type: str
    parent_id: str
    exact_key: str          # normalised concat of match_exact_cols (e.g. address)
    fuzzy_keys: list[str]    # normalised match_fuzzy values (e.g. legal/trading names)


@dataclass
class ReferenceTable:
    """Resolved reference: entities + curated alias overrides."""

    entities: list[ReferenceEntity]
    aliases: dict[str, str] = field(default_factory=dict)  # normalised alias -> entity_id

    def entity_by_id(self, entity_id: str) -> ReferenceEntity | None:
        for e in self.entities:
            if e.entity_id == entity_id:
                return e
        return None


def load_reference(config: ReferenceConfig) -> ReferenceTable:
    """Load a reference register into a normalised :class:`ReferenceTable`."""
    if config.format != "csv":
        raise NotImplementedError(
            f"reference format {config.format!r} not implemented (v1 supports 'csv')"
        )
    rows = _read_csv(config.path)
    entities: list[ReferenceEntity] = []
    for r in rows:
        entity_id = (r.get(config.id_col) or "").strip()
        if not entity_id:
            continue
        exact_parts = [r.get(c, "") or "" for c in config.match_exact_cols]
        exact_key = normalise_address(" ".join(p for p in exact_parts if p)) if exact_parts else ""
        fuzzy_keys = [
            normalise_name(r.get(c)) for c in config.match_fuzzy_cols if r.get(c)
        ]
        entities.append(ReferenceEntity(
            entity_id=entity_id,
            name=(r.get(config.name_col) or "").strip(),
            entity_type=config.entity_type,
            parent_id=(r.get(config.parent_id_col) or "").strip() if config.parent_id_col else "",
            exact_key=exact_key,
            fuzzy_keys=[k for k in fuzzy_keys if k],
        ))

    aliases = _load_aliases(config.alias_table) if config.alias_table else {}
    return ReferenceTable(entities=entities, aliases=aliases)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _load_aliases(path: Path) -> dict[str, str]:
    """Load a corpus alias table (columns ``alias``, ``entity_id``).

    Aliases cover entities the register doesn't carry (e.g. a prior
    trustee). Keyed on the normalised alias name.
    """
    out: dict[str, str] = {}
    for r in _read_csv(path):
        alias = normalise_name(r.get("alias"))
        eid = (r.get("entity_id") or "").strip()
        if alias and eid:
            out[alias] = eid
    return out
