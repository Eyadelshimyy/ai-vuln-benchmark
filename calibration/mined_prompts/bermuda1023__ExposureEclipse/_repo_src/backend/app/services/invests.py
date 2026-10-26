"""ATCF cyclone numbers 90-99 are NHC invest slots.

The live-storm list does not probe or offer them. An id in that range is
not a selectable storm, so a bookmark cannot reopen the old invest bundle.
"""

from __future__ import annotations


def is_invest_id(atcf_id: str) -> bool:
    """True if ``atcf_id`` is an invest slot (cyclone number 90-99)."""
    if len(atcf_id) < 8:
        return False
    try:
        cy = int(atcf_id[2:4])
    except ValueError:
        return False
    return 90 <= cy <= 99


__all__ = ["is_invest_id"]
