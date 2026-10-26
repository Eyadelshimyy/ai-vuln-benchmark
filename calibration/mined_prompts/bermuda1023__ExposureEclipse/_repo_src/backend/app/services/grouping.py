"""Dataset-group combination methods — CONTRACTS.md §4 / CALCULATION_RULES.md
§"Dataset Group: Max Across Perils at Current Viewed Grain".

These helpers operate on facts already drawn from the member EDMs of a group;
the caller decides which facts belong to which member (typically via
``fact.dataset_id``). The combination math itself only needs the peril and
the grain.

The default for a multi-peril group is
:attr:`CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN` (CLAUDE.md rule 3).
Summing across peril EDMs is allowed ONLY when the user has confirmed the
EDMs are distinct exposure segments (``distinct_segments_confirmed=True``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from app.models.enums import CombinationMethod
from app.models.exposure import ExposureFactNormalized

from .calculations import _grain_key  # internal reuse — same grain semantics


def _per_peril_tiv(
    facts: Iterable[ExposureFactNormalized], grain: Sequence[str]
) -> dict[tuple, dict[Any, float]]:
    """{grain_key: {peril: Σ TIV}} — the per-peril buckets every combiner builds on."""
    buckets: dict[tuple, dict[Any, float]] = defaultdict(lambda: defaultdict(float))
    for f in facts:
        key = _grain_key(f, grain)
        buckets[key][f.peril] += float(f.tiv or 0.0)
    return buckets


def combine_at_grain(
    facts: Iterable[ExposureFactNormalized],
    grain: Sequence[str],
    method: CombinationMethod,
    distinct_segments_confirmed: bool = False,
    base_dataset_id: str | None = None,
) -> dict[tuple, float]:
    """Combine TIV across member EDMs per ``grain`` key.

    * ``MAX_ACROSS_PERILS_AT_VIEW_GRAIN`` — per grain key, ``max(Σ TIV per peril)``.
      Each peril's TIV is first summed within the key (multiple member EDMs
      sharing a peril would be added together — that's still one peril) and the
      maximum across distinct perils wins. This is the default and obeys
      CLAUDE.md rule 3.
    * ``SUM_DISTINCT_SEGMENTS`` — Σ all TIV; raises ``ValueError`` unless
      ``distinct_segments_confirmed=True``.
    * ``SELECTED_EDM_AS_BASE`` — uses ``base_dataset_id`` (kwarg) as the
      exposure base; only facts from that dataset contribute TIV. The other
      EDMs are still in the group for peril views, but they don't move the
      headline TIV. Raises ``ValueError`` if ``base_dataset_id`` is not set.
    * ``KEEP_PERILS_SEPARATE`` — no combination; returns one entry per
      ``(grain_key + (peril,))``.
    * ``CUSTOM`` — reserved v2; raises ``NotImplementedError``.
    """
    if method == CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN:
        buckets = _per_peril_tiv(facts, grain)
        return {key: max(per_peril.values()) for key, per_peril in buckets.items() if per_peril}

    if method == CombinationMethod.SUM_DISTINCT_SEGMENTS:
        if not distinct_segments_confirmed:
            raise ValueError(
                "SUM_DISTINCT_SEGMENTS requires distinct_segments_confirmed=True "
                "(CLAUDE.md rule 3 — never silently sum across peril EDMs)."
            )
        out: dict[tuple, float] = defaultdict(float)
        for f in facts:
            out[_grain_key(f, grain)] += float(f.tiv or 0.0)
        return dict(out)

    if method == CombinationMethod.SELECTED_EDM_AS_BASE:
        if not base_dataset_id:
            raise ValueError(
                "SELECTED_EDM_AS_BASE requires base_dataset_id (the EDM whose TIV "
                "is the exposure base for the group)."
            )
        out = defaultdict(float)
        for f in facts:
            if f.dataset_id == base_dataset_id:
                out[_grain_key(f, grain)] += float(f.tiv or 0.0)
        return dict(out)

    if method == CombinationMethod.KEEP_PERILS_SEPARATE:
        # One entry per (grain_key + peril). Sum TIV within (key, peril) so
        # multiple member EDMs sharing a peril don't appear as duplicates.
        out_kp: dict[tuple, float] = defaultdict(float)
        for f in facts:
            out_kp[_grain_key(f, grain) + (f.peril,)] += float(f.tiv or 0.0)
        return dict(out_kp)

    if method == CombinationMethod.CUSTOM:
        raise NotImplementedError("CombinationMethod.CUSTOM is reserved for v2.")

    raise ValueError(f"Unknown CombinationMethod: {method!r}")


def location_count_at_max_peril(
    facts: Iterable[ExposureFactNormalized],
    grain: Sequence[str],
) -> dict[tuple, int]:
    """Location count under max-across-perils per CALCULATION_RULES.md.

    "Location count under max-across-perils: report the location count of
    the EDM (peril) that supplied the max TIV for that key." We do NOT sum
    counts across perils — that would double-count physical locations.

    Implementation: per ``(grain_key, peril)`` accumulate TIV + location_count;
    for each key keep the count of the peril that contributed the highest TIV.
    """
    # {key: {peril: [tiv_sum, loc_sum]}}
    buckets: dict[tuple, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for f in facts:
        key = _grain_key(f, grain)
        cell = buckets[key][f.peril]
        cell[0] += float(f.tiv or 0.0)
        cell[1] += int(f.location_count or 0)

    out: dict[tuple, int] = {}
    for key, per_peril in buckets.items():
        if not per_peril:
            continue
        winning_peril = max(per_peril, key=lambda p: per_peril[p][0])
        out[key] = int(per_peril[winning_peril][1])
    return out


__all__ = ["combine_at_grain", "location_count_at_max_peril"]
