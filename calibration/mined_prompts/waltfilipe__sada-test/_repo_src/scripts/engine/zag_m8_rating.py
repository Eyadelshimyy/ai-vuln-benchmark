"""Zagueiro tri-composite rating — thin wrapper over generic engine."""

from __future__ import annotations

import pandas as pd

from .zag_tri_composite_config import apply_zag_tri_composite_ratings


def apply_zag_m8_ratings(out: pd.DataFrame) -> pd.DataFrame:
    """Attach tri-composite profile scores and α-blended overall rating."""
    return apply_zag_tri_composite_ratings(out)
