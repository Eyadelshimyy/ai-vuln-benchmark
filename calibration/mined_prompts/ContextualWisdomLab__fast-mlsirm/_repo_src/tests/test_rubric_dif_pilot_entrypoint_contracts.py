"""Public entrypoint compatibility for the governed DIF pilot handoff."""

from __future__ import annotations

import inspect

from fast_mlsirm.dif import (
    logistic_dif,
    logistic_dif_purified,
    mantel_haenszel_dif,
    mantel_haenszel_dif_purified,
    sibtest,
)


def test_dif_handoff_keywords_match_all_documented_observed_score_entrypoints():
    """Every documented binary DIF API must accept the emitted keyword pair."""
    emitted_keywords = {"responses", "group"}

    for function in (
        mantel_haenszel_dif,
        mantel_haenszel_dif_purified,
        logistic_dif,
        logistic_dif_purified,
        sibtest,
    ):
        assert emitted_keywords <= set(inspect.signature(function).parameters)
