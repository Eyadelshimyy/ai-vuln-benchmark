"""Boundary coverage for governed RAG calibration assembly."""

from __future__ import annotations

import pytest

from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.rag_calibration import build_rag_facets_calibration_bundle


def test_rag_facets_bundle_rejects_non_triple_execution() -> None:
    """Every calibration execution must preserve request, result, and engine."""
    with pytest.raises(AssessmentSpecError) as caught:
        build_rag_facets_calibration_bundle(((object(), object()),))

    assert caught.value.code == "invalid_rag_calibration_execution"
