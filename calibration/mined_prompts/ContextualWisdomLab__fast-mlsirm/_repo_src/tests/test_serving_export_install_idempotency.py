"""Serving safety installer idempotency regressions."""

from __future__ import annotations

import fast_mlsirm._serving_export_safety as serving_safety
from fast_mlsirm import serving


def test_serving_export_safety_install_is_idempotent() -> None:
    """Repeated installation must not wrap serving entry points again."""
    validate_before = serving._validate_bundle
    export_before = serving.export_serving_bundle

    serving_safety.install(serving)

    assert serving._validate_bundle is validate_before
    assert serving.export_serving_bundle is export_before
