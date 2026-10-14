"""Regression coverage for strict non-finite bounded JSON admission."""

from __future__ import annotations

import pytest

from fast_mlsirm.io import _load_json_bounded


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_load_json_bounded_rejects_nonfinite_constants_by_default(tmp_path, literal):
    """Default artifact JSON rejects non-standard non-finite constants."""
    artifact = tmp_path / "nonfinite.json"
    artifact.write_text(f'{{"value": {literal}}}', encoding="utf-8")

    with pytest.raises(
        ValueError, match="contains a non-finite JSON numeric value"
    ) as exc_info:
        _load_json_bounded(artifact, source="artifact JSON")

    assert literal not in str(exc_info.value)


def test_load_json_bounded_preserves_explicit_parse_constant_policy(tmp_path):
    """An explicit caller constant policy remains the intentional escape hatch."""
    artifact = tmp_path / "custom-constant.json"
    artifact.write_text('{"value": NaN}', encoding="utf-8")
    observed: list[str] = []

    def parse_constant(literal: str) -> str:
        observed.append(literal)
        return "custom"

    assert _load_json_bounded(
        artifact,
        source="artifact JSON",
        parse_constant=parse_constant,
    ) == {"value": "custom"}
    assert observed == ["NaN"]
