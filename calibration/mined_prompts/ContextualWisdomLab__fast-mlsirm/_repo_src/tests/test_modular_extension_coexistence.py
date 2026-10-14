"""Regression contracts for coexisting modular PyO3 feature surfaces."""

from __future__ import annotations

from pathlib import Path

import fast_mlsirm


_REPOSITORY_ROOT = Path(__file__).parents[1]
_ENTRYPOINT = _REPOSITORY_ROOT / "crates" / "fast-mlsirm-py" / "src" / "entrypoint.rs"


def test_shared_library_registers_all_secondary_extension_modules():
    """Adding one bounded Rust surface must not replace existing registrations."""
    source = _ENTRYPOINT.read_text(encoding="utf-8")
    assert 'include!("lib.rs");' in source
    assert "mod ata_bindings;" in source
    assert "mod bifactor_bindings;" in source
    assert "mod rating_range_bindings;" in source
    assert "mod regression_bindings;" in source
    assert "mod rotation_bindings;" in source


def test_package_root_exposes_both_feature_families():
    """Users can access bifactor and rotation APIs in one installed package."""
    expected = {
        "BifactorScoreabilityResult",
        "bifactor_scoreability",
        "bifactor_scoreability_from_logit_slopes",
        "RotationCriterionInfo",
        "RotationSolution",
        "available_rotation_criteria",
        "rotate_factor_loadings",
        "rotation_criterion_value_gradient",
    }
    assert expected.issubset(set(fast_mlsirm.__all__))
    for name in expected:
        assert hasattr(fast_mlsirm, name)
