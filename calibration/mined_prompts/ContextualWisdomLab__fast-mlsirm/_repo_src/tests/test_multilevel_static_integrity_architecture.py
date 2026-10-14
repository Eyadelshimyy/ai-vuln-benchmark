"""Static architecture contracts for multilevel artifact integrity."""

from __future__ import annotations

from pathlib import Path

import fast_mlsirm.multilevel.contracts as contracts


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MULTILEVEL_ROOT = REPOSITORY_ROOT / "python" / "fast_mlsirm" / "multilevel"


def test_integrity_guards_are_defined_in_the_canonical_contract_module() -> None:
    """Public guards and builders must not depend on runtime class mutation."""

    guarded_members = (
        contracts.ContextMembership._assert_integrity,
        contracts.ContextMembershipDesign._assert_integrity,
        contracts.TemporalOccasion._assert_integrity,
        contracts.LongitudinalStateSpec._assert_integrity,
        contracts.LongitudinalDesign._assert_integrity,
        contracts.build_context_membership_design,
        contracts.build_longitudinal_design,
    )

    assert all(member.__module__ == contracts.__name__ for member in guarded_members)


def test_multilevel_package_has_no_runtime_integrity_installer() -> None:
    """The package must expose static source, not an import-time monkeypatch layer."""

    assert not (MULTILEVEL_ROOT / "_integrity.py").exists()
    init_source = (MULTILEVEL_ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "install_contract_integrity" not in init_source
