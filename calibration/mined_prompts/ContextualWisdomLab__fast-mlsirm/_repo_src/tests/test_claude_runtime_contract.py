"""Prevent canonical guidance from drifting from shipped runtime policy."""

from __future__ import annotations

from pathlib import Path
import inspect
import re
import tomllib

from fast_mlsirm.backend import AUTO_BACKEND_UNAVAILABLE_MESSAGE, resolve_backend
from fast_mlsirm.cli import main
from fast_mlsirm.config import FitConfig


ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_CONTRACT_START = "<!-- BEGIN fast-mlsirm-runtime-contract -->"
_RUNTIME_CONTRACT_END = "<!-- END fast-mlsirm-runtime-contract -->"
_STALE_AUTO_NUMPY_FALLBACK = (
    "transparently falls back to the NumPy reference implementation"
)
_STALE_PARITY_AND_FALLBACK = "parity testing and fallback"
_STALE_NUMPY_DEFAULT = "NumPy reference backend as the default runtime path"
_STALE_CLI_AUTO_FALLBACK = "falls back to NumPy otherwise"
_STALE_CLI_HELP_FALLBACK = "numpy reference fallback"
_STALE_WHEEL_NUMPY_DEFAULT = "Installed wheels can use the NumPy backend by default"
_STALE_OPTIONAL_RUST_BACKEND = "optional Rust backend"
_STALE_OPTIONAL_ACCELERATION = "Rust/PyO3 is optional"
_STALE_OPTIONAL_RUST_CORE = "optional Rust core"
_STALE_AUTO_ACCEPTS_NUMPY = "fit auto backend is not numpy or rust"
_BUYER_FACING_SURFACES = (
    ROOT / "README.md",
    ROOT / "docs" / "commercial_readiness.md",
    ROOT / "docs" / "buyer_demo_storyboard.md",
    ROOT / "python" / "fast_mlsirm" / "cli.py",
    ROOT / "python" / "fast_mlsirm" / "config.py",
    ROOT / "python" / "fast_mlsirm" / "backend.py",
    ROOT / "scripts" / "release_acceptance.py",
    ROOT / "scripts" / "sales_readiness.py",
)


def _runtime_contract(guidance: str) -> dict[str, str]:
    """Return the single machine-readable runtime contract from agent guidance."""
    if guidance.count(_RUNTIME_CONTRACT_START) != 1:
        raise AssertionError("CLAUDE.md must contain exactly one runtime-contract start marker")
    if guidance.count(_RUNTIME_CONTRACT_END) != 1:
        raise AssertionError("CLAUDE.md must contain exactly one runtime-contract end marker")
    payload = guidance.split(_RUNTIME_CONTRACT_START, 1)[1].split(
        _RUNTIME_CONTRACT_END, 1
    )[0]
    match = re.fullmatch(r"\s*```toml\s*(.*?)\s*```\s*", payload, re.DOTALL)
    if match is None:
        raise AssertionError("runtime-contract markers must wrap exactly one toml fence")
    loaded = tomllib.loads(match.group(1))
    if "runtime_contract" not in loaded:
        raise AssertionError("toml fence must define a [runtime_contract] table")
    contract = loaded["runtime_contract"]
    if not all(type(value) is str for value in contract.values()):
        raise AssertionError("runtime_contract values must be strings")
    return contract


def test_claude_machine_runtime_contract_matches_shipped_policy() -> None:
    """Canonical machine guidance must pin package and backend ownership policy."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    guidance = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    contract = _runtime_contract(guidance)

    assert contract == {
        "requires_python": project["project"]["requires-python"],
        "auto_backend": "rust_required",
        "numpy_role": "reference_parity_only",
    }
    assert contract["auto_backend"] == "rust_required"
    assert "fails closed" in (resolve_backend.__doc__ or "")
    assert "never silently" in (resolve_backend.__doc__ or "")
    assert "fast_mlsirm._core" in AUTO_BACKEND_UNAVAILABLE_MESSAGE
    assert "backend='numpy'" in AUTO_BACKEND_UNAVAILABLE_MESSAGE


def test_claude_python_floor_matches_package_metadata() -> None:
    """Agent guidance must advertise the exact supported Python floor."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requires_python = project["project"]["requires-python"]
    guidance = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    assert f'`requires-python = "{requires_python}"`' in guidance
    assert "`>=3.10`" not in guidance


def test_claude_auto_backend_matches_fail_closed_runtime() -> None:
    """Canonical guidance must not advertise a silent NumPy production fallback."""
    guidance = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    normalized = " ".join(guidance.split())

    assert "`auto` fails closed when the compiled Rust core is unavailable" in normalized
    assert "otherwise the numerically-identical NumPy reference" not in normalized


def test_readme_auto_backend_matches_fail_closed_runtime() -> None:
    """Buyer-facing install and CLI guidance must match shipped auto-backend ownership."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    assert _STALE_AUTO_NUMPY_FALLBACK not in normalized
    assert _STALE_PARITY_AND_FALLBACK not in normalized
    assert _STALE_CLI_AUTO_FALLBACK not in normalized
    assert "fails closed when that extension is unavailable" in normalized
    assert "Automatic resolution never silently selects NumPy" in normalized
    assert "fails closed otherwise" in normalized


def test_commercial_readiness_names_rust_as_default_auto_path() -> None:
    """Sales-review copy must not name NumPy as the default runtime path."""
    readiness = (ROOT / "docs" / "commercial_readiness.md").read_text(encoding="utf-8")

    assert _STALE_NUMPY_DEFAULT not in readiness
    assert _STALE_WHEEL_NUMPY_DEFAULT not in readiness
    assert "Rust/PyO3 backend as the default `auto` runtime path" in readiness
    assert "Explicit NumPy reference backend for parity testing only" in readiness
    assert "Installed wheels ship the compiled Rust core" in readiness


def test_prd_current_capabilities_keep_numpy_as_explicit_parity() -> None:
    """Protected-main PRD capabilities must not advertise auto-to-NumPy fallback."""
    prd = (ROOT / "docs" / "PRD.md").read_text(encoding="utf-8")

    assert "NumPy reference/fallback paths and parity tests" not in prd
    assert "explicit NumPy reference/parity path" in prd
    assert "`auto` fails closed without the compiled Rust core" in prd
    assert "parity/fallback where explicitly governed" not in prd


def test_architecture_summaries_do_not_call_auto_a_transparent_fallback() -> None:
    """Architecture summaries must not describe auto resolution as transparent fallback."""
    trd = (ROOT / "docs" / "TRD.md").read_text(encoding="utf-8")
    summary = (ROOT / "docs" / "prd_trd_summary.md").read_text(encoding="utf-8")
    adr = (ROOT / "docs" / "adr" / "0002-rust-first-numerical-ownership.md").read_text(
        encoding="utf-8"
    )

    assert "transparent reference/fallback paths" not in trd
    assert "governed reference/parity paths" in trd
    assert "transparent governed reference/fallback paths" not in summary
    assert "public production backend architecture exposes only `auto` and `rust`" in adr
    assert "`fit_reference` API" in adr
    assert "`fit --reference` CLI mode" in adr
    assert "not a `FitConfig` production choice" in adr
    assert "controlled fallback" not in adr


def test_buyer_facing_surfaces_do_not_advertise_auto_numpy_fallback() -> None:
    """Purchaser-visible install, CLI, API, and sales copy must stay fail-closed."""
    stale_claims = (
        _STALE_AUTO_NUMPY_FALLBACK,
        _STALE_PARITY_AND_FALLBACK,
        _STALE_NUMPY_DEFAULT,
        _STALE_CLI_AUTO_FALLBACK,
        _STALE_CLI_HELP_FALLBACK,
        _STALE_WHEEL_NUMPY_DEFAULT,
        _STALE_OPTIONAL_RUST_BACKEND,
        _STALE_OPTIONAL_ACCELERATION,
        _STALE_OPTIONAL_RUST_CORE,
        _STALE_AUTO_ACCEPTS_NUMPY,
    )
    for path in _BUYER_FACING_SURFACES:
        text = path.read_text(encoding="utf-8")
        for claim in stale_claims:
            assert claim not in text, f"{path} still contains {claim!r}"


def test_fitconfig_source_matches_fail_closed_auto_policy() -> None:
    """Public FitConfig comments must not describe silent NumPy auto fallback."""
    source = inspect.getsource(FitConfig)

    assert _STALE_AUTO_NUMPY_FALLBACK not in source
    assert "fails closed" in source
    assert "never silently selects NumPy" in source


def test_fit_cli_help_names_fail_closed_auto_backend(capsys) -> None:
    """`fast-mlsirm fit --help` must tell a purchaser that auto fails closed."""
    try:
        main(["fit", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("fit --help must exit through argparse SystemExit")

    help_text = " ".join(capsys.readouterr().out.split())
    assert _STALE_CLI_HELP_FALLBACK not in help_text
    assert "fails closed otherwise" in help_text
    assert "use --reference for the explicit NumPy parity path" in help_text


def test_buyer_demo_storyboard_names_fail_closed_rust_owner() -> None:
    """A purchaser walkthrough must not treat Rust as optional acceleration."""
    storyboard = (ROOT / "docs" / "buyer_demo_storyboard.md").read_text(encoding="utf-8")

    assert _STALE_OPTIONAL_ACCELERATION not in storyboard
    assert "NumPy is the reference backend" not in storyboard
    assert "fails closed without `fast_mlsirm._core`" in storyboard
    assert 'Pass `backend=numpy` only for' in storyboard


def test_readme_layout_names_compiled_rust_binding() -> None:
    """Repository layout copy must not call the compiled core optional."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert _STALE_OPTIONAL_RUST_BACKEND not in readme
    assert "PyO3 binding for the compiled Rust backend" in readme


def test_sales_check_import_help_does_not_call_rust_optional() -> None:
    """Sales import help must name the compiled core as a require-rust action."""
    source = (ROOT / "scripts" / "sales_readiness.py").read_text(encoding="utf-8")

    assert _STALE_OPTIONAL_RUST_CORE not in source
    assert "also import fast_mlsirm._core when --require-rust is set" in source


def test_release_acceptance_auto_fit_requires_rust_owner() -> None:
    """Release acceptance must reject NumPy as an automatic fit outcome."""
    source = (ROOT / "scripts" / "release_acceptance.py").read_text(encoding="utf-8")

    assert _STALE_AUTO_ACCEPTS_NUMPY not in source
    assert "fit auto backend must resolve to rust" in source
    assert "def _require_auto_fit_resolved_to_rust(" in source
