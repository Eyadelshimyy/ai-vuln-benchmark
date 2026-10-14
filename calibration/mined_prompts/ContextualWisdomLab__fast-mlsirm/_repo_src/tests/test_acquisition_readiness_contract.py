"""Fail-first public contracts for acquisition/commercial readiness governance.

The readiness verifier may validate an explicitly supplied deal scenario, but it
must not fabricate a monetary target or use a legacy transaction-value label as
proof of product quality.  These tests pin the smallest public CLI boundary
before the implementation and downstream manifest migration are changed.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_sales_readiness():
    """Load the repository sales-readiness script as an importable module."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "sales_readiness.py"
    spec = importlib.util.spec_from_file_location("sales_readiness_acquisition", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_readiness_does_not_fabricate_a_contract_value() -> None:
    """No deal value is present unless the caller explicitly supplies one."""
    parser = _load_sales_readiness().build_parser()

    args = parser.parse_args(["--acceptance", "acceptance_summary.json"])

    assert args.contract_value_krw is None


def test_generic_acquisition_readiness_gate_is_public() -> None:
    """Evidence completeness uses a truthful generic acquisition-readiness flag."""
    parser = _load_sales_readiness().build_parser()

    args = parser.parse_args(
        [
            "--acceptance",
            "acceptance_summary.json",
            "--require-acquisition-readiness",
        ]
    )

    assert args.require_acquisition_readiness is True


def test_cli_help_does_not_present_krw_2b_as_readiness_proof() -> None:
    """The public verifier description must not encode KRW 2B as a quality gate."""
    help_text = _load_sales_readiness().build_parser().format_help()

    assert "KRW 2B review" not in help_text
    assert "Target contract value for this gate" not in help_text


def test_generic_profile_activates_all_acquisition_validators_without_20b(
    tmp_path, monkeypatch
) -> None:
    """The generic gate is complete, price-neutral, and separately identified."""
    module = _load_sales_readiness()
    called: dict[str, dict[str, object]] = {}

    def stub(name):
        def validator(*args, **kwargs):
            called[name] = kwargs
            return [{"name": f"{name}:active", "ok": True}]

        return validator

    for name in (
        "_validate_required_files",
        "_validate_doc_tokens",
        "_validate_public_description_boundary",
        "_validate_acceptance_summary",
        "_validate_dist",
        "_validate_buyer_packet",
        "_validate_benchmark_report",
        "_validate_release_evidence_index",
        "_validate_procurement_due_diligence",
        "_validate_pr_queue_governance",
        "_validate_figma_evidence_sync",
    ):
        monkeypatch.setattr(module, name, stub(name))

    def unexpected_20b(*args, **kwargs):
        raise AssertionError("generic readiness must not invoke the legacy 20B profile")

    monkeypatch.setattr(module, "_validate_20b_product_evidence", unexpected_20b)

    args = argparse.Namespace(
        repo_root=str(tmp_path),
        acceptance=str(tmp_path / "acceptance_summary.json"),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=False,
        require_20b_product=False,
        require_acquisition_readiness=True,
        check_import=False,
        contract_value_krw=None,
        max_acceptance_seconds=None,
        buyer_packet_manifest=None,
        require_buyer_packet=False,
        benchmark_report=None,
        require_benchmark_report=False,
        release_evidence_index=None,
        require_release_evidence_index=False,
        procurement_due_diligence=None,
        require_procurement_due_diligence=False,
        pr_queue_governance=None,
        require_pr_queue_governance=False,
        figma_evidence_sync=None,
        require_figma_evidence_sync=False,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "ok"
    assert manifest["require_acquisition_readiness"] is True
    assert manifest["require_20b_product"] is False
    assert manifest["legacy_20b_compatibility_mode"] is False
    assert manifest["transaction_scenario"] is None
    assert manifest["required_acquisition_validators"] == sorted(
        module.REQUIRED_ACQUISITION_VALIDATORS
    )
    assert {
        f"_validate_{name}" for name in module.REQUIRED_ACQUISITION_VALIDATORS
    }.issubset(called)
    for name in module.REQUIRED_ACQUISITION_VALIDATORS:
        assert called[f"_validate_{name}"]["required"] is True


def test_transaction_value_token_is_legacy_only() -> None:
    """A KRW 2B token belongs to compatibility evidence, not generic readiness."""
    module = _load_sales_readiness()

    assert "KRW 2,000,000,000" not in module.REQUIRED_DOC_TOKENS[
        "docs/enterprise_sales_readiness.md"
    ]
    assert "KRW 2,000,000,000" in module.REQUIRED_20B_DOC_TOKENS[
        "docs/enterprise_sales_readiness.md"
    ]
