import argparse
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path


def _load_sales_readiness():
    script = Path(__file__).resolve().parents[1] / "scripts" / "sales_readiness.py"
    spec = importlib.util.spec_from_file_location("sales_readiness", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps({"backend": "rust"}) if path.name == "fit_summary.json" else "ok"
    path.write_text(content, encoding="utf-8")
    return str(path)


def _write_acceptance(tmp_path: Path, *, include_rust: bool = True) -> Path:
    module = _load_sales_readiness()
    artifacts = tmp_path / "artifacts"
    summary = {
        "status": "ok",
        "out": str(artifacts),
        "total_duration_seconds": 0.25,
        "steps": [
            {
                "command": "simulate",
                "duration_seconds": 0.01,
                "files": {
                    "responses": _touch(artifacts / "simulate" / "responses.npy"),
                    "factors": _touch(artifacts / "simulate" / "item_factor.csv"),
                },
            },
            {
                "command": "fit",
                "backend": "rust",
                "duration_seconds": 0.02,
                "out": str(artifacts / "fit_auto"),
                "files": {"summary": _touch(artifacts / "fit_auto" / "fit_summary.json")},
            },
            {
                "command": "diagnose-fit",
                "duration_seconds": 0.03,
                "files": {"diagnostics": _touch(artifacts / "diagnostics_fit" / "fit_diagnostics.json")},
            },
            {
                "command": "diagnose-dimensions",
                "duration_seconds": 0.04,
                "files": {
                    "diagnostics": _touch(artifacts / "diagnostics_dimensions" / "dimension_diagnostics.json")
                },
            },
            {
                "command": "render-report",
                "duration_seconds": 0.05,
                "files": {"report": _touch(artifacts / "fit_report.html")},
            },
        ],
    }
    if include_rust:
        summary["steps"].append(
            {
                "command": "fit",
                "backend": "rust",
                "duration_seconds": 0.02,
                "out": str(artifacts / "fit_rust"),
                "files": {"summary": _touch(artifacts / "fit_rust" / "fit_summary.json")},
            }
        )
    path = artifacts / "acceptance_summary.json"
    path.write_text(module.json.dumps(summary), encoding="utf-8")
    return path


def _write_required_policy_files(root: Path) -> None:
    module = _load_sales_readiness()
    for relative in module.REQUIRED_POLICY_FILES:
        _touch(root / relative)
    token_docs = {
        "README.md": """
        fast-mlsirm
        Project Status
        """,
        "docs/commercial_readiness.md": """
        Commercial Readiness
        Seller Acceptance Checklist
        Enterprise Sales Gate
        Security
        Support
        Release Gate
        scripts/release_acceptance.py
        scripts/sales_readiness.py
        scripts/build_release_evidence_index.py
        scripts/build_commercial_release.py
        scripts/build_procurement_due_diligence.py
        scripts/build_pr_queue_governance.py
        scripts/build_figma_evidence_sync.py
        """,
        "docs/enterprise_sales_readiness.md": """
        Enterprise Sales Readiness
        KRW 2,000,000,000
        Procurement Evidence
        Customer Acceptance Evidence
        Go/No-Go
        Out of Scope
        """,
        "docs/release_acceptance.md": """
        acceptance_summary.json
        sales_readiness_manifest.json
        release_evidence_index.json
        commercial_release_manifest.json
        procurement_due_diligence_manifest.json
        pr_queue_governance_manifest.json
        figma_evidence_sync_manifest.json
        --require-rust
        """,
    }
    for relative, text in token_docs.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _write_20b_product_files(root: Path, *, code_connect: bool = False) -> None:
    docs = {
        "docs/20b_product_readiness.md": """
        KRW 2,000,000,000
        Buyer-Facing Product Standard
        Product Design Scope
        Data Analytics Scope
        Figma Code Connect
        benchmark_report.html
        release_evidence_index.html
        commercial_release_report.html
        procurement_due_diligence_report.html
        pr_queue_governance_report.html
        figma_evidence_sync_report.html
        Go/No-Go
        """,
        "docs/buyer_demo_storyboard.md": """
        Package Evidence
        Synthetic Data
        Fit Workflow
        Diagnostics Workflow
        Procurement Packet
        """,
        "docs/figma_product_design_packet.md": """
        Figma Code Connect is explicitly out of scope
        Package Evidence
        Procurement Packet
        figma_design_packet.json
        """,
        "docs/roi_evidence_model.md": """
        Driver Metrics
        Required Evidence
        roi_manifest.json
        benchmark_report.json
        Caveats
        """,
        "examples/enterprise_demo/README.md": "enterprise demo evidence",
        "docs/superpowers/specs/2026-07-02-20b-product-readiness-design.md": "design spec",
        "docs/superpowers/plans/2026-07-02-20b-product-readiness.md": "implementation plan",
        "docs/superpowers/specs/2026-07-02-20b-benchmark-evidence-design.md": "benchmark design spec",
        "docs/superpowers/plans/2026-07-02-20b-benchmark-evidence.md": "benchmark implementation plan",
        "docs/superpowers/specs/2026-07-02-20b-release-evidence-index-design.md": "release index design spec",
        "docs/superpowers/plans/2026-07-02-20b-release-evidence-index.md": "release index implementation plan",
        "docs/superpowers/specs/2026-07-03-20b-commercial-release-builder-design.md": "commercial release builder design spec",
        "docs/superpowers/plans/2026-07-03-20b-commercial-release-builder.md": "commercial release builder implementation plan",
        "docs/superpowers/specs/2026-07-03-20b-procurement-due-diligence-design.md": "procurement due diligence design spec",
        "docs/superpowers/plans/2026-07-03-20b-procurement-due-diligence.md": "procurement due diligence implementation plan",
        "docs/superpowers/specs/2026-07-03-20b-pr-queue-governance-design.md": "PR queue governance design spec",
        "docs/superpowers/plans/2026-07-03-20b-pr-queue-governance.md": "PR queue governance implementation plan",
        "docs/superpowers/specs/2026-07-03-20b-figma-evidence-sync-design.md": "Figma evidence sync design spec",
        "docs/superpowers/plans/2026-07-03-20b-figma-evidence-sync.md": "Figma evidence sync implementation plan",
    }
    for relative, text in docs.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    manifests = {
        "examples/enterprise_demo/roi_manifest.json": {
            "contract_value_krw": 2_000_000_000,
            "position": "Procurement evidence model, not a valuation guarantee.",
            "drivers": [{"name": "analyst_hours_saved"}],
            "required_evidence": ["acceptance_summary.json"],
            "go_no_go": {"requires_all_sales_readiness_checks_ok": True},
        },
        "examples/enterprise_demo/benchmark_manifest.json": {
            "benchmark_scope": "Synthetic dense MLS2PLM release-acceptance scenarios.",
            "runtime_budget_seconds": 120,
            "scenarios": [{"name": "small_rust_backend"}],
            "required_artifacts": ["fit_summary.json"],
            "caveats": ["acceptance benchmark contract"],
        },
        "examples/enterprise_demo/figma_design_packet.json": {
            "code_connect": code_connect,
            "mode": "static_product_storyboard",
            "source": "docs/buyer_demo_storyboard.md",
            "figma_artifact_url": "https://www.figma.com/design/example",
            "frames": [{"id": "01-package-evidence"}],
            "handoff": {"product_design_scope": "static buyer workflow"},
        },
        "examples/enterprise_demo/product_completion_manifest.json": {
            "contract_value_krw": 2_000_000_000,
            "scorecard_version": "2026-07-02",
            "checks": [
                {"id": "release_acceptance", "status": "go"},
                {"id": "html_report_csp", "status": "go"},
                {"id": "cli_stack_trace_guard", "status": "go"},
                {"id": "report_table_accessibility", "status": "go"},
                {"id": "figma_buyer_review", "status": "go"},
                {"id": "buyer_evidence_packet", "status": "go"},
                {"id": "buyer_evidence_html_report", "status": "go"},
                {"id": "automated_benchmark_report", "status": "go"},
                {"id": "release_evidence_index", "status": "go"},
                {"id": "commercial_release_builder", "status": "go"},
                {"id": "procurement_due_diligence", "status": "go"},
                {"id": "pr_queue_governance", "status": "go"},
                {"id": "figma_evidence_sync", "status": "go"},
            ],
            "go_no_go": {"requires_all_checks_go": True},
        },
    }
    for relative, payload in manifests.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_sales_readiness_manifest_passes_with_acceptance_evidence(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    out = tmp_path / "sales_readiness_manifest.json"
    args = argparse.Namespace(
        repo_root=".",
        acceptance=str(acceptance),
        out=str(out),
        dist=None,
        require_rust=True,
        require_20b_product=False,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "ok"
    assert manifest["contract_value_krw"] == 2_000_000_000
    assert out.exists()


def test_sales_readiness_fails_when_explicit_rust_evidence_is_missing(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path, include_rust=False)
    args = argparse.Namespace(
        repo_root=".",
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=False,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "acceptance:explicit_rust_fit" in failed_names


def test_project_version_parser_supports_python_310_fallback():
    module = _load_sales_readiness()

    version = module._parse_project_version(
        """
        [build-system]
        requires = ["maturin"]

        [project]
        name = "fast-mlsirm"
        version = "0.1.0"
        """
    )

    assert version == "0.1.0"


def test_sales_readiness_passes_with_20b_product_evidence(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "ok"
    assert manifest["require_20b_product"] is True
    check_names = {check["name"] for check in manifest["checks"]}
    assert "20b:figma_code_connect_disabled" in check_names


def test_sales_readiness_fails_when_readme_leaks_internal_commercial_language(
    tmp_path,
):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    readme = repo_root / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + "\n## Commercial Readiness\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=False,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed = {check["name"] for check in manifest["failed_checks"]}
    assert "public_boundary:README.md" in failed


def test_sales_readiness_fails_when_20b_artifact_is_missing(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    (repo_root / "docs" / "20b_product_readiness.md").unlink()
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "20b:file:docs/20b_product_readiness.md" in failed_names


def test_sales_readiness_fails_when_figma_code_connect_is_enabled(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root, code_connect=True)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "20b:figma_code_connect_disabled" in failed_names


def test_sales_readiness_fails_when_optional_figma_url_is_not_a_design_file(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    path = repo_root / "examples" / "enterprise_demo" / "figma_design_packet.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["figma_artifact_url"] = "https://www.figma.com/board/example"
    path.write_text(json.dumps(payload), encoding="utf-8")
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    figma_url_check = next(
        check for check in manifest["failed_checks"] if check["name"] == "20b:figma_artifact_url"
    )
    assert "https://www.figma.com/design/" in figma_url_check["detail"]
    assert "/board/" in figma_url_check["detail"]


def test_sales_readiness_fails_when_completion_scorecard_is_not_go(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    path = repo_root / "examples" / "enterprise_demo" / "product_completion_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"][1]["status"] = "blocked"
    path.write_text(json.dumps(payload), encoding="utf-8")
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    scorecard_check = next(
        check for check in manifest["failed_checks"] if check["name"] == "20b:completion_scorecard"
    )
    assert scorecard_check["non_go"] == ["html_report_csp"]


def _write_buyer_packet_manifest(
    tmp_path: Path,
    *,
    zip_sha: str | None = None,
    report_sha: str | None = None,
) -> Path:
    packet_zip = tmp_path / "buyer_packet.zip"
    with zipfile.ZipFile(packet_zip, "w") as packet:
        packet.writestr("buyer_evidence_manifest.json", "{}")
    actual_sha = hashlib.sha256(packet_zip.read_bytes()).hexdigest()
    html_report = tmp_path / "buyer_evidence_report.html"
    html_report.write_text("<!doctype html><title>Buyer Evidence Review</title>", encoding="utf-8")
    actual_report_sha = hashlib.sha256(html_report.read_bytes()).hexdigest()
    manifest = {
        "status": "ok",
        "contract_value_krw": 2_000_000_000,
        "artifact_count": 12,
        "coverage": {
            "acceptance_summary": True,
            "sales_readiness_manifest": True,
            "wheel": True,
            "sdist": True,
            "product_docs": True,
            "product_manifests": True,
            "acceptance_artifacts": True,
            "html_report": True,
        },
        "report_file": str(html_report),
        "report_sha256": report_sha or actual_report_sha,
        "zip_file": str(packet_zip),
        "zip_sha256": zip_sha or actual_sha,
    }
    path = tmp_path / "buyer_evidence_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_benchmark_report(tmp_path: Path, *, html_sha: str | None = None, budget_ok: bool = True) -> Path:
    html_report = tmp_path / "benchmark_report.html"
    html_report.write_text("<!doctype html><title>Benchmark Evidence Report</title>", encoding="utf-8")
    actual_html_sha = hashlib.sha256(html_report.read_bytes()).hexdigest()
    report = {
        "status": "ok" if budget_ok else "failed",
        "runtime_budget_seconds": 120,
        "total_duration_seconds": 0.25 if budget_ok else 121,
        "budget_ok": budget_ok,
        "scenario_coverage": {
            "required_backends": ["auto", "rust"],
            "observed_backends": ["auto", "rust"],
            "missing_backends": [],
        },
        "artifact_coverage": {
            "required": ["fit_summary.json"],
            "present": ["fit_summary.json"],
            "missing": [],
            "missing_paths": [],
        },
        "html_report_file": str(html_report),
        "html_report_sha256": html_sha or actual_html_sha,
    }
    path = tmp_path / "benchmark_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _write_release_evidence_index(tmp_path: Path, *, html_sha: str | None = None) -> Path:
    html_report = tmp_path / "release" / "release_evidence_index.html"
    html_report.parent.mkdir(parents=True, exist_ok=True)
    html_report.write_text("<!doctype html><title>Release Evidence Index</title>", encoding="utf-8")
    actual_html_sha = hashlib.sha256(html_report.read_bytes()).hexdigest()
    manifest = {
        "status": "ok",
        "contract_value_krw": 2_000_000_000,
        "coverage": {
            "acceptance_summary": True,
            "sales_readiness_manifest": True,
            "benchmark_report": True,
            "benchmark_html_report": True,
            "buyer_packet_manifest": True,
            "buyer_packet_zip": True,
            "buyer_packet_html_report": True,
            "wheel": True,
            "sdist": True,
        },
        "dist": {
            "artifacts": [
                {"kind": "wheel", "size_bytes": 5, "sha256": "a" * 64},
                {"kind": "sdist", "size_bytes": 5, "sha256": "b" * 64},
            ]
        },
        "failures": [],
        "html_report_file": str(html_report),
        "html_report_sha256": html_sha or actual_html_sha,
    }
    path = tmp_path / "release" / "release_evidence_index.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_procurement_due_diligence(
    tmp_path: Path,
    *,
    html_sha: str | None = None,
    status: str = "ok",
) -> Path:
    html_report = tmp_path / "procurement" / "procurement_due_diligence_report.html"
    html_report.parent.mkdir(parents=True, exist_ok=True)
    html_report.write_text("<!doctype html><title>Procurement Due Diligence</title>", encoding="utf-8")
    actual_html_sha = hashlib.sha256(html_report.read_bytes()).hexdigest()
    manifest = {
        "status": status,
        "contract_value_krw": 2_000_000_000,
        "checks": [
            {"name": "dist:wheel", "category": "package", "ok": True},
            {"name": "policy_file:README.md", "category": "policy", "ok": True},
            {"name": "commercial_release:status", "category": "commercial_release", "ok": True},
            {"name": "github:snapshot", "category": "github", "ok": True},
        ],
        "failed_checks": [] if status == "ok" else [{"name": "status", "category": "commercial_release", "ok": False}],
        "html_report_file": str(html_report),
        "html_report_sha256": html_sha or actual_html_sha,
    }
    path = tmp_path / "procurement" / "procurement_due_diligence_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_pr_queue_governance(
    tmp_path: Path,
    *,
    html_sha: str | None = None,
    status: str = "ok",
) -> Path:
    html_report = tmp_path / "pr-queue" / "pr_queue_governance_report.html"
    html_report.parent.mkdir(parents=True, exist_ok=True)
    html_report.write_text("<!doctype html><title>PR Queue Governance</title>", encoding="utf-8")
    actual_html_sha = hashlib.sha256(html_report.read_bytes()).hexdigest()
    manifest = {
        "status": status,
        "contract_value_krw": 2_000_000_000,
        "open_pr_count": 3,
        "risk_counts": {
            "changes_requested": 1,
            "stale": 1,
            "release_scope_conflict": 1,
            "review_or_check_delay": 1,
        },
        "checks": [
            {"name": "github:snapshot", "category": "github", "ok": True},
            {"name": "queue:classified", "category": "queue_state", "ok": True},
            {"name": "risk:coverage", "category": "risk_classification", "ok": True},
            {"name": "release:boundary", "category": "release_boundary", "ok": True},
        ],
        "failed_checks": [] if status == "ok" else [{"name": "github:snapshot", "category": "github", "ok": False}],
        "html_report_file": str(html_report),
        "html_report_sha256": html_sha or actual_html_sha,
    }
    path = tmp_path / "pr-queue" / "pr_queue_governance_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_figma_evidence_sync(
    tmp_path: Path,
    *,
    html_sha: str | None = None,
    status: str = "ok",
) -> Path:
    html_report = tmp_path / "figma-sync" / "figma_evidence_sync_report.html"
    html_report.parent.mkdir(parents=True, exist_ok=True)
    html_report.write_text("<!doctype html><title>Figma Evidence Sync</title>", encoding="utf-8")
    actual_html_sha = hashlib.sha256(html_report.read_bytes()).hexdigest()
    manifest = {
        "status": status,
        "contract_value_krw": 2_000_000_000,
        "code_connect": False,
        "frame_coverage": {"missing": []},
        "required_token_coverage": {"missing": []},
        "checks": [
            {"name": "figma:packet", "category": "figma_packet", "ok": True},
            {"name": "figma:code_connect_disabled", "category": "figma_policy", "ok": True},
            {"name": "figma:frame_coverage", "category": "figma_frames", "ok": True},
            {"name": "figma:required_tokens", "category": "figma_tokens", "ok": True},
        ],
        "failed_checks": [] if status == "ok" else [{"name": "figma:packet", "category": "figma_packet", "ok": False}],
        "html_report_file": str(html_report),
        "html_report_sha256": html_sha or actual_html_sha,
    }
    path = tmp_path / "figma-sync" / "figma_evidence_sync_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_sales_readiness_validates_required_buyer_packet(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    packet_manifest = _write_buyer_packet_manifest(tmp_path)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        buyer_packet_manifest=str(packet_manifest),
        require_buyer_packet=True,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "ok"
    check_names = {check["name"] for check in manifest["checks"]}
    assert "buyer_packet:coverage" in check_names
    assert "buyer_packet:zip_sha256" in check_names
    assert "buyer_packet:html_report_sha256" in check_names


def test_sales_readiness_validates_required_benchmark_report(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    benchmark_report = _write_benchmark_report(tmp_path)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        buyer_packet_manifest=None,
        require_buyer_packet=False,
        benchmark_report=str(benchmark_report),
        require_benchmark_report=True,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "ok"
    check_names = {check["name"] for check in manifest["checks"]}
    assert "benchmark_report:runtime_budget" in check_names
    assert "benchmark_report:html_report_sha256" in check_names


def test_sales_readiness_validates_required_release_evidence_index(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    release_index = _write_release_evidence_index(tmp_path)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        buyer_packet_manifest=None,
        require_buyer_packet=False,
        benchmark_report=None,
        require_benchmark_report=False,
        release_evidence_index=str(release_index),
        require_release_evidence_index=True,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "ok"
    check_names = {check["name"] for check in manifest["checks"]}
    assert "release_evidence_index:coverage" in check_names
    assert "release_evidence_index:html_report_sha256" in check_names


def test_sales_readiness_validates_required_procurement_due_diligence(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    procurement = _write_procurement_due_diligence(tmp_path)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        buyer_packet_manifest=None,
        require_buyer_packet=False,
        benchmark_report=None,
        require_benchmark_report=False,
        release_evidence_index=None,
        require_release_evidence_index=False,
        procurement_due_diligence=str(procurement),
        require_procurement_due_diligence=True,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "ok"
    assert manifest["require_procurement_due_diligence"] is True
    check_names = {check["name"] for check in manifest["checks"]}
    assert "procurement_due_diligence:category_coverage" in check_names
    assert "procurement_due_diligence:html_report_sha256" in check_names


def test_sales_readiness_validates_required_pr_queue_governance(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    pr_queue = _write_pr_queue_governance(tmp_path)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        buyer_packet_manifest=None,
        require_buyer_packet=False,
        benchmark_report=None,
        require_benchmark_report=False,
        release_evidence_index=None,
        require_release_evidence_index=False,
        procurement_due_diligence=None,
        require_procurement_due_diligence=False,
        pr_queue_governance=str(pr_queue),
        require_pr_queue_governance=True,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "ok"
    assert manifest["require_pr_queue_governance"] is True
    check_names = {check["name"] for check in manifest["checks"]}
    assert "pr_queue_governance:category_coverage" in check_names
    assert "pr_queue_governance:html_report_sha256" in check_names


def test_sales_readiness_validates_required_figma_evidence_sync(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    figma_sync = _write_figma_evidence_sync(tmp_path)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
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
        figma_evidence_sync=str(figma_sync),
        require_figma_evidence_sync=True,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "ok"
    assert manifest["require_figma_evidence_sync"] is True
    check_names = {check["name"] for check in manifest["checks"]}
    assert "figma_evidence_sync:category_coverage" in check_names
    assert "figma_evidence_sync:html_report_sha256" in check_names


def test_sales_readiness_fails_when_benchmark_report_sha_mismatches(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    benchmark_report = _write_benchmark_report(tmp_path, html_sha="0" * 64)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        buyer_packet_manifest=None,
        require_buyer_packet=False,
        benchmark_report=str(benchmark_report),
        require_benchmark_report=True,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "benchmark_report:html_report_sha256" in failed_names


def test_sales_readiness_fails_when_release_evidence_index_sha_mismatches(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    release_index = _write_release_evidence_index(tmp_path, html_sha="0" * 64)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        buyer_packet_manifest=None,
        require_buyer_packet=False,
        benchmark_report=None,
        require_benchmark_report=False,
        release_evidence_index=str(release_index),
        require_release_evidence_index=True,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "release_evidence_index:html_report_sha256" in failed_names


def test_sales_readiness_fails_when_procurement_due_diligence_sha_mismatches(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    procurement = _write_procurement_due_diligence(tmp_path, html_sha="0" * 64)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        buyer_packet_manifest=None,
        require_buyer_packet=False,
        benchmark_report=None,
        require_benchmark_report=False,
        release_evidence_index=None,
        require_release_evidence_index=False,
        procurement_due_diligence=str(procurement),
        require_procurement_due_diligence=True,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "procurement_due_diligence:html_report_sha256" in failed_names


def test_sales_readiness_fails_when_pr_queue_governance_sha_mismatches(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    pr_queue = _write_pr_queue_governance(tmp_path, html_sha="0" * 64)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        buyer_packet_manifest=None,
        require_buyer_packet=False,
        benchmark_report=None,
        require_benchmark_report=False,
        release_evidence_index=None,
        require_release_evidence_index=False,
        procurement_due_diligence=None,
        require_procurement_due_diligence=False,
        pr_queue_governance=str(pr_queue),
        require_pr_queue_governance=True,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "pr_queue_governance:html_report_sha256" in failed_names


def test_sales_readiness_fails_when_figma_evidence_sync_sha_mismatches(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    figma_sync = _write_figma_evidence_sync(tmp_path, html_sha="0" * 64)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
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
        figma_evidence_sync=str(figma_sync),
        require_figma_evidence_sync=True,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "figma_evidence_sync:html_report_sha256" in failed_names


def test_sales_readiness_fails_when_buyer_packet_sha_mismatches(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    packet_manifest = _write_buyer_packet_manifest(tmp_path, zip_sha="0" * 64)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        buyer_packet_manifest=str(packet_manifest),
        require_buyer_packet=True,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "buyer_packet:zip_sha256" in failed_names


def test_sales_readiness_fails_when_buyer_packet_report_sha_mismatches(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    packet_manifest = _write_buyer_packet_manifest(tmp_path, report_sha="0" * 64)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        buyer_packet_manifest=str(packet_manifest),
        require_buyer_packet=True,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "buyer_packet:html_report_sha256" in failed_names


def test_sales_readiness_fails_gracefully_when_20b_json_has_wrong_shape(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    (repo_root / "examples" / "enterprise_demo" / "roi_manifest.json").write_text("[]", encoding="utf-8")
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "20b:json_shape:examples/enterprise_demo/roi_manifest.json" in failed_names


def test_sales_readiness_treats_required_20b_json_none_as_empty(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    path = repo_root / "examples" / "enterprise_demo" / "roi_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["drivers"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    json_field_check = next(
        check
        for check in manifest["failed_checks"]
        if check["name"] == "20b:json_fields:examples/enterprise_demo/roi_manifest.json"
    )
    assert json_field_check["empty"] == ["drivers"]
