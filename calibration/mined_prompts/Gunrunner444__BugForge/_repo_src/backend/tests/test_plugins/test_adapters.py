"""Tests for the Phase 1 adapter registry, language adapters, and AI providers."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.adapters.languages import PythonAdapter
from app.adapters.languages.base import LanguageAdapter
from app.adapters.reports import LocalReportProvider
from app.adapters.scope import ManualScopeProvider
from app.analysis.engine import StaticAnalysisEngine
from app.analysis.python_analyzer import MutableDefaultArgumentRule
from app.analyzers.language_detector import detect_languages, language_for_path
from app.core.config import Settings
from app.domain.evidence import Evidence, EvidenceBundle, EvidenceKind, EvidenceSource
from app.domain.findings import FindingStatus, SecurityFinding
from app.domain.language import LanguageCapability
from app.domain.scope import ScopeConstraint
from app.plugins import (
    AdapterNotFoundError,
    AdapterRegistry,
    DuplicateAdapterError,
    UnsupportedCapabilityError,
    get_plugin_catalog,
    reset_plugin_catalog,
)
from app.plugins.errors import AdapterNotImplementedError


def make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "ai_provider": "mock",
        "ai_model": "test-model",
        "ai_api_key": "",
        "ai_base_url": "",
        "discovery_mode": "disabled",
        "discovery_languages": "",
        "discovery_excluded_topics": "",
        "discovery_excluded_owners": "",
        "language_analyzers": "",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _fresh_catalog() -> None:
    reset_plugin_catalog()
    yield
    reset_plugin_catalog()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_register_and_retrieve() -> None:
    registry: AdapterRegistry[str] = AdapterRegistry("widget")
    registry.register("alpha", lambda: "A", aliases=("a",))
    assert registry.create("alpha") == "A"
    assert registry.create("A") == "A"
    assert registry.has("alpha")
    assert registry.available_ids() == ["alpha"]


def test_registry_missing_adapter_lists_known() -> None:
    registry: AdapterRegistry[str] = AdapterRegistry("widget")
    registry.register("alpha", lambda: "A")
    with pytest.raises(AdapterNotFoundError, match="Unknown widget 'nope'") as exc:
        registry.create("nope")
    assert "alpha" in str(exc.value)


def test_registry_duplicate_rejected() -> None:
    registry: AdapterRegistry[str] = AdapterRegistry("widget")
    registry.register("alpha", lambda: "A")
    with pytest.raises(DuplicateAdapterError):
        registry.register("alpha", lambda: "B")


def test_builtin_catalog_registers_expected_families() -> None:
    catalog = get_plugin_catalog()
    assert catalog.languages.has("python")
    assert catalog.languages.has("javascript")
    assert catalog.ai_providers.has("mock")
    assert catalog.ai_providers.has("openai")
    assert catalog.ai_providers.has("anthropic")
    assert catalog.ai_providers.has("local")
    assert catalog.ai_providers.has("ollama")  # alias
    assert catalog.report_providers.has("local")
    assert catalog.scope_providers.has("manual")
    assert catalog.evidence_collectors.has("static_analysis")


# ---------------------------------------------------------------------------
# Language adapters
# ---------------------------------------------------------------------------


def test_python_detected_through_adapter_registry() -> None:
    catalog = get_plugin_catalog()
    adapter = catalog.languages.for_path(Path("src/app.py"))
    assert adapter is not None
    assert adapter.language_id == "python"
    assert language_for_path(Path("foo.py")) == "python"
    assert language_for_path(Path("foo.pyi")) == "python"


def test_python_analysis_via_adapter(tmp_path: Path) -> None:
    source = "def foo(x=[]):\n    pass\n"
    path = tmp_path / "mod.py"
    path.write_text(source)
    adapter = PythonAdapter()
    findings = adapter.analyze_file(path, source)
    assert any(f.category == "mutable_default_argument" for f in findings)


def test_python_parse_via_adapter(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_text("import os\n\ndef hello():\n    return 1\n")
    result = PythonAdapter().parse_file(path)
    assert result.language == "python"
    assert any(imp.module == "os" for imp in result.imports)
    assert any(entity.name == "hello" for entity in result.entities)


def test_static_engine_uses_python_adapter(tmp_path: Path) -> None:
    py_file = tmp_path / "code.py"
    py_file.write_text("def foo(x=[]):\n    if x == None: pass\n")
    findings = StaticAnalysisEngine().analyze_repository(tmp_path, [py_file])
    categories = {f.category for f in findings}
    assert "mutable_default_argument" in categories
    assert "comparison_to_none" in categories


def test_static_engine_custom_python_rules(tmp_path: Path) -> None:
    py_file = tmp_path / "code.py"
    py_file.write_text("def foo(x=[]): pass\n")
    findings = StaticAnalysisEngine(rules=[MutableDefaultArgumentRule()]).analyze_repository(
        tmp_path, [py_file]
    )
    assert len(findings) == 1


def test_unsupported_language_fails_cleanly_on_analyze() -> None:
    adapter = get_plugin_catalog().languages.get("markdown")
    assert adapter.supports(LanguageCapability.DETECTION)
    assert not adapter.supports(LanguageCapability.STATIC_ANALYSIS)
    with pytest.raises(UnsupportedCapabilityError, match="markdown"):
        adapter.analyze_file(Path("README.md"), "# hi")


def test_unsupported_language_fails_cleanly_on_parse() -> None:
    adapter = get_plugin_catalog().languages.get("yaml")
    with pytest.raises(UnsupportedCapabilityError, match="yaml"):
        adapter.parse_file(Path("config.yaml"))


def test_unknown_language_path_is_none() -> None:
    assert language_for_path(Path("notes.xyz")) is None
    assert get_plugin_catalog().languages.for_path(Path("notes.xyz")) is None


def test_unknown_language_id_raises() -> None:
    with pytest.raises(AdapterNotFoundError, match="cobol"):
        get_plugin_catalog().languages.get("cobol")


def test_detect_languages_still_counts_through_registry() -> None:
    paths = [Path("a.py"), Path("b.py"), Path("c.ts"), Path("d.js")]
    stats = detect_languages(paths)
    by_lang = {item.language: item.file_count for item in stats}
    assert by_lang["python"] == 2
    assert by_lang["typescript"] == 1
    assert by_lang["javascript"] == 1


def test_language_analyzers_invalid_id_errors() -> None:
    from app.core.config import get_settings

    original = settings_language_analyzers()
    try:
        get_settings.cache_clear()
        # Patch the process-wide settings object used by the engine.
        from app.core import config as config_mod

        object.__setattr__(config_mod.settings, "language_analyzers", "not-a-language")
        with pytest.raises(AdapterNotFoundError, match="not-a-language"):
            StaticAnalysisEngine().analyze_repository(Path("."), [])
    finally:
        from app.core import config as config_mod

        object.__setattr__(config_mod.settings, "language_analyzers", original)


def test_language_analyzers_detection_only_language_errors() -> None:
    from app.core import config as config_mod

    original = config_mod.settings.language_analyzers
    try:
        object.__setattr__(config_mod.settings, "language_analyzers", "markdown")
        with pytest.raises(UnsupportedCapabilityError, match="markdown"):
            StaticAnalysisEngine().analyze_repository(Path("."), [])
    finally:
        object.__setattr__(config_mod.settings, "language_analyzers", original)


def settings_language_analyzers() -> str:
    from app.core import config as config_mod

    return config_mod.settings.language_analyzers


def test_future_language_can_register_without_touching_core() -> None:
    """Example used in docs: add a language by registering an adapter."""

    class ToyAdapter(LanguageAdapter):
        @property
        def language_id(self) -> str:
            return "toy"

        @property
        def display_name(self) -> str:
            return "Toy"

        @property
        def file_extensions(self) -> frozenset[str]:
            return frozenset({".toy"})

        @property
        def capabilities(self) -> frozenset[LanguageCapability]:
            return frozenset({LanguageCapability.DETECTION, LanguageCapability.SOURCE})

    catalog = get_plugin_catalog()
    catalog.languages.register_adapter(ToyAdapter())
    assert catalog.languages.language_id_for_path(Path("hello.toy")) == "toy"


# ---------------------------------------------------------------------------
# AI providers
# ---------------------------------------------------------------------------


def test_ai_provider_selection_mock() -> None:
    from app.ai.factory import create_provider
    from app.ai.mock_provider import MockLLMProvider

    provider = create_provider(make_settings(ai_provider="mock"))
    assert isinstance(provider, MockLLMProvider)


@pytest.mark.asyncio
async def test_mock_ai_provider_still_works() -> None:
    from app.ai.mock_provider import MockLLMProvider
    from app.ai.provider import DebuggingRequest

    provider = MockLLMProvider()
    assert await provider.is_available() is True
    response = await provider.analyze(
        DebuggingRequest(
            project_name="demo",
            repository_path="/tmp/demo",
            failing_tests=[],
            static_findings=[],
            source_files=[],
        )
    )
    assert response.provider == "mock"
    assert response.error is None


def test_openai_and_local_provider_selection() -> None:
    from app.ai.factory import create_provider
    from app.ai.local_provider import LocalAIProvider
    from app.ai.openai_provider import OpenAIProvider

    openai = create_provider(make_settings(ai_provider="openai", ai_api_key="sk-test"))
    assert isinstance(openai, OpenAIProvider)

    local = create_provider(make_settings(ai_provider="local", ai_model="qwen3"))
    assert isinstance(local, LocalAIProvider)
    assert local.model_name == "qwen3"

    ollama = create_provider(make_settings(ai_provider="ollama", ai_model="llama3"))
    assert isinstance(ollama, LocalAIProvider)


def test_missing_ai_provider_fails_cleanly() -> None:
    from app.plugins import get_plugin_catalog

    with pytest.raises(AdapterNotFoundError, match="unknown-ai"):
        get_plugin_catalog().ai_providers.create("unknown-ai")


def test_invalid_ai_provider_config_errors() -> None:
    with pytest.raises((ValueError, ValidationError), match="ai_provider"):
        make_settings(ai_provider="not-a-provider")


def test_mlx_backend_constructs() -> None:
    from app.ai.local_provider import LocalAIProvider

    provider = LocalAIProvider(
        api_key="",
        model="Qwen3.6-35B-A3B-8bit",
        base_url="http://127.0.0.1:8080/v1",
        backend="mlx",
    )
    assert provider.provider_name == "mlx"
    assert provider.capabilities().thinking is True
    assert provider.capabilities().local_execution is True


def test_get_provider_uses_registry() -> None:
    from app.ai import get_provider
    from app.ai.mock_provider import MockLLMProvider

    assert isinstance(get_provider(), MockLLMProvider)


# ---------------------------------------------------------------------------
# Findings / evidence
# ---------------------------------------------------------------------------


def test_hypothesis_is_never_auto_verified() -> None:
    finding = SecurityFinding.from_hypothesis(
        "Possible XSS",
        "The model thinks the input is unsanitized",
    )
    assert finding.status is FindingStatus.POTENTIAL
    assert finding.is_verified is False


def test_verified_finding_requires_evidence() -> None:
    with pytest.raises(ValueError, match="independent"):
        SecurityFinding.verified("RCE", evidence=EvidenceBundle())


def test_composite_evidence_from_static_and_tests() -> None:
    from app.adapters.evidence import default_debugging_collectors
    from app.ai.provider import TestFailureEvidence
    from app.analysis.finding import Finding

    static = Finding(
        category="bare_except",
        severity="medium",
        confidence="high",
        file_path="app.py",
        line=10,
        end_line=12,
        message="Bare except",
        explanation="...",
        analyzer="bare_except",
    )
    failure = TestFailureEvidence(
        node_id="tests/test_app.py::test_foo",
        test_file="tests/test_app.py",
        test_name="test_foo",
        traceback="AssertionError",
        stdout=None,
        stderr=None,
        duration_seconds=0.1,
    )
    bundle = default_debugging_collectors().collect_bundle(
        EvidenceSource(static_findings=[static], test_failures=[failure])
    )
    assert len(bundle.of_kind(EvidenceKind.STATIC_ANALYSIS)) == 1
    assert len(bundle.of_kind(EvidenceKind.TEST_FAILURE)) == 1

    finding = SecurityFinding.potential(
        "Possible bug",
        hypothesis="AI guess",
        evidence=bundle,
    )
    assert finding.status is FindingStatus.POTENTIAL
    assert len(finding.evidence) == 2


def test_local_report_provider_renders_potential_and_verified() -> None:
    from app.domain.trusted_evidence import issue_for_finding

    shell = SecurityFinding.potential("Confirmed", description="shown by tests")
    evidence = EvidenceBundle.from_items(
        [
            issue_for_finding(
                shell,
                Evidence(
                    kind=EvidenceKind.HTTP_RESPONSE,
                    source="lab-http",
                    summary="Reproduced consistently",
                    details="shown",
                ),
                "exec-report",
            )
        ]
    )
    findings = [
        SecurityFinding.from_hypothesis("Guess", "model said so"),
        SecurityFinding.verified("Confirmed", evidence=evidence, description="shown by tests"),
    ]
    report = LocalReportProvider().render(findings)
    assert report.generated_by == "local"
    assert report.destination == "local"
    assert report.submitted_remotely is False
    assert "[potential]" in report.body
    assert "[verified]" in report.body
    assert "Guess" in report.body
    assert "local rendering" in report.body.lower()
    assert "Human review" in report.body
    assert report.metadata["submitted_remotely"] == "false"
    with pytest.raises(AdapterNotImplementedError, match="does not submit"):
        LocalReportProvider().submit(report)


def test_manual_scope_denies_by_default() -> None:
    provider = ManualScopeProvider()
    assert provider.is_in_scope("https://example.com/app") is False


def test_manual_scope_allow_and_exclude() -> None:
    provider = ManualScopeProvider(
        ScopeConstraint(
            allowed_hosts=("*.example.com", "api.other.test"),
            excluded_hosts=("admin.example.com",),
            allowed_methods=("GET", "POST"),
        )
    )
    assert provider.is_in_scope("https://app.example.com/x", method="GET") is True
    assert provider.is_in_scope("https://admin.example.com/x", method="GET") is False
    assert provider.is_in_scope("https://api.other.test/v1", method="DELETE") is False


def test_report_and_scope_come_from_catalog() -> None:
    catalog = get_plugin_catalog()
    report = catalog.report_providers.create("local")
    scope = catalog.scope_providers.create("manual")
    assert isinstance(report, LocalReportProvider)
    assert isinstance(scope, ManualScopeProvider)


# ---------------------------------------------------------------------------
# Future adapter families — registration only
# ---------------------------------------------------------------------------


def test_security_tool_browser_proxy_fuzzer_can_register() -> None:
    from app.adapters.browsers.base import BrowserAdapter, BrowserSnapshot
    from app.adapters.fuzzing.base import FuzzingAdapter, FuzzingTarget
    from app.adapters.proxies.base import ProxyAdapter
    from app.adapters.security_tools.base import SecurityToolAdapter, SecurityToolCapability
    from app.domain.http import HttpExchange
    from app.domain.scope import ScopeConstraint

    class FakeScanner(SecurityToolAdapter):
        @property
        def tool_id(self) -> str:
            return "fake-scanner"

        @property
        def display_name(self) -> str:
            return "Fake Scanner"

        def is_available(self) -> bool:
            return True

        def capabilities(self) -> frozenset[SecurityToolCapability]:
            return frozenset({SecurityToolCapability.PASSIVE_EVIDENCE})

        def collect_passive_evidence(self, *, scope: ScopeConstraint):
            return []

    class FakeBrowser(BrowserAdapter):
        @property
        def adapter_id(self) -> str:
            return "fake-browser"

        def is_available(self) -> bool:
            return False

        async def snapshot(self) -> BrowserSnapshot:
            return BrowserSnapshot(url="https://example.test")

    class FakeProxy(ProxyAdapter):
        @property
        def adapter_id(self) -> str:
            return "fake-proxy"

        def is_available(self) -> bool:
            return True

        def _fetch_captured_exchanges(self):
            return [
                HttpExchange(method="GET", url="https://example.test/"),
            ]

    class FakeFuzzer(FuzzingAdapter):
        @property
        def adapter_id(self) -> str:
            return "fake-fuzzer"

        def capabilities(self) -> frozenset[FuzzingTarget]:
            return frozenset({FuzzingTarget.PARAMETER})

    catalog = get_plugin_catalog()
    catalog.security_tools.register("fake-scanner", FakeScanner)
    catalog.browsers.register("fake-browser", FakeBrowser)
    catalog.proxies.register("fake-proxy", FakeProxy)
    catalog.fuzzers.register("fake-fuzzer", FakeFuzzer)

    assert catalog.security_tools.create("fake-scanner").is_available() is True
    in_scope = ScopeConstraint(allowed_hosts=("example.test",))
    assert catalog.proxies.create("fake-proxy").fetch_exchanges(scope=in_scope)[0].method == "GET"
    with pytest.raises(UnsupportedCapabilityError):
        catalog.fuzzers.create("fake-fuzzer").fuzz(
            target=FuzzingTarget.PARAMETER,
            scope=ScopeConstraint(allowed_hosts=("example.test",), allow_active_testing=True),
            target_url="https://example.test/",
        )
