"""Phase 1 hardening: language-neutral core, registries, verification, AI, scope."""

from __future__ import annotations

import inspect
import typing
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.adapters.languages.base import LanguageAdapter
from app.adapters.languages.python import PythonAdapter
from app.adapters.languages.registry import LanguageRegistry
from app.ai.local_provider import LocalAIProvider
from app.analysis.engine import StaticAnalysisEngine
from app.analysis.python_analyzer import MutableDefaultArgumentRule
from app.analyzers.python.parser import ParseResult
from app.analyzers.repo_analyzer import FileAnalysisResult
from app.domain.evidence import (
    VERIFICATION_PROVENANCE,
    Evidence,
    EvidenceBundle,
    EvidenceKind,
    EvidenceProvenance,
)
from app.domain.findings import FindingStatus, HumanReviewState, SecurityFinding
from app.domain.language import LanguageCapability
from app.domain.source import LanguageParseResult
from app.domain.trusted_evidence import issue_for_finding
from app.plugins import AdapterNotFoundError, AdapterRegistry, DuplicateAdapterError
from app.plugins.errors import (
    ActiveTestingNotPermittedError,
    AdapterConflictError,
    AdapterNotImplementedError,
    OutOfScopeError,
    UnsupportedCapabilityError,
)


def _toy_adapter(
    language_id: str = "toy",
    extensions: frozenset[str] | None = None,
) -> LanguageAdapter:
    ext = extensions if extensions is not None else frozenset({".toy"})

    class ToyAdapter(LanguageAdapter):
        @property
        def language_id(self) -> str:
            return language_id

        @property
        def display_name(self) -> str:
            return language_id.title()

        @property
        def file_extensions(self) -> frozenset[str]:
            return ext

        @property
        def capabilities(self) -> frozenset[LanguageCapability]:
            return frozenset({LanguageCapability.DETECTION})

    return ToyAdapter()


# ---------------------------------------------------------------------------
# Language-neutral parse contract
# ---------------------------------------------------------------------------


def test_language_adapter_parse_contract_is_language_neutral() -> None:
    hints = typing.get_type_hints(LanguageAdapter.parse_file)
    assert hints["return"] is LanguageParseResult
    import app.adapters.languages.base as base_mod

    base_src = inspect.getsource(base_mod)
    assert "app.analyzers.python.parser" not in base_src
    assert "from app.analyzers.python" not in base_src
    assert "LanguageParseResult" in base_src


def test_language_adapter_module_does_not_import_python_parser() -> None:
    import app.adapters.languages.base as base_mod
    import app.analysis.engine as engine_mod
    import app.analyzers.repo_analyzer as repo_mod

    for module in (base_mod, repo_mod, engine_mod):
        source = inspect.getsource(module)
        assert "app.analyzers.python.parser" not in source
        assert "from app.analyzers.python" not in source


def test_static_engine_does_not_special_case_python() -> None:
    import app.analysis.engine as engine_mod

    source = inspect.getsource(engine_mod)
    assert "PythonAdapter" not in source
    assert "isinstance" not in inspect.getsource(StaticAnalysisEngine)


def test_file_analysis_result_uses_neutral_parse_type() -> None:
    hints = typing.get_type_hints(FileAnalysisResult)
    assert hints["parse_result"] == LanguageParseResult | None


def test_python_parse_result_satisfies_neutral_contract(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_text("import os\n\ndef hello():\n    return 1\n")
    result = PythonAdapter().parse_file(path)
    assert isinstance(result, LanguageParseResult)
    assert isinstance(result, ParseResult)
    assert result.language == "python"
    assert result.file_path.endswith("mod.py")
    assert result.line_count >= 3
    assert any(imp.module == "os" for imp in result.imports)
    assert any(entity.name == "hello" for entity in result.entities)


# ---------------------------------------------------------------------------
# Static analysis rule overrides
# ---------------------------------------------------------------------------


def test_python_adapter_with_static_rules(tmp_path: Path) -> None:
    path = tmp_path / "code.py"
    source = "def foo(x=[]):\n    if x == None: pass\n"
    path.write_text(source)
    default = PythonAdapter().analyze_file(path, source)
    assert len(default) >= 2
    narrowed = PythonAdapter().with_static_rules([MutableDefaultArgumentRule()])
    findings = narrowed.analyze_file(path, source)
    assert len(findings) == 1
    assert findings[0].category == "mutable_default_argument"


def test_static_engine_custom_rules_still_work(tmp_path: Path) -> None:
    py_file = tmp_path / "code.py"
    py_file.write_text("def foo(x=[]):\n    if x == None: pass\n")
    findings = StaticAnalysisEngine(rules=[MutableDefaultArgumentRule()]).analyze_repository(
        tmp_path, [py_file]
    )
    assert len(findings) == 1
    assert findings[0].category == "mutable_default_argument"


# ---------------------------------------------------------------------------
# AdapterRegistry identity
# ---------------------------------------------------------------------------


def test_registry_duplicate_canonical_rejected() -> None:
    registry: AdapterRegistry[str] = AdapterRegistry("widget")
    registry.register("alpha", lambda: "A")
    with pytest.raises(DuplicateAdapterError, match="already registered"):
        registry.register("alpha", lambda: "B")


def test_registry_canonical_conflicts_with_existing_alias() -> None:
    registry: AdapterRegistry[str] = AdapterRegistry("widget")
    registry.register("alpha", lambda: "A", aliases=("a",))
    with pytest.raises(DuplicateAdapterError, match="already an alias"):
        registry.register("a", lambda: "B")


def test_registry_alias_conflicts_with_existing_canonical() -> None:
    registry: AdapterRegistry[str] = AdapterRegistry("widget")
    registry.register("alpha", lambda: "A")
    with pytest.raises(DuplicateAdapterError, match="conflicts with canonical"):
        registry.register("beta", lambda: "B", aliases=("alpha",))


def test_registry_alias_conflicts_with_existing_alias() -> None:
    registry: AdapterRegistry[str] = AdapterRegistry("widget")
    registry.register("alpha", lambda: "A", aliases=("shared",))
    with pytest.raises(DuplicateAdapterError, match="already registered for"):
        registry.register("beta", lambda: "B", aliases=("shared",))


def test_registry_does_not_silently_overwrite_alias() -> None:
    registry: AdapterRegistry[str] = AdapterRegistry("widget")
    registry.register("alpha", lambda: "A", aliases=("a",))
    with pytest.raises(DuplicateAdapterError):
        registry.register("beta", lambda: "B", aliases=("a",))
    assert registry.create("a") == "A"


def test_registry_unregister_and_aliases() -> None:
    registry: AdapterRegistry[str] = AdapterRegistry("widget")
    registry.register("alpha", lambda: "A", aliases=("a",))
    registry.unregister("alpha")
    assert registry.has("alpha") is False
    assert registry.has("a") is False
    assert registry.available_ids() == []
    with pytest.raises(AdapterNotFoundError, match="Unknown widget"):
        registry.create("a")


def test_registry_replace_true_swaps_factory_and_aliases() -> None:
    registry: AdapterRegistry[str] = AdapterRegistry("widget")
    registry.register("alpha", lambda: "A", aliases=("a",))
    registry.register("alpha", lambda: "B", aliases=("aa",), replace=True)
    assert registry.create("alpha") == "B"
    assert registry.create("aa") == "B"
    with pytest.raises(AdapterNotFoundError):
        registry.create("a")


def test_registry_normalized_ids() -> None:
    registry: AdapterRegistry[str] = AdapterRegistry("widget")
    registry.register("  Alpha ", lambda: "A", aliases=("A-1",))
    assert registry.has("ALPHA")
    assert registry.create("alpha") == "A"
    assert registry.resolve_id("A-1") == "alpha"
    assert registry.available_ids() == ["alpha"]


def test_registry_unknown_lookup_lists_known() -> None:
    registry: AdapterRegistry[str] = AdapterRegistry("widget")
    registry.register("alpha", lambda: "A")
    with pytest.raises(AdapterNotFoundError, match="alpha") as exc:
        registry.create("missing")
    assert "missing" in str(exc.value)


def test_registry_replace_failure_preserves_previous_state() -> None:
    class BoomRegistry(AdapterRegistry[str]):
        fail_next = False

        def _commit_registration(
            self,
            canonical: str,
            factory: object,
            alias_keys: tuple[str, ...],
            *,
            description: str,
            replace: bool,
        ) -> None:
            if self.fail_next:
                raise RuntimeError("install failed")
            super()._commit_registration(
                canonical,
                factory,  # type: ignore[arg-type]
                alias_keys,
                description=description,
                replace=replace,
            )

    registry = BoomRegistry("widget")
    registry.register("alpha", lambda: "A", aliases=("a",))
    registry.fail_next = True
    with pytest.raises(RuntimeError, match="install failed"):
        registry.register("alpha", lambda: "B", aliases=("b",), replace=True)
    assert registry.create("alpha") == "A"
    assert registry.create("a") == "A"
    with pytest.raises(AdapterNotFoundError):
        registry.create("b")


# ---------------------------------------------------------------------------
# LanguageRegistry replace
# ---------------------------------------------------------------------------


def test_language_replace_drops_stale_extensions() -> None:
    registry = LanguageRegistry()
    registry.register_adapter(_toy_adapter("toy", frozenset({".toy", ".old"})))
    assert registry.language_id_for_path(Path("x.old")) == "toy"
    registry.register_adapter(_toy_adapter("toy", frozenset({".toy"})), replace=True)
    assert registry.language_id_for_path(Path("x.toy")) == "toy"
    assert registry.language_id_for_path(Path("x.old")) is None
    assert registry.get("toy").file_extensions == frozenset({".toy"})


def test_language_replace_rejects_empty_id_without_losing_previous() -> None:
    registry = LanguageRegistry()
    original = _toy_adapter("toy", frozenset({".toy"}))
    registry.register_adapter(original)
    with pytest.raises(ValueError, match="non-empty"):
        registry.register_adapter(_toy_adapter("  ", frozenset({".toy"})), replace=True)
    assert registry.get("toy") is original
    assert registry.language_id_for_path(Path("a.toy")) == "toy"


def test_language_extension_conflict_leaves_original() -> None:
    registry = LanguageRegistry()
    registry.register_adapter(PythonAdapter())
    with pytest.raises(AdapterConflictError, match=".py"):
        registry.register_adapter(_toy_adapter("toy", frozenset({".py"})))
    assert registry.language_id_for_path(Path("mod.py")) == "python"


def test_language_register_rollback_on_install_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = LanguageRegistry()
    original = _toy_adapter("toy", frozenset({".toy"}))
    registry.register_adapter(original)

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("install failed")

    monkeypatch.setattr(AdapterRegistry, "register", boom)
    with pytest.raises(RuntimeError, match="install failed"):
        registry.register_adapter(_toy_adapter("toy", frozenset({".toy", ".new"})), replace=True)
    assert registry.get("toy") is original
    assert registry.language_id_for_path(Path("a.toy")) == "toy"
    assert registry.language_id_for_path(Path("a.new")) is None


# ---------------------------------------------------------------------------
# Evidence / verification invariant
# ---------------------------------------------------------------------------


def _repro_evidence() -> Evidence:
    return Evidence(
        kind=EvidenceKind.REPRODUCTION,
        source="reproduction_engine",
        summary="Reproduced on two runs",
        metadata={"outcome": "reproduced", "reproduced": "true", "execution_id": "repro-1"},
    )


def _http_evidence() -> Evidence:
    return Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="lab-http",
        summary="response showed the payload",
        details="uid=0",
        metadata={"execution_id": "verify-1"},
    )


def test_potential_finding() -> None:
    finding = SecurityFinding.potential("Possible XSS", hypothesis="unsanitized input")
    assert finding.status is FindingStatus.POTENTIAL
    assert finding.is_verified is False
    assert finding.human_review_state is HumanReviewState.UNREVIEWED


def test_rejected_finding() -> None:
    finding = SecurityFinding.rejected("Not a bug", evidence=[_repro_evidence()])
    assert finding.status is FindingStatus.REJECTED
    assert finding.is_verified is False


def test_valid_verified_finding() -> None:
    with pytest.raises(ValueError, match="independent|server-issued"):
        SecurityFinding.verified("Confirmed XSS", evidence=[_repro_evidence()])
    shell = SecurityFinding.potential("Confirmed XSS")
    finding = SecurityFinding.verified(
        "Confirmed XSS", evidence=[issue_for_finding(shell, _http_evidence(), "verify-1")]
    )
    assert finding.status is FindingStatus.VERIFIED
    assert finding.is_verified is True
    assert finding.evidence.verifying_items()


def test_invalid_verified_finding_without_evidence() -> None:
    with pytest.raises(ValueError, match="independent|server-issued"):
        SecurityFinding.verified("RCE", evidence=EvidenceBundle())


def test_ai_only_evidence_rejected_for_verification() -> None:
    claim = "The model thinks this is XSS"
    evidence = Evidence.from_ai(claim)
    assert evidence.provenance is EvidenceProvenance.AI_HYPOTHESIS
    assert evidence.contributes_to_verification is False
    with pytest.raises(ValueError, match="independent"):
        SecurityFinding.verified("XSS", evidence=[evidence])


def test_ai_claim_create_evidence_verified_sequence_is_rejected() -> None:
    ai_claim = "SQL injection in /search"
    evidence = Evidence.from_ai(ai_claim)
    with pytest.raises(ValueError, match="independent"):
        SecurityFinding.verified("SQLi", evidence=[evidence])


def test_static_and_source_evidence_cannot_verify() -> None:
    static = Evidence(
        kind=EvidenceKind.STATIC_ANALYSIS,
        source="static",
        summary="Possible issue",
    )
    source = Evidence(
        kind=EvidenceKind.SOURCE_CODE,
        source="repository",
        summary="app.py excerpt",
    )
    assert static.provenance not in VERIFICATION_PROVENANCE
    with pytest.raises(ValueError, match="independent"):
        SecurityFinding.verified("Maybe", evidence=[static, source])


def test_status_cannot_be_mutated_in_place() -> None:
    finding = SecurityFinding.potential("Guess")
    with pytest.raises(FrozenInstanceError):
        finding.status = FindingStatus.VERIFIED  # type: ignore[misc]


def test_ai_evidence_cannot_be_relabeled_as_execution() -> None:
    evidence = Evidence.from_ai("hypothesis text")
    with pytest.raises(FrozenInstanceError):
        evidence.provenance = EvidenceProvenance.EXECUTION  # type: ignore[misc]


def test_verify_transition_requires_independent_evidence() -> None:
    finding = SecurityFinding.from_hypothesis("Guess", "model said so")
    with pytest.raises(ValueError, match="independent"):
        finding.verify([Evidence.from_ai("still just a guess")])
    with pytest.raises(ValueError, match="independent"):
        finding.verify([_repro_evidence()])
    verified = finding.verify([issue_for_finding(finding, _http_evidence(), "verify-1")])
    assert verified.status is FindingStatus.VERIFIED
    assert finding.status is FindingStatus.POTENTIAL


def test_reproduce_and_human_accept_transitions() -> None:
    finding = SecurityFinding.potential("Possible IDOR")
    with pytest.raises(ValueError, match="requires"):
        finding.reproduce()
    reproduced = finding.reproduce([_repro_evidence()])
    assert reproduced.status is FindingStatus.REPRODUCED
    with pytest.raises(ValueError, match="reproduced or independently verified"):
        finding.human_accept()
    accepted = reproduced.human_accept()
    assert accepted.status is FindingStatus.HUMAN_ACCEPTED
    assert accepted.human_review_state is HumanReviewState.ACCEPTED
    assert accepted.is_verified is True
    rejected = finding.reject()
    with pytest.raises(ValueError, match="Rejected"):
        rejected.reproduce([_repro_evidence()])
    with pytest.raises(ValueError, match="Rejected"):
        rejected.human_accept()


def test_screenshot_and_log_are_explicit_provenance() -> None:
    shot = Evidence(kind=EvidenceKind.SCREENSHOT, source="browser", summary="page.png")
    log = Evidence(kind=EvidenceKind.LOG, source="zap", summary="scan log")
    assert shot.provenance is EvidenceProvenance.SCREENSHOT
    assert log.provenance is EvidenceProvenance.LOG
    assert shot.provenance in VERIFICATION_PROVENANCE
    assert log.provenance in VERIFICATION_PROVENANCE


def test_rejected_cannot_be_verified() -> None:
    finding = SecurityFinding.rejected("Nope")
    with pytest.raises(ValueError, match="Rejected"):
        finding.verify([_repro_evidence()])


def test_human_review_does_not_change_verification() -> None:
    finding = SecurityFinding.potential("Guess").with_review(HumanReviewState.IN_REVIEW)
    assert finding.status is FindingStatus.POTENTIAL
    assert finding.human_review_state is HumanReviewState.IN_REVIEW


# ---------------------------------------------------------------------------
# Generic AI abstraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_complete_and_capabilities() -> None:
    from app.ai.mock_provider import MockLLMProvider
    from app.ai.provider import CompletionRequest

    provider = MockLLMProvider()
    caps = provider.capabilities()
    assert caps.chat is True
    assert caps.structured_output is True
    assert caps.tool_calls is True
    assert caps.thinking_can_disable is True
    response = await provider.complete(
        CompletionRequest(system_prompt="sys", user_message="hello", json_mode=True)
    )
    assert response.provider == "mock"
    assert response.error is None
    assert response.content


@pytest.mark.asyncio
async def test_openai_and_anthropic_advertise_distinct_capabilities() -> None:
    from app.ai.anthropic_provider import AnthropicProvider
    from app.ai.openai_provider import OpenAIProvider

    openai = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini", max_tokens=111)
    anthropic = AnthropicProvider(api_key="ant-test", model="claude", max_tokens=222)
    assert openai.capabilities().structured_output is True
    assert openai.capabilities().max_output_tokens == 111
    assert openai.capabilities().thinking is False
    assert anthropic.capabilities().max_output_tokens == 222
    assert anthropic.capabilities().tool_calls is False
    assert await openai.is_available() is True
    assert await anthropic.is_available() is True


# ---------------------------------------------------------------------------
# LocalAIProvider availability
# ---------------------------------------------------------------------------


def _local_provider(**kwargs: object) -> LocalAIProvider:
    defaults: dict[str, object] = {
        "api_key": "unused",
        "model": "llama3",
        "base_url": "http://127.0.0.1:11434/v1",
    }
    defaults.update(kwargs)
    return LocalAIProvider(**defaults)  # type: ignore[arg-type]


def _patch_models_get(response: object | Exception):
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    if isinstance(response, Exception):
        mock_client.get = AsyncMock(side_effect=response)
    else:
        mock_client.get = AsyncMock(return_value=response)
    return patch("app.ai.health.httpx.AsyncClient", return_value=mock_client)


@pytest.mark.asyncio
async def test_local_provider_available_when_reachable() -> None:
    import httpx

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"id": "llama3"}]}
    with _patch_models_get(mock_response):
        provider = _local_provider()
        assert await provider.is_available() is True
        health = await provider.health()
    assert health.reachable is True
    assert health.configured is True
    assert "unused" not in str(health)


@pytest.mark.asyncio
async def test_local_provider_unavailable_when_server_down() -> None:
    import httpx

    with _patch_models_get(httpx.ConnectError("connection refused")):
        provider = _local_provider()
        assert await provider.is_available() is False
        health = await provider.health()
    assert health.reachable is False
    assert health.configured is True
    assert health.error is not None
    assert "unused" not in health.error


@pytest.mark.asyncio
async def test_local_provider_invalid_base_url() -> None:
    provider = _local_provider(base_url="not-a-url", api_key="super-secret-key")
    assert await provider.is_available() is False
    health = await provider.health()
    assert health.configured is False
    assert health.reachable is False
    assert health.error == "invalid_or_missing_base_url"
    assert "super-secret-key" not in str(health)
    assert health.error is not None and "super-secret-key" not in health.error


@pytest.mark.asyncio
async def test_local_provider_unconfigured_empty_netloc() -> None:
    provider = _local_provider(base_url="http://")
    assert await provider.is_available() is False
    health = await provider.health()
    assert health.configured is False
    assert health.reachable is False


@pytest.mark.asyncio
async def test_local_provider_openai_compatible_capabilities() -> None:
    provider = _local_provider()
    caps = provider.capabilities()
    assert caps.supports_local_models is True
    assert caps.thinking is False
    assert caps.thinking_can_disable is True
    assert caps.tool_calls is False
    assert caps.local_execution is True


# ---------------------------------------------------------------------------
# Scope / browser / proxy / fuzzer contracts
# ---------------------------------------------------------------------------


def test_in_scope_does_not_imply_active_testing() -> None:
    from app.adapters.scope import ManualScopeProvider
    from app.domain.scope import ScopeConstraint

    provider = ManualScopeProvider(
        ScopeConstraint(allowed_hosts=("example.com",), allowed_methods=("GET",))
    )
    assert provider.is_in_scope("https://example.com/app") is True
    assert provider.is_method_permitted("GET") is True
    assert provider.is_method_permitted("DELETE") is False
    assert provider.is_active_testing_permitted("https://example.com/app") is False
    with pytest.raises(ActiveTestingNotPermittedError):
        provider.require_active_testing("https://example.com/app")


def test_active_testing_requires_explicit_opt_in() -> None:
    from app.adapters.scope import ManualScopeProvider
    from app.domain.scope import ScopeConstraint

    provider = ManualScopeProvider(
        ScopeConstraint(
            allowed_hosts=("example.com",),
            allow_active_testing=True,
            allowed_methods=("GET",),
        )
    )
    assert provider.is_active_testing_permitted("https://example.com/app", method="GET") is True
    provider.require_active_testing("https://example.com/app", method="GET")
    with pytest.raises(OutOfScopeError):
        provider.require_in_scope("https://evil.example")


@pytest.mark.asyncio
async def test_browser_navigate_enforces_scope_before_implementation() -> None:
    from app.adapters.browsers.base import BrowserAdapter
    from app.domain.scope import ScopeConstraint

    class EmptyBrowser(BrowserAdapter):
        @property
        def adapter_id(self) -> str:
            return "empty"

        def is_available(self) -> bool:
            return False

    browser = EmptyBrowser()
    in_scope = ScopeConstraint(allowed_hosts=("example.com",))
    with pytest.raises(ActiveTestingNotPermittedError):
        await browser.navigate("https://example.com/", scope=in_scope)
    allowed = ScopeConstraint(allowed_hosts=("example.com",), allow_active_testing=True)
    with pytest.raises(OutOfScopeError):
        await browser.navigate("https://evil.test/", scope=allowed)
    with pytest.raises(AdapterNotImplementedError):
        await browser.navigate("https://example.com/", scope=allowed)


def test_proxy_fetch_filters_out_of_scope() -> None:
    from app.adapters.proxies.base import ProxyAdapter
    from app.domain.http import HttpExchange
    from app.domain.scope import ScopeConstraint

    class CaptureProxy(ProxyAdapter):
        @property
        def adapter_id(self) -> str:
            return "capture"

        def is_available(self) -> bool:
            return True

        def _fetch_captured_exchanges(self):
            return [
                HttpExchange(method="GET", url="https://example.com/ok"),
                HttpExchange(method="GET", url="https://evil.test/nope"),
            ]

    proxy = CaptureProxy()
    kept = proxy.fetch_exchanges(scope=ScopeConstraint(allowed_hosts=("example.com",)))
    assert len(kept) == 1
    assert kept[0].url.endswith("/ok")


def test_fuzzer_refuses_without_active_testing() -> None:
    from app.adapters.fuzzing.base import FuzzingAdapter, FuzzingTarget
    from app.domain.scope import ScopeConstraint

    class EmptyFuzzer(FuzzingAdapter):
        @property
        def adapter_id(self) -> str:
            return "empty"

    with pytest.raises(OutOfScopeError):
        EmptyFuzzer().fuzz(target=FuzzingTarget.PARAMETER, scope=ScopeConstraint())
    scoped = ScopeConstraint(allowed_hosts=("example.com",))
    with pytest.raises(ActiveTestingNotPermittedError):
        EmptyFuzzer().fuzz(
            target=FuzzingTarget.PARAMETER,
            scope=scoped,
            target_url="https://example.com/",
        )


def test_security_tool_active_scan_requires_authorization() -> None:
    from app.adapters.security_tools.base import SecurityToolAdapter
    from app.domain.scope import ScopeConstraint

    class EmptyScanner(SecurityToolAdapter):
        @property
        def tool_id(self) -> str:
            return "empty"

        @property
        def display_name(self) -> str:
            return "Empty"

        def is_available(self) -> bool:
            return False

    scanner = EmptyScanner()
    with pytest.raises(OutOfScopeError):
        scanner.active_scan(scope=ScopeConstraint(), target="https://example.com")
    scoped = ScopeConstraint(allowed_hosts=("example.com",), allow_active_testing=True)
    with pytest.raises(UnsupportedCapabilityError):
        scanner.active_scan(scope=scoped, target="https://example.com")


def test_local_report_separates_statuses_and_review() -> None:
    from app.adapters.reports import LocalReportProvider

    findings = [
        SecurityFinding.potential("Maybe"),
        SecurityFinding.verified(
            "Yes",
            evidence=[
                issue_for_finding(SecurityFinding.potential("Yes"), _http_evidence(), "verify-1")
            ],
        ).with_review(HumanReviewState.ACCEPTED),
        SecurityFinding.rejected("No"),
    ]
    report = LocalReportProvider().render(findings)
    assert report.potential_count == 1
    assert report.verified_count == 1
    assert report.rejected_count == 1
    assert report.review_counts()[HumanReviewState.ACCEPTED.value] == 1
    assert report.submitted_remotely is False
