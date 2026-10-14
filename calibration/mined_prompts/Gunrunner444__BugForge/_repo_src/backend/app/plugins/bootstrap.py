"""Register built-in adapters. Called once when the catalog is first created."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.adapters.evidence.collectors import (
    ReproductionEvidenceCollector,
    SourceCodeEvidenceCollector,
    StaticAnalysisEvidenceCollector,
    TestFailureEvidenceCollector,
)
from app.adapters.languages.known import AUXILIARY_LANGUAGES, PROGRAMMING_LANGUAGE_ADAPTERS
from app.adapters.languages.python import PythonAdapter
from app.adapters.reports.local import LocalReportProvider
from app.adapters.scope.manual import ManualScopeProvider

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.plugins.catalog import PluginCatalog


def _register_security_testing(catalog: PluginCatalog) -> None:
    from app.adapters.browsers.playwright import PlaywrightBrowserAdapter
    from app.adapters.fuzzing.controlled import ControlledFuzzingAdapter
    from app.adapters.proxies.burp import BurpHistoryAdapter
    from app.adapters.proxies.har import HarProxyAdapter
    from app.adapters.security_tools.nuclei import NucleiAdapter
    from app.adapters.security_tools.semgrep import SemgrepAdapter
    from app.adapters.security_tools.zap import ZapAdapter

    catalog.browsers.register(
        "playwright", PlaywrightBrowserAdapter, description="Playwright evidence collection"
    )
    catalog.proxies.register("har", HarProxyAdapter, description="HAR traffic ingestion")
    catalog.proxies.register("burp", BurpHistoryAdapter, description="Burp HTTP history import")
    catalog.security_tools.register("zap", ZapAdapter, description="OWASP ZAP automation/alerts")
    catalog.security_tools.register("nuclei", NucleiAdapter, description="Nuclei template scans")
    catalog.security_tools.register(
        "semgrep", SemgrepAdapter, description="Optional Semgrep JSON ingest"
    )
    catalog.fuzzers.register(
        "controlled",
        ControlledFuzzingAdapter,
        description="Gated fuzzing engine (requires a SecurityTestEngine)",
    )
    _register_discovery(catalog)


def _register_discovery(catalog: PluginCatalog) -> None:
    from app.adapters.discovery.economic import EconomicEngine
    from app.adapters.discovery.external import (
        EchidnaEngine,
        FoundryEngine,
        HalmosEngine,
        MedusaEngine,
        SlitherEngine,
        WakeEngine,
    )
    from app.adapters.discovery.ityfuzz import ItyFuzzEngine
    from app.adapters.discovery.protocol import ProtocolEngine
    from app.adapters.discovery.runtime import RuntimeEngine
    from app.adapters.discovery.runtimes import (
        CargoTestEngine,
        GoTestEngine,
        NativeFuzzEngine,
        SanitizerEngine,
    )
    from app.discovery.builtin import BugforgeStaticEngine

    catalog.discovery_engines.register(
        "bugforge-static", BugforgeStaticEngine, description="In-process static analysis"
    )
    catalog.discovery_engines.register("slither", SlitherEngine, description="Optional Slither")
    catalog.discovery_engines.register("foundry", FoundryEngine, description="Optional Foundry")
    catalog.discovery_engines.register("echidna", EchidnaEngine, description="Optional Echidna")
    catalog.discovery_engines.register("medusa", MedusaEngine, description="Optional Medusa")
    catalog.discovery_engines.register("halmos", HalmosEngine, description="Optional Halmos")
    catalog.discovery_engines.register("wake", WakeEngine, description="Optional Wake")
    catalog.discovery_engines.register(
        "ityfuzz", ItyFuzzEngine, description="Optional local ItyFuzz exploration"
    )
    catalog.discovery_engines.register(
        "bugforge-economic",
        EconomicEngine,
        description="In-process economic observations; not a protocol simulator",
    )
    catalog.discovery_engines.register(
        "bugforge-protocol",
        ProtocolEngine,
        description="In-process protocol graph; not verification",
    )
    catalog.discovery_engines.register(
        "bugforge-runtime",
        RuntimeEngine,
        description="Sandboxed runtime and differential observations",
    )
    catalog.discovery_engines.register("go-test", GoTestEngine, description="Go test and fuzz")
    catalog.discovery_engines.register("cargo-test", CargoTestEngine, description="Cargo test")
    catalog.discovery_engines.register(
        "native-fuzz", NativeFuzzEngine, description="C/C++ coverage fuzzing when a harness exists"
    )
    catalog.discovery_engines.register(
        "sanitizer",
        SanitizerEngine,
        description="Sanitizer-aware execution when clang or cargo exists",
    )


def register_builtin_adapters(catalog: PluginCatalog) -> None:
    _register_languages(catalog)
    _register_ai_providers(catalog)
    _register_reporting(catalog)
    _register_scope(catalog)
    _register_evidence_collectors(catalog)
    _register_security_testing(catalog)


def _register_languages(catalog: PluginCatalog) -> None:
    catalog.languages.register_adapter(PythonAdapter())
    for adapter_cls in PROGRAMMING_LANGUAGE_ADAPTERS:
        catalog.languages.register_adapter(adapter_cls())
    for adapter in AUXILIARY_LANGUAGES:
        catalog.languages.register_adapter(adapter)


def _register_ai_providers(catalog: PluginCatalog) -> None:
    from app.ai.anthropic_provider import AnthropicProvider
    from app.ai.local_provider import LocalAIProvider
    from app.ai.mock_provider import MockLLMProvider
    from app.ai.openai_provider import OpenAIProvider

    def mock_factory(settings: Settings | None = None) -> MockLLMProvider:
        return MockLLMProvider()

    def openai_factory(settings: Settings) -> OpenAIProvider:
        return OpenAIProvider(
            api_key=settings.ai_api_key,
            model=settings.ai_model,
            base_url=settings.ai_base_url,
            max_tokens=settings.ai_max_output_tokens,
            temperature=settings.ai_temperature,
            timeout_seconds=settings.ai_timeout_seconds,
            max_retries=settings.ai_max_retries,
        )

    def anthropic_factory(settings: Settings) -> AnthropicProvider:
        return AnthropicProvider(
            api_key=settings.ai_api_key,
            model=settings.ai_model,
            max_tokens=settings.ai_max_output_tokens,
            temperature=settings.ai_temperature,
            timeout_seconds=settings.ai_timeout_seconds,
            max_retries=settings.ai_max_retries,
        )

    def local_factory(settings: Settings) -> LocalAIProvider:
        return LocalAIProvider.from_settings(settings)

    def mlx_factory(settings: Settings) -> LocalAIProvider:
        from app.ai.mlx_provider import DEFAULT_MLX_MODEL

        model = settings.ai_model
        if not model or model == "gpt-4o-mini":
            model = settings.mlx_model or DEFAULT_MLX_MODEL
        return LocalAIProvider(
            api_key=settings.ai_api_key,
            model=model,
            base_url=settings.ai_base_url or settings.mlx_base_url,
            max_tokens=settings.ai_max_output_tokens,
            temperature=settings.ai_temperature,
            timeout_seconds=max(settings.ai_timeout_seconds, 30),
            max_retries=settings.ai_max_retries,
            backend="mlx",
            provider_name="mlx",
            thinking_enabled=settings.ai_thinking_enabled,
            native_json_mode=settings.ai_native_json_mode,
            max_context_tokens=settings.ai_max_context_tokens,
        )

    catalog.ai_providers.register(
        "mock", mock_factory, aliases=("testing",), description="Deterministic mock provider"
    )
    catalog.ai_providers.register("openai", openai_factory, description="OpenAI chat completions")
    catalog.ai_providers.register(
        "anthropic", anthropic_factory, description="Anthropic Messages API"
    )
    catalog.ai_providers.register(
        "local",
        local_factory,
        aliases=("ollama", "openai_compatible"),
        description="Local OpenAI-compatible models (Ollama, LM Studio)",
    )
    catalog.ai_providers.register(
        "mlx",
        mlx_factory,
        description="Local MLX OpenAI-compatible server (configurable model)",
    )


def _register_reporting(catalog: PluginCatalog) -> None:
    catalog.report_providers.register(
        "local", LocalReportProvider, description="Local markdown report"
    )
    from app.adapters.hackerone.provider import HackerOneProvider

    catalog.report_providers.register(
        "hackerone", HackerOneProvider, description="HackerOne Hacker API reporting"
    )


def _register_scope(catalog: PluginCatalog) -> None:
    catalog.scope_providers.register(
        "manual", ManualScopeProvider, description="Operator-supplied scope"
    )
    from app.adapters.hackerone.provider import HackerOneScopeProvider

    catalog.scope_providers.register(
        "hackerone", HackerOneScopeProvider, description="HackerOne structured scope"
    )


def _register_evidence_collectors(catalog: PluginCatalog) -> None:
    catalog.evidence_collectors.register("static_analysis", StaticAnalysisEvidenceCollector)
    catalog.evidence_collectors.register("test_failure", TestFailureEvidenceCollector)
    catalog.evidence_collectors.register("source_code", SourceCodeEvidenceCollector)
    catalog.evidence_collectors.register("reproduction", ReproductionEvidenceCollector)
