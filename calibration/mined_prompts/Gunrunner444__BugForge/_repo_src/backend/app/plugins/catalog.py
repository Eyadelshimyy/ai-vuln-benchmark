"""Central plugin catalog — one registry per adapter family."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.adapters.browsers.base import BrowserAdapter
from app.adapters.evidence.base import EvidenceCollector
from app.adapters.fuzzing.base import FuzzingAdapter
from app.adapters.languages.registry import LanguageRegistry
from app.adapters.proxies.base import ProxyAdapter
from app.adapters.reports.base import ReportProvider
from app.adapters.scope.base import ScopeProvider
from app.adapters.security_tools.base import SecurityToolAdapter
from app.ai.provider import LLMProvider
from app.plugins.registry import AdapterRegistry


@dataclass
class PluginCatalog:
    languages: LanguageRegistry = field(default_factory=LanguageRegistry)
    ai_providers: AdapterRegistry[LLMProvider] = field(
        default_factory=lambda: AdapterRegistry[LLMProvider]("AI provider")
    )
    security_tools: AdapterRegistry[SecurityToolAdapter] = field(
        default_factory=lambda: AdapterRegistry[SecurityToolAdapter]("security tool")
    )
    browsers: AdapterRegistry[BrowserAdapter] = field(
        default_factory=lambda: AdapterRegistry[BrowserAdapter]("browser")
    )
    proxies: AdapterRegistry[ProxyAdapter] = field(
        default_factory=lambda: AdapterRegistry[ProxyAdapter]("proxy")
    )
    fuzzers: AdapterRegistry[FuzzingAdapter] = field(
        default_factory=lambda: AdapterRegistry[FuzzingAdapter]("fuzzer")
    )
    report_providers: AdapterRegistry[ReportProvider] = field(
        default_factory=lambda: AdapterRegistry[ReportProvider]("report provider")
    )
    scope_providers: AdapterRegistry[ScopeProvider] = field(
        default_factory=lambda: AdapterRegistry[ScopeProvider]("scope provider")
    )
    evidence_collectors: AdapterRegistry[EvidenceCollector] = field(
        default_factory=lambda: AdapterRegistry[EvidenceCollector]("evidence collector")
    )
    discovery_engines: AdapterRegistry[object] = field(
        default_factory=lambda: AdapterRegistry[object]("discovery engine")
    )


_catalog: PluginCatalog | None = None


def get_plugin_catalog() -> PluginCatalog:
    """Return the process-wide catalog, registering built-in adapters on first use."""
    global _catalog
    if _catalog is None:
        from app.plugins.bootstrap import register_builtin_adapters

        catalog = PluginCatalog()
        register_builtin_adapters(catalog)
        _catalog = catalog
    return _catalog


def reset_plugin_catalog() -> None:
    """Drop the process-wide catalog. Used by tests."""
    global _catalog
    _catalog = None
