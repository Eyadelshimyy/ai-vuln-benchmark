"""In-process static engine. It reuses SecurityAnalysisEngine and does not verify."""

from __future__ import annotations

from pathlib import Path

from app.discovery.capabilities import EngineAvailability, EngineCapability, ResultStatus
from app.discovery.engine import AnalysisRequest, DiscoveryEngine
from app.discovery.results import DynamicFinding, DynamicResult


class BugforgeStaticEngine(DiscoveryEngine):
    @property
    def engine_id(self) -> str:
        return "bugforge-static"

    @property
    def display_name(self) -> str:
        return "BugForge static analysis"

    @property
    def supported_languages(self) -> frozenset[str]:
        from app.plugins import get_plugin_catalog

        return frozenset(
            adapter.language_id for adapter in get_plugin_catalog().languages.security_analyzers()
        )

    def capabilities(self) -> frozenset[EngineCapability]:
        return frozenset({EngineCapability.STATIC_ANALYSIS, EngineCapability.RESULTS_INGESTION})

    def availability(self) -> EngineAvailability:
        return EngineAvailability.AVAILABLE

    def version(self) -> str:
        return "phase27"

    def _analyze_target(self, request: AnalysisRequest) -> DynamicResult:
        from app.security.engine import SecurityAnalysisEngine

        files = _files_for_request(request)
        result = SecurityAnalysisEngine().analyze_repository(request.repo_root, files)
        findings = tuple(
            DynamicFinding(
                detector_id=item.rule_id,
                title=item.title,
                confidence=item.confidence,
                function=item.metadata.get("function", ""),
                contract=item.metadata.get("contract", ""),
                file_path=item.file_path,
                line=item.line,
                description=item.summary,
                status="potential",
            )
            for item in result.observations
            if not request.function or item.metadata.get("function") in {"", request.function}
        )
        return DynamicResult(
            engine=self.engine_id,
            engine_version=self.version(),
            language=request.language,
            target=request.target or str(request.repo_root),
            status=ResultStatus.INGESTED,
            executed=True,
            findings=findings,
            source_file=request.source_file,
            function=request.function,
            contract=request.contract,
            provenance="bugforge_static",
            metadata={"verified": "false", "files": str(result.files_analyzed)},
        )


def foundry_project(root: Path) -> bool:
    return (root / "foundry.toml").is_file()


def _files_for_request(request: AnalysisRequest) -> list[Path]:
    """Collect files for the requested language. A Python request never scans Solidity."""
    from app.plugins import get_plugin_catalog
    from app.plugins.errors import AdapterNotFoundError

    explicit = [request.repo_root / path for path in request.files]
    explicit = [path for path in explicit if path.is_file()]
    if explicit:
        if not request.language:
            return explicit
        return [path for path in explicit if _matches_language(path, request.language)]
    languages = get_plugin_catalog().languages
    suffixes: set[str] = set()
    if request.language:
        try:
            adapter = languages.get(request.language)
        except (KeyError, AdapterNotFoundError):
            return []
        suffixes.update(item.lower() for item in adapter.file_extensions)
        return _walk(request.repo_root, suffixes)
    for adapter in languages.security_analyzers():
        suffixes.update(item.lower() for item in adapter.file_extensions)
    return _walk(request.repo_root, suffixes)


def _matches_language(path: Path, language: str) -> bool:
    from app.plugins import get_plugin_catalog
    from app.plugins.errors import AdapterNotFoundError

    try:
        adapter = get_plugin_catalog().languages.get(language)
    except (KeyError, AdapterNotFoundError):
        return False
    return path.suffix.lower() in {item.lower() for item in adapter.file_extensions}


def _walk(root: Path, extensions: set[str]) -> list[Path]:
    if not extensions or not root.exists():
        return []
    found = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions and ".git" not in path.parts
    ]
    return found[:40]
