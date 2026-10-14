"""Security analysis agent: static findings → context → local AI hypothesis.

The AI never creates a verified finding. Output stays potential or
corroborated static hypothesis.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.ai.provider import CompletionRequest, LLMProvider
from app.ai.structured import StructuredParseError, extract_json_object
from app.domain.findings import SecurityFinding
from app.security.context import SecurityContextBuilder
from app.security.engine import SecurityAnalysisEngine, SecurityScanResult
from app.security.findings import attach_ai_hypothesis
from app.security.prompts import build_security_user_message, security_system_prompt

logger = logging.getLogger(__name__)

_HYPOTHESIS_KEYS = ("title", "hypothesis", "not_verified")


class SecurityAnalysisAgent:
    """Orchestrate repository security analysis with optional AI enrichment."""

    def __init__(
        self,
        *,
        engine: SecurityAnalysisEngine | None = None,
        provider: LLMProvider | None = None,
        context_builder: SecurityContextBuilder | None = None,
    ) -> None:
        self._engine = engine or SecurityAnalysisEngine()
        self._provider = provider
        self._context = context_builder or SecurityContextBuilder()

    def scan(self, repo_path: Path, file_paths: list[Path] | None = None) -> SecurityScanResult:
        paths = file_paths if file_paths is not None else _walk_files(repo_path)
        return self._engine.analyze_repository(repo_path, paths)

    async def analyze(
        self,
        repo_path: Path,
        file_paths: list[Path] | None = None,
        *,
        use_ai: bool = True,
    ) -> SecurityScanResult:
        result = self.scan(repo_path, file_paths)
        if not use_ai or not result.findings:
            return result
        provider = self._provider
        if provider is None:
            from app.ai.factory import create_provider
            from app.core.config import settings

            provider = create_provider(settings)
        caps = provider.capabilities()
        if not (caps.chat or caps.text_generation or caps.structured_generation):
            logger.info("AI provider cannot generate text; skipping hypothesis enrichment")
            return result

        enriched: list[SecurityFinding] = []
        graphs_abs = result.graphs
        for cluster, finding in zip(result.clusters, result.findings, strict=False):
            context = self._context.build(
                repo_root=repo_path,
                cluster=cluster,
                graphs=graphs_abs,
                frameworks=result.frameworks,
            )
            request = CompletionRequest(
                system_prompt=security_system_prompt(),
                user_message=build_security_user_message(context),
                json_mode=True,
                thinking=False,
            )
            try:
                response = await provider.complete(request)
            except Exception as exc:
                logger.warning("Security AI complete failed: %s", exc)
                enriched.append(finding)
                continue
            if response.error or not response.content:
                enriched.append(finding)
                continue
            try:
                payload = extract_json_object(response.content, required_keys=_HYPOTHESIS_KEYS)
            except StructuredParseError as exc:
                logger.info("Discarding malformed security AI output: %s", exc)
                enriched.append(finding)
                continue
            if payload.get("not_verified") is False:
                # Model tried to verify; keep the static finding and ignore the claim.
                logger.info("Ignoring AI claim of verification")
            hypothesis = str(payload.get("hypothesis") or "")
            analysis = str(payload.get("reasoning_summary") or hypothesis)
            impact = str(payload.get("impact") or "") or None
            title = str(payload.get("title") or finding.title)
            updated = attach_ai_hypothesis(
                finding,
                hypothesis=hypothesis or finding.hypothesis or title,
                analysis=analysis,
                impact=impact,
            )
            enriched.append(updated)

        result.findings = enriched
        return result


def _walk_files(repo_path: Path) -> list[Path]:
    from app.analyzers.repo_analyzer import RepoAnalyzer

    analysis = RepoAnalyzer().analyze(repo_path)
    return [fr.absolute_path for fr in analysis.file_results]
