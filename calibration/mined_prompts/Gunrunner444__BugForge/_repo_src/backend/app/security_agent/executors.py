"""Real tool executors. Authorization is not a result. Stubs are not executable."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import urlparse

from app.adapters.browsers.playwright import PlaywrightBrowserAdapter
from app.adapters.security_tools.nuclei import NucleiAdapter, NucleiTemplatePolicy
from app.adapters.security_tools.zap import ZapAdapter
from app.core.config import get_settings
from app.domain.scope import ScopeConstraint
from app.security_agent.evidence_graph import EvidenceGraph
from app.security_agent.injection import untrusted_observation
from app.security_agent.reproduction import ReproductionAction, ReproductionEngine, ReproductionPlan
from app.security_agent.states import (
    EvidenceGraphKind,
    ResearchMode,
    ToolCapability,
    ToolResultQuality,
)
from app.security_agent.tools import Executor, ToolRegistry, with_capability
from app.security_testing.api_import import import_api
from app.security_testing.api_tests import APITestRunner
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.errors import AuthorizationDeniedError
from app.security_testing.failures import ToolExecutionResult, ToolExecutionState
from app.security_testing.fuzzing import FuzzingEngine, FuzzLimits, MutationKind, SeedRequest
from app.security_testing.process import ProcessRunner
from app.security_testing.sanitization import wrap_untrusted
from app.security_testing.secrets import redact_text


@dataclass
class ToolContext:
    engine: SecurityTestEngine
    graph: EvidenceGraph
    project_id: str
    session_id: str
    mode: ResearchMode
    program_handle: str
    repo_root: Path
    exchanges: dict[str, dict[str, Any]] = field(default_factory=dict)
    cancelled: Callable[[], bool] = lambda: False
    zap_runner: ProcessRunner | None = None
    nuclei_runner: ProcessRunner | None = None
    zap_binary: str | None = None
    nuclei_binary: str | None = None
    browser: Any = None
    hypotheses: Callable[[], dict[str, str]] = field(default_factory=lambda: lambda: {})
    exploratory_records: list[Any] = field(default_factory=list)


def bind_engine_tools(ctx: ToolContext, registry: ToolRegistry) -> ToolRegistry:
    registry.bind_executor("http_request", _http_request(ctx))
    registry.bind_executor("browser_navigate", _browser_navigate(ctx))
    registry.bind_executor("source_inspect", _source_inspect(ctx))
    registry.bind_executor("evidence_inspect", _evidence_inspect(ctx))
    registry.bind_executor("zap_scan", _zap_scan(ctx))
    registry.bind_executor("nuclei_scan", _nuclei_scan(ctx))
    registry.bind_executor("fuzz", _fuzz(ctx))
    registry.bind_executor("reproduce", _reproduce(ctx))
    registry.bind_executor("proxy_evidence", _proxy_evidence(ctx))
    registry.bind_executor("api_test", _api_test(ctx))
    registry.bind_executor("exploratory_test", _exploratory_test(ctx))
    _apply_availability(ctx, registry)
    return registry


def _apply_availability(ctx: ToolContext, registry: ToolRegistry) -> None:
    browser = ctx.browser or PlaywrightBrowserAdapter(engine=ctx.engine)
    zap = ZapAdapter(engine=ctx.engine, runner=ctx.zap_runner, binary=ctx.zap_binary)
    nuclei = NucleiAdapter(engine=ctx.engine, runner=ctx.nuclei_runner, binary=ctx.nuclei_binary)
    mapping = {
        "browser_navigate": ToolCapability.EXECUTABLE
        if browser.is_available()
        else ToolCapability.UNAVAILABLE,
        "zap_scan": ToolCapability.EXECUTABLE
        if zap.is_available()
        else ToolCapability.RESULTS_INGESTIBLE,
        "nuclei_scan": ToolCapability.EXECUTABLE
        if nuclei.is_available()
        else ToolCapability.RESULTS_INGESTIBLE,
    }
    for name, capability in mapping.items():
        registry.replace_spec(with_capability(registry.spec(name), capability))


def _http_request(ctx: ToolContext) -> Executor:
    async def run(arguments: dict[str, Any]) -> dict[str, Any]:
        if ctx.cancelled():
            return _quality(ToolResultQuality.BLOCKED, {"reason": "cancelled"})
        exchange = await ctx.engine.http("agent_http").request(
            str(arguments.get("method") or "GET"),
            str(arguments["url"]),
            headers=arguments.get("headers") or None,
            content=arguments.get("content"),
            active=bool(arguments.get("active", True)),
        )
        if isinstance(exchange, ToolExecutionResult):
            return _from_tool_result("http_request", exchange)
        body = wrap_untrusted(
            "http-response", redact_text(str(exchange.response_body or ""))[:1000]
        )
        payload: dict[str, Any] = {
            "quality": ToolResultQuality.SUCCESS.value,
            "status": exchange.response_status,
            "body": body,
            "url": exchange.url,
            "exchange_id": exchange.request_id,
            "scope": exchange.scope_decision,
        }
        ctx.exchanges[exchange.request_id] = _exchange_snapshot(exchange)
        req = ctx.graph.add(
            kind=EvidenceGraphKind.REQUEST.value,
            provenance="http_observation",
            summary=f"{exchange.method} {exchange.url}",
            source="http_request",
            extra={"exchange_id": exchange.request_id, "status": exchange.response_status},
        )
        ctx.graph.add(
            kind=EvidenceGraphKind.RESPONSE.value,
            provenance="http_observation",
            summary=f"status={exchange.response_status}",
            source="http_request",
            extra={"exchange_id": exchange.request_id},
            parent_id=req.id,
            relation="responds_to",
        )
        payload["evidence_ids"] = [req.id]
        return payload

    return run


def _browser_navigate(ctx: ToolContext) -> Executor:
    async def run(arguments: dict[str, Any]) -> dict[str, Any]:
        if ctx.cancelled():
            return _quality(ToolResultQuality.BLOCKED, {"reason": "cancelled"})
        adapter = ctx.browser or PlaywrightBrowserAdapter(engine=ctx.engine)
        adapter.attach_engine(ctx.engine)
        url = str(arguments["url"])
        if not adapter.is_available():
            if ctx.engine.safety.dry_run:
                return {
                    "quality": ToolResultQuality.NO_RESULT.value,
                    "capability": ToolCapability.PLANNING_ONLY.value,
                    "url": url,
                    "executed": False,
                    "reason": "Playwright unavailable; dry-run planning only",
                }
            return {
                "quality": ToolResultQuality.UNAVAILABLE.value,
                "capability": ToolCapability.UNAVAILABLE.value,
                "executed": False,
                "reason": "Playwright is not installed; browser navigation was not executed",
            }
        await adapter.navigate(url)
        if ctx.cancelled():
            return _quality(ToolResultQuality.PARTIAL, {"reason": "cancelled after navigation"})
        evidence = await adapter.capture()
        snapshot = await adapter.snapshot()
        network = [
            {
                "method": item.method,
                "url": redact_text(item.url),
                "status": item.response_status,
                "exchange_id": item.request_id,
            }
            for item in evidence.network[:40]
        ]
        for item in evidence.network:
            ctx.exchanges[item.request_id] = _exchange_snapshot(item)
        node = ctx.graph.add(
            kind=EvidenceGraphKind.BROWSER_OBSERVATION.value,
            provenance="browser_observation",
            summary=f"navigated {redact_text(snapshot.url or url)} title={redact_text(snapshot.title or '')}",
            source="browser_navigate",
            extra={
                "url": redact_text(snapshot.url or url),
                "title": redact_text(snapshot.title or ""),
                "screenshot": snapshot.screenshot_path,
                "network_count": len(network),
                "console_count": len(snapshot.console),
            },
        )
        return {
            "quality": ToolResultQuality.SUCCESS.value,
            "executed": True,
            "url": redact_text(snapshot.url or url),
            "title": redact_text(snapshot.title or ""),
            "network": network,
            "console": [
                untrusted_observation("browser-console", item) for item in snapshot.console[:30]
            ]
            if arguments.get("capture_console", True)
            else [],
            "screenshot": snapshot.screenshot_path
            if arguments.get("capture_screenshot", True)
            else None,
            "evidence_id": node.id,
        }

    return run


def _source_inspect(ctx: ToolContext) -> Executor:
    async def run(arguments: dict[str, Any]) -> dict[str, Any]:
        settings = get_settings()
        rel = str(arguments.get("path") or "").strip()
        if not rel:
            return _quality(ToolResultQuality.FAILED, {"reason": "path required"})
        try:
            path = _safe_repo_path(ctx.repo_root, rel)
        except ValueError as exc:
            return _quality(ToolResultQuality.BLOCKED, {"reason": str(exc)})
        if not path.is_file():
            return {
                "quality": ToolResultQuality.NO_RESULT.value,
                "path": rel,
                "reason": "file not found",
                "untrusted": True,
            }
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = int(arguments.get("start_line") or 1)
        end = int(
            arguments.get("end_line")
            or min(len(lines), start + settings.security_agent_source_max_lines - 1)
        )
        start = max(1, start)
        end = min(len(lines), max(start, end))
        excerpt_lines = lines[start - 1 : end]
        query = str(arguments.get("query") or "").strip()
        if query:
            excerpt_lines = [line for line in excerpt_lines if query.lower() in line.lower()]
        excerpt = "\n".join(excerpt_lines)
        excerpt = redact_text(excerpt)[: settings.security_agent_source_excerpt_bytes]
        labeled = untrusted_observation("source", excerpt)
        node = ctx.graph.add(
            kind=EvidenceGraphKind.SOURCE.value,
            provenance="source_observation",
            summary=f"{rel}:{start}-{end}",
            source="source_inspect",
            extra={"path": rel, "start_line": start, "end_line": end, "query": query},
        )
        return {
            "quality": ToolResultQuality.SUCCESS.value
            if excerpt
            else ToolResultQuality.NO_RESULT.value,
            "path": rel,
            "start_line": start,
            "end_line": end,
            "excerpt": labeled,
            "untrusted": True,
            "query_echoed": False,
            "evidence_id": node.id,
        }

    return run


def _evidence_inspect(ctx: ToolContext) -> Executor:
    async def run(arguments: dict[str, Any]) -> dict[str, Any]:
        evidence_id = str(arguments.get("evidence_id") or "")
        inspected = ctx.graph.inspect(
            evidence_id, session_id=ctx.session_id, project_id=ctx.project_id
        )
        if inspected is None:
            return {
                "quality": ToolResultQuality.NO_RESULT.value,
                "evidence_id": evidence_id,
                "reason": "unknown evidence for this session",
            }
        execution = None
        for parent in inspected.get("parents") or []:
            if (
                isinstance(parent, dict)
                and parent.get("kind") == EvidenceGraphKind.TOOL_EXECUTION.value
            ):
                execution = parent
        return {
            "quality": ToolResultQuality.SUCCESS.value,
            "evidence_id": evidence_id,
            "metadata": {
                "kind": inspected["kind"],
                "provenance": inspected["provenance"],
                "source": inspected["source"],
                "created_at": inspected["created_at"],
            },
            "provenance": inspected["provenance"],
            "parents": inspected["parents"],
            "children": inspected["children"],
            "content": untrusted_observation("evidence", str(inspected.get("summary") or "")),
            "linked_tool_execution": execution,
            "extra": inspected.get("extra") or {},
        }

    return run


def _proxy_evidence(ctx: ToolContext) -> Executor:
    async def run(arguments: dict[str, Any]) -> dict[str, Any]:
        exchange_id = str(arguments.get("exchange_id") or "")
        item = ctx.exchanges.get(exchange_id)
        if item is None:
            return {
                "quality": ToolResultQuality.NO_RESULT.value,
                "exchange_id": exchange_id,
                "reason": "unknown exchange for this session",
            }
        node = ctx.graph.add(
            kind=EvidenceGraphKind.REQUEST.value,
            provenance="http_observation",
            summary=f"proxy {item.get('method')} {item.get('url')}",
            source="proxy_evidence",
            extra={"exchange_id": exchange_id},
        )
        return {
            "quality": ToolResultQuality.SUCCESS.value,
            "exchange_id": exchange_id,
            "request": item.get("request"),
            "response": item.get("response"),
            "scope": item.get("scope"),
            "tool": item.get("tool"),
            "source": item.get("source"),
            "evidence_id": node.id,
        }

    return run


def _zap_scan(ctx: ToolContext) -> Executor:
    async def run(arguments: dict[str, Any]) -> dict[str, Any]:
        if ctx.cancelled():
            return _quality(ToolResultQuality.BLOCKED, {"reason": "cancelled"})
        adapter = ZapAdapter(engine=ctx.engine, runner=ctx.zap_runner, binary=ctx.zap_binary)
        target = str(arguments.get("target") or "")
        if arguments.get("ingest_only") and arguments.get("alerts_json"):
            alerts = adapter.ingest_alerts(str(arguments["alerts_json"]))
            ids = _ingest_scanner_evidence(ctx, "zap", alerts)
            return {
                "quality": ToolResultQuality.RESULTS_AVAILABLE.value,
                "executed": False,
                "ingested": len(alerts),
                "evidence_ids": ids,
            }
        started = monotonic()
        try:
            evidence = list(adapter.active_scan(scope=_constraint(ctx), target=target))
        except AuthorizationDeniedError as exc:
            return _quality(ToolResultQuality.BLOCKED, {"reason": str(exc), "executed": False})
        elapsed = monotonic() - started
        result = adapter.last_result
        ids = _ingest_scanner_evidence(ctx, "zap", evidence, elapsed=elapsed)
        quality = _quality_from_state(result.state if result else ToolExecutionState.FAILED)
        return {
            "quality": quality.value,
            "executed": result is not None
            and result.state
            not in {
                ToolExecutionState.TOOL_UNAVAILABLE,
                ToolExecutionState.PLANNED,
                ToolExecutionState.AUTHORIZED,
                ToolExecutionState.DRY_RUN,
            },
            "state": result.state.value if result else ToolExecutionState.FAILED.value,
            "detail": result.detail if result else "",
            "alerts": len(
                [item for item in evidence if item.kind.value == "scanner" and item.summary]
            ),
            "evidence_ids": ids,
            "scan_seconds": elapsed,
        }

    return run


def _nuclei_scan(ctx: ToolContext) -> Executor:
    async def run(arguments: dict[str, Any]) -> dict[str, Any]:
        if ctx.cancelled():
            return _quality(ToolResultQuality.BLOCKED, {"reason": "cancelled"})
        policy = NucleiTemplatePolicy()
        template = str(arguments.get("template") or "").strip()
        if template:
            allowed = {item.lower() for item in policy.allowed_templates}
            if not allowed or template.lower() not in allowed:
                return _quality(
                    ToolResultQuality.BLOCKED,
                    {
                        "reason": "template not permitted by NucleiTemplatePolicy",
                        "template": template,
                    },
                )
            policy = NucleiTemplatePolicy(
                allowed_templates=(template,),
                allowed_tags=policy.allowed_tags,
                denied_tags=policy.denied_tags,
            )
        adapter = NucleiAdapter(
            engine=ctx.engine,
            policy=policy,
            runner=ctx.nuclei_runner,
            binary=ctx.nuclei_binary,
        )
        target = str(arguments.get("target") or "")
        if arguments.get("ingest_only") and arguments.get("alerts_json"):
            results = adapter.ingest_jsonl(str(arguments["alerts_json"]))
            ids = _ingest_scanner_evidence(ctx, "nuclei", results)
            return {
                "quality": ToolResultQuality.RESULTS_AVAILABLE.value,
                "executed": False,
                "ingested": len(results),
                "evidence_ids": ids,
            }
        started = monotonic()
        try:
            evidence = list(adapter.active_scan(scope=_constraint(ctx), target=target))
        except AuthorizationDeniedError as exc:
            return _quality(ToolResultQuality.BLOCKED, {"reason": str(exc), "executed": False})
        elapsed = monotonic() - started
        result = adapter.last_result
        ids = _ingest_scanner_evidence(ctx, "nuclei", evidence, elapsed=elapsed)
        quality = _quality_from_state(result.state if result else ToolExecutionState.FAILED)
        return {
            "quality": quality.value,
            "executed": bool(
                result
                and result.state
                in {ToolExecutionState.RESULTS_AVAILABLE, ToolExecutionState.RESULTS_INGESTED}
            ),
            "state": result.state.value if result else ToolExecutionState.FAILED.value,
            "detail": result.detail if result else "",
            "results": len([item for item in evidence if item.kind.value == "scanner"]),
            "evidence_ids": ids,
            "scan_seconds": elapsed,
        }

    return run


def _fuzz(ctx: ToolContext) -> Executor:
    async def run(arguments: dict[str, Any]) -> dict[str, Any]:
        if ctx.cancelled():
            return _quality(ToolResultQuality.BLOCKED, {"reason": "cancelled"})
        kinds: list[MutationKind] = []
        for item in arguments.get("kinds") or ["query"]:
            try:
                kinds.append(MutationKind(str(item)))
            except ValueError:
                kinds.append(MutationKind.QUERY)
        requested = FuzzLimits(
            request_limit=int(arguments.get("count") or 3),
            requests_per_second=float(
                arguments.get("requests_per_second") or ctx.engine.safety.limits.requests_per_second
            ),
            concurrency=int(arguments.get("concurrency") or 1),
            timeout_seconds=float(
                arguments.get("timeout_seconds") or ctx.engine.safety.limits.timeout_seconds
            ),
            payload_count=int(arguments.get("count") or 3),
            max_body_size=int(
                arguments.get("max_payload_size") or ctx.engine.safety.limits.max_payload_bytes
            ),
        )
        fuzzer = FuzzingEngine(ctx.engine, limits=requested)
        seed = SeedRequest(
            method=str(arguments.get("method") or "GET"),
            url=str(arguments["url"]),
            headers=tuple((str(k), str(v)) for k, v in (arguments.get("headers") or {}).items()),
            body=arguments.get("content"),
        )
        try:
            result = await fuzzer.fuzz(seed, kinds=tuple(kinds))
        except AuthorizationDeniedError as exc:
            return _quality(ToolResultQuality.BLOCKED, {"reason": str(exc)})
        if isinstance(result, ToolExecutionResult):
            return _from_tool_result("fuzz", result)
        ids = []
        interesting = 0
        for item in result:
            if "Baseline" in item.summary or item.summary.startswith("Stopped"):
                continue
            interesting += 1
            node = ctx.graph.add(
                kind=EvidenceGraphKind.OBSERVATION.value,
                provenance="fuzzing_result",
                summary=item.summary,
                source="fuzz",
            )
            ids.append(node.id)
        return {
            "quality": ToolResultQuality.SUCCESS.value
            if result
            else ToolResultQuality.NO_RESULT.value,
            "interesting": interesting,
            "total_evidence": len(result),
            "not_all_vulnerabilities": True,
            "evidence_ids": ids,
            "limits": {
                "request_limit": fuzzer.limits.request_limit,
                "payload_count": fuzzer.limits.payload_count,
                "rps": fuzzer.limits.requests_per_second,
            },
        }

    return run


def _reproduce(ctx: ToolContext) -> Executor:
    async def run(arguments: dict[str, Any]) -> dict[str, Any]:
        if ctx.cancelled():
            return _quality(ToolResultQuality.BLOCKED, {"reason": "cancelled"})
        actions: list[ReproductionAction] = []
        for raw in arguments.get("actions") or []:
            if not isinstance(raw, dict):
                continue
            actions.append(
                ReproductionAction(
                    method=str(raw.get("method") or "GET"),
                    url=str(raw.get("url") or ""),
                    content=raw.get("content"),
                    expected_status=raw.get("expected_status"),
                    expected_body_contains=raw.get("expected_body_contains"),
                    headers=dict(raw.get("headers") or {}),
                )
            )
        if not actions and arguments.get("url"):
            actions.append(
                ReproductionAction(
                    method=str(arguments.get("method") or "GET"),
                    url=str(arguments["url"]),
                    expected_status=arguments.get("expected_status"),
                    expected_body_contains=arguments.get("expected_body_contains")
                    or arguments.get("expected_result")
                    or None,
                )
            )
        plan = ReproductionPlan(
            preconditions=tuple(arguments.get("preconditions") or ()),
            setup=str(arguments.get("setup") or ""),
            actions=tuple(actions),
            expected_result=str(arguments.get("expected_result") or ""),
            cleanup=str(arguments.get("cleanup") or ""),
            reproducibility_count=int(arguments.get("reproducibility_count") or 1),
            id=str(arguments.get("plan_id") or "") or ReproductionPlan().id,
        )
        done = await ReproductionEngine(ctx.engine).execute(plan)
        node = ctx.graph.add(
            kind=EvidenceGraphKind.REPRODUCTION.value,
            provenance="reproduction",
            summary=f"{done.outcome.value}: {done.actual_result}"[:500],
            source="reproduce",
            extra={"outcome": done.outcome.value, "plan_id": done.id},
        )
        quality = {
            "reproduced": ToolResultQuality.SUCCESS,
            "not_reproduced": ToolResultQuality.NO_RESULT,
            "blocked": ToolResultQuality.BLOCKED,
            "environment_error": ToolResultQuality.FAILED,
            "inconclusive": ToolResultQuality.PARTIAL,
        }.get(done.outcome.value, ToolResultQuality.PARTIAL)
        return {
            **done.snapshot(),
            "quality": quality.value,
            "evidence_id": node.id,
            "status_alone_is_not_success": True,
        }

    return run


def _api_test(ctx: ToolContext) -> Executor:
    async def run(arguments: dict[str, Any]) -> dict[str, Any]:
        if ctx.cancelled():
            return _quality(ToolResultQuality.BLOCKED, {"reason": "cancelled"})
        spec_path = arguments.get("spec_path")
        if not spec_path:
            return _quality(
                ToolResultQuality.FAILED,
                {"reason": "spec_path is required for api_test"},
            )
        try:
            resolved = _safe_repo_path(ctx.repo_root, str(spec_path))
        except ValueError as exc:
            return _quality(ToolResultQuality.BLOCKED, {"reason": str(exc)})
        if not resolved.is_file():
            return _quality(
                ToolResultQuality.NO_RESULT, {"reason": "spec not found", "path": spec_path}
            )
        spec = import_api(resolved, source_format=arguments.get("spec_format"))
        runner = APITestRunner(ctx.engine)
        tests = runner.generator.generate(spec, base_url=arguments.get("url") or spec.base_url)
        tests = tests[: int(arguments.get("max_tests") or 8)]
        if not arguments.get("execute", True):
            return {
                "quality": ToolResultQuality.SUCCESS.value,
                "executed": False,
                "capability": ToolCapability.PLANNING_ONLY.value,
                "spec": spec.title,
                "format": spec.source_format,
                "endpoints": [
                    {"method": item.method, "path": item.path, "summary": item.summary}
                    for item in spec.endpoints[:40]
                ],
                "candidates": [
                    {"title": item.title, "method": item.method, "url": item.url} for item in tests
                ],
            }
        live = bool(arguments.get("execute", True))
        require_human = ctx.mode is ResearchMode.LIVE_HACKERONE
        result = await runner.execute(tests, require_human=require_human)
        if isinstance(result, ToolExecutionResult):
            return _from_tool_result("api_test", result)
        ids = []
        for item in result:
            node = ctx.graph.add(
                kind=EvidenceGraphKind.OBSERVATION.value,
                provenance="api_test",
                summary=item.summary,
                source="api_test",
            )
            ids.append(node.id)
        return {
            "quality": ToolResultQuality.SUCCESS.value
            if result
            else ToolResultQuality.NO_RESULT.value,
            "executed": live,
            "spec": spec.title,
            "format": spec.source_format,
            "tests": len(tests),
            "evidence_ids": ids,
            "used_spec_path": str(spec_path),
        }

    return run


def _safe_repo_path(root: Path, relative: str) -> Path:
    from app.security_agent.repo_lock import safe_source_path
    from app.security_testing.errors import RestrictedActivityError

    try:
        return safe_source_path(root, relative)
    except RestrictedActivityError as exc:
        raise ValueError(str(exc) or "path_traversal") from exc


def _constraint(ctx: ToolContext) -> ScopeConstraint:
    hosts = tuple(rule.identifier for rule in ctx.engine.session.scope.includes) or (
        urlparse(ctx.engine.session.scope.program_name or "localhost").hostname or "localhost",
    )
    return ScopeConstraint(
        allowed_hosts=tuple(str(host) for host in hosts if host),
        allow_active_testing=ctx.engine.session.active_testing_enabled,
        program_name=ctx.engine.session.scope.program_name,
    )


def _exchange_snapshot(exchange: Any) -> dict[str, Any]:
    request_headers = {
        item.name: redact_text(item.value)
        for item in getattr(exchange, "request_headers", ())
        if item.name.lower() not in {"authorization", "cookie"}
    }
    return {
        "id": exchange.request_id,
        "method": exchange.method,
        "url": redact_text(exchange.url),
        "status": exchange.response_status,
        "scope": exchange.scope_decision,
        "tool": exchange.source_tool,
        "source": exchange.source_tool,
        "request": {
            "method": exchange.method,
            "url": redact_text(exchange.url),
            "headers": request_headers,
            "body": redact_text(str(exchange.request_body or ""))[:1000],
        },
        "response": {
            "status": exchange.response_status,
            "body": untrusted_observation(
                "http-response", redact_text(str(exchange.response_body or ""))[:1000]
            ),
        },
    }


def _ingest_scanner_evidence(
    ctx: ToolContext, source: str, evidence: list[Any], *, elapsed: float = 0.0
) -> list[str]:
    ids: list[str] = []
    for item in evidence:
        kind = EvidenceGraphKind.SCANNER_RESULT.value
        provenance = "scanner_result"
        is_finding = item.kind.value == "scanner"
        if item.kind.value in {"scanner_plan", "tool_status"}:
            kind = EvidenceGraphKind.TOOL_EXECUTION.value
            provenance = "tool_status" if item.kind.value == "tool_status" else "scanner_plan"
            if not is_finding:
                node = ctx.graph.add(
                    kind=kind,
                    provenance=provenance,
                    summary=item.summary,
                    source=source,
                    extra={"elapsed": elapsed, "is_finding": False},
                )
                ids.append(node.id)
                continue
        node = ctx.graph.add(
            kind=kind,
            provenance=provenance,
            summary=item.summary,
            source=source,
            extra={"elapsed": elapsed},
        )
        ids.append(node.id)
    return ids


def _quality(quality: ToolResultQuality, extra: dict[str, Any]) -> dict[str, Any]:
    return {"quality": quality.value, **extra}


def _quality_from_state(state: ToolExecutionState) -> ToolResultQuality:
    mapping = {
        ToolExecutionState.RESULTS_AVAILABLE: ToolResultQuality.RESULTS_AVAILABLE,
        ToolExecutionState.RESULTS_INGESTED: ToolResultQuality.RESULTS_AVAILABLE,
        ToolExecutionState.OK: ToolResultQuality.SUCCESS,
        ToolExecutionState.COMPLETED: ToolResultQuality.SUCCESS,
        ToolExecutionState.TOOL_UNAVAILABLE: ToolResultQuality.UNAVAILABLE,
        ToolExecutionState.TIMEOUT: ToolResultQuality.TIMEOUT,
        ToolExecutionState.FAILED: ToolResultQuality.FAILED,
        ToolExecutionState.EXECUTION_ERROR: ToolResultQuality.FAILED,
        ToolExecutionState.SAFETY_BLOCKED: ToolResultQuality.BLOCKED,
        ToolExecutionState.INVALID_SCOPE: ToolResultQuality.BLOCKED,
        ToolExecutionState.DENIED: ToolResultQuality.BLOCKED,
        ToolExecutionState.APPROVAL_REQUIRED: ToolResultQuality.APPROVAL_REQUIRED,
        ToolExecutionState.DRY_RUN: ToolResultQuality.NO_RESULT,
        ToolExecutionState.PLANNED: ToolResultQuality.NO_RESULT,
        ToolExecutionState.AUTHORIZED: ToolResultQuality.NO_RESULT,
    }
    return mapping.get(state, ToolResultQuality.FAILED)


def _exploratory_test(ctx: ToolContext) -> Executor:
    async def run(arguments: dict[str, Any]) -> dict[str, Any]:
        import shutil

        from app.security_testing.exploratory import (
            ExploratoryContext,
            ExploratoryTestCandidate,
            ExploratoryTestPolicy,
            cached_exploratory_engine,
            docker_executor_for_exploratory,
            public_attempt,
            snapshot_hash,
        )

        policy = ExploratoryTestPolicy.from_settings()
        settings = get_settings()
        docker_ok = shutil.which("docker") is not None
        engine = cached_exploratory_engine(
            ctx.session_id,
            ctx.exploratory_records,
            policy=policy,
            executor=docker_executor_for_exploratory(
                docker_ok, image=settings.security_agent_exploratory_python_image
            ),
            docker_available=docker_ok,
            pytest_ready=bool(settings.security_agent_exploratory_pytest_ready),
        )
        candidate = ExploratoryTestCandidate(
            target=str(arguments.get("target_file") or ctx.repo_root),
            language=str(arguments.get("language") or ""),
            framework=str(arguments.get("framework") or ""),
            target_symbol=str(arguments.get("target_symbol") or ""),
            hypothesis_id=str(arguments.get("hypothesis_id") or ""),
            test_code=str(arguments.get("test_code") or ""),
            rationale=str(arguments.get("reason") or ""),
            expected_behavior=str(arguments.get("expected_behavior") or ""),
            oracle=str(arguments.get("oracle") or ""),
            confidence=str(arguments.get("confidence") or "low"),
            follow_up=str(arguments.get("follow_up") or ""),
            parent_attempt_id=str(arguments.get("parent_attempt_id") or ""),
            target_file=str(arguments.get("target_file") or ""),
            project_id=str(arguments.get("project_id") or ctx.project_id),
            session_id=str(arguments.get("session_id") or ctx.session_id),
        )
        context = ExploratoryContext(
            session_id=ctx.session_id,
            project_id=ctx.project_id,
            mode=ctx.mode.value,
            hypotheses=ctx.hypotheses(),
            repo_path=str(ctx.repo_root),
            commit_sha=_repo_commit(ctx.repo_root),
            snapshot_hash=snapshot_hash(str(ctx.repo_root)),
            graph=ctx.graph,
        )
        attempt = await engine.run(candidate, context)
        payload = public_attempt(attempt)
        payload["duration_seconds"] = attempt.duration
        payload["quality"] = _exploratory_quality(attempt.classification)
        payload["executed"] = attempt.executed
        payload["verified"] = False
        payload["llm_invoked"] = False
        return payload

    return run


def _repo_commit(path: Path) -> str:
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env={"PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip()[:64]


def _exploratory_quality(classification: str) -> str:
    if classification in {"BLOCKED", "UNSUPPORTED"}:
        return ToolResultQuality.BLOCKED.value
    if classification == "UNAVAILABLE":
        return ToolResultQuality.UNAVAILABLE.value
    if classification == "TIMEOUT":
        return ToolResultQuality.TIMEOUT.value
    if classification in {"ERROR", "INVALID"}:
        return ToolResultQuality.FAILED.value
    return ToolResultQuality.SUCCESS.value


def _from_tool_result(tool: str, result: ToolExecutionResult) -> dict[str, Any]:
    return {
        "quality": _quality_from_state(result.state).value,
        "tool": tool,
        "state": result.state.value,
        "detail": result.detail,
        "executed": result.state
        in {
            ToolExecutionState.RESULTS_AVAILABLE,
            ToolExecutionState.RESULTS_INGESTED,
            ToolExecutionState.COMPLETED,
            ToolExecutionState.OK,
        },
        "is_finding": False,
    }
