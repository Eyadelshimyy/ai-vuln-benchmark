"""HackerOne program lookup, structured scope sync, and reporting provider."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from app.adapters.hackerone.client import HackerOneApiClient
from app.adapters.hackerone.credentials import HackerOneCredentials
from app.adapters.hackerone.errors import HackerOneError
from app.adapters.hackerone.evaluator import HackerOneScopeEvaluator
from app.adapters.hackerone.intents import ReportIntentWorkflow
from app.adapters.hackerone.models import (
    HackerOneProgram,
    ProgramSyncStatus,
    ScopeExclusionRecord,
    ScopeMode,
    StructuredScopeRecord,
    WeaknessRecord,
    asset_type_from_hackerone,
    attributes,
    node_id,
    parse_hackerone_id,
)
from app.adapters.hackerone.reports import HackerOneReportDraft, HackerOneReportWorkflow
from app.adapters.hackerone.scope_hash import hash_program_scope
from app.adapters.reports.base import ReportProvider
from app.adapters.scope.base import ScopeProvider
from app.domain.evidence import EvidenceBundle
from app.domain.findings import SecurityFinding
from app.domain.reports import SecurityReport
from app.domain.scope import ScopeConstraint
from app.plugins.errors import AdapterNotImplementedError
from app.security_testing.operator_auth import OperatorSession
from app.security_testing.sanitization import wrap_untrusted
from app.security_testing.target import NETWORK_ASSET_TYPES


class HackerOneScopeProvider(ScopeProvider):
    """Scope provider that always requires an explicit HackerOneProgram.

    There is no global "active program". Evaluating program A never uses
    program B's structured scope.
    """

    def __init__(self) -> None:
        self.evaluator = HackerOneScopeEvaluator()

    @property
    def provider_id(self) -> str:
        return "hackerone"

    def get_scope(self) -> ScopeConstraint:
        raise HackerOneError(
            "HackerOne scope requires an explicit program. Use get_scope_for(program).",
            code="no_program",
        )

    def get_scope_for(self, program: HackerOneProgram) -> ScopeConstraint:
        hosts: list[str] = []
        # Scope exclusions are report-category/reward exclusions, not target denials.
        for record in program.structured_scopes:
            if record.asset_type in NETWORK_ASSET_TYPES:
                hosts.append(record.asset_identifier)
        return ScopeConstraint(
            allowed_hosts=tuple(hosts),
            excluded_hosts=(),
            instructions=wrap_untrusted("hackerone-program", program.instructions),
            allow_active_testing=program.active_testing_approved,
            program_name=program.name or program.handle,
        )

    def evaluate(self, program: HackerOneProgram, target: str, **kwargs: Any) -> Any:
        return self.evaluator.evaluate(program, target, **kwargs)


class HackerOneProvider(ReportProvider):
    """ReportProvider + program/scope/weakness sync. HackerOne HTTP stays here."""

    def __init__(
        self,
        *,
        credentials: HackerOneCredentials | None = None,
        client: HackerOneApiClient | None = None,
    ) -> None:
        self.credentials = credentials or HackerOneCredentials.from_env()
        self.client = client or HackerOneApiClient(self.credentials)
        self.programs: dict[str, HackerOneProgram] = {}
        self.scope_provider = HackerOneScopeProvider()
        self.reports = HackerOneReportWorkflow(self.client)
        self.intents = ReportIntentWorkflow(self.reports)
        self.syncs: list[dict[str, Any]] = []

    @property
    def provider_id(self) -> str:
        return "hackerone"

    def connection_status(self) -> dict[str, Any]:
        return {
            "configured": self.credentials.configured,
            "username_present": bool(self.credentials.username),
            "token_present": bool(self.credentials.configured and self.credentials.token),
            "base_url": self.credentials.base_url,
            "programs": [item.snapshot() for item in self.programs.values()],
        }

    def lookup_program(self, handle: str) -> HackerOneProgram:
        payload = self.client.get(f"hackers/programs/{handle}")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        attrs = attributes(data)
        scope_mode = _scope_mode(attrs)
        existing = self.programs.get(handle)
        program = HackerOneProgram(
            handle=str(attrs.get("handle") or handle),
            name=str(attrs.get("name") or handle),
            program_id=node_id(data) or None,
            program_url=_program_url(attrs, handle),
            fetched_at=datetime.now(UTC),
            sync_status=ProgramSyncStatus.OK,
            scope_mode=scope_mode,
            offers_bounties=_bool(attrs.get("offers_bounties")),
            requires_severity=_bool(attrs.get("requires_severity"), default=True) is not False,
            instructions=str(attrs.get("policy") or attrs.get("about") or ""),
            structured_scopes=existing.structured_scopes if existing else (),
            exclusions=existing.exclusions if existing else (),
            weaknesses=existing.weaknesses if existing else (),
            scope_version=existing.scope_version if existing else 0,
            scope_sync_complete=existing.scope_sync_complete if existing else False,
            last_scope_id=existing.last_scope_id if existing else None,
            open_scope_policy=existing.open_scope_policy if existing else "",
            open_scope_acknowledged=existing.open_scope_acknowledged if existing else False,
            active_testing_approved=existing.active_testing_approved if existing else False,
            scope_content_hash=existing.scope_content_hash if existing else "",
            weaknesses_synced_at=existing.weaknesses_synced_at if existing else None,
            scope_pages_fetched=existing.scope_pages_fetched if existing else 0,
        )
        self.programs[program.handle] = program
        return program

    def evaluate_scope(self, handle: str, target: str, **kwargs: Any) -> Any:
        program = self.programs.get(handle)
        if program is None:
            raise HackerOneError(
                "Sync the HackerOne program before evaluating scope", code="no_program"
            )
        return self.scope_provider.evaluate(program, target, **kwargs)

    def scope_for(self, handle: str) -> ScopeConstraint:
        program = self.programs.get(handle)
        if program is None:
            raise HackerOneError("Unknown HackerOne program", code="no_program")
        return self.scope_provider.get_scope_for(program)

    def sync_scope(self, handle: str) -> HackerOneProgram:
        """Fetch → validate → stage → replace atomically. Keep last-good on error."""
        program = self.programs.get(handle) or self.lookup_program(handle)
        previous = _copy_program(program)
        try:
            structured, scope_meta = self._structured_scopes(handle)
            exclusions = self._exclusions(handle)
            weaknesses = self._weaknesses(handle)
            complete = bool(scope_meta.get("scope_sync_complete"))
            if scope_meta.get("error"):
                raise HackerOneError(
                    str(scope_meta.get("error")),
                    code="scope_sync_incomplete",
                )
            if not complete:
                restored = previous
                restored.sync_status = ProgramSyncStatus.INCOMPLETE
                restored.scope_sync_complete = False
                restored.error = "Structured scope sync is incomplete; last-good snapshot kept"
                restored.continuation_state = str(scope_meta.get("continuation_state") or "")
                restored.last_scope_id = (
                    str(scope_meta.get("last_scope_id"))
                    if scope_meta.get("last_scope_id")
                    else restored.last_scope_id
                )
                restored.scope_pages_fetched = int(scope_meta.get("pages_fetched") or 0)
                restored.fetched_at = datetime.now(UTC)
                program = restored
                self.syncs.append(
                    {
                        "handle": handle,
                        "status": "incomplete",
                        "error": restored.error,
                        "scope_sync_complete": False,
                        "last_scope_id": restored.last_scope_id,
                        "fetched_at": restored.fetched_at.isoformat(),
                    }
                )
                self.programs[program.handle] = program
                return program
            staged = _copy_program(previous)
            staged.structured_scopes = tuple(structured)
            staged.exclusions = tuple(exclusions)
            staged.weaknesses = tuple(weaknesses)
            staged.fetched_at = datetime.now(UTC)
            staged.weaknesses_synced_at = staged.fetched_at
            staged.sync_status = ProgramSyncStatus.OK
            staged.error = None
            staged.scope_count = len(structured)
            staged.scope_sync_complete = True
            staged.last_scope_id = (
                str(scope_meta.get("last_scope_id")) if scope_meta.get("last_scope_id") else None
            )
            staged.continuation_state = str(scope_meta.get("continuation_state") or "complete")
            staged.scope_pages_fetched = int(scope_meta.get("pages_fetched") or 0)
            content_hash = hash_program_scope(staged)
            if content_hash == previous.scope_content_hash and previous.scope_content_hash:
                staged.scope_version = previous.scope_version
            else:
                staged.scope_version = previous.scope_version + 1
            staged.scope_content_hash = content_hash
            program = staged
            self.syncs.append(
                {
                    "handle": handle,
                    "status": "ok",
                    "scope_count": program.scope_count,
                    "last_scope_id": program.last_scope_id,
                    "scope_sync_complete": program.scope_sync_complete,
                    "scope_version": program.scope_version,
                    "scope_content_hash": program.scope_content_hash,
                    "weakness_count": len(weaknesses),
                    "fetched_at": program.fetched_at.isoformat() if program.fetched_at else None,
                }
            )
        except HackerOneError as exc:
            restored = previous
            restored.sync_status = ProgramSyncStatus.ERROR
            restored.error = str(exc)
            restored.scope_sync_complete = previous.scope_sync_complete
            restored.fetched_at = datetime.now(UTC)
            program = restored
            self.syncs.append(
                {
                    "handle": handle,
                    "status": "error",
                    "error": str(exc),
                    "scope_sync_complete": restored.scope_sync_complete,
                    "fetched_at": restored.fetched_at.isoformat() if restored.fetched_at else None,
                }
            )
        self.programs[program.handle] = program
        return program

    def sync_weaknesses(self, handle: str) -> HackerOneProgram:
        program = self.programs.get(handle) or self.lookup_program(handle)
        try:
            program.weaknesses = tuple(self._weaknesses(handle))
            program.weaknesses_synced_at = datetime.now(UTC)
            program.sync_status = ProgramSyncStatus.OK
            program.error = None
            new_hash = hash_program_scope(program)
            if new_hash != program.scope_content_hash:
                program.scope_version += 1
                program.scope_content_hash = new_hash
        except HackerOneError as exc:
            program.sync_status = ProgramSyncStatus.ERROR
            program.error = str(exc)
        self.programs[program.handle] = program
        return program

    def acknowledge_open_scope(
        self, handle: str, *, session: OperatorSession, policy: str
    ) -> HackerOneProgram:
        session.assert_human()
        program = self.programs.get(handle)
        if program is None:
            raise HackerOneError(
                "Import the program before acknowledging open-scope policy", code="no_program"
            )
        program.open_scope_acknowledged = True
        program.open_scope_policy = policy
        return program

    def approve_active_testing(self, handle: str, *, session: OperatorSession) -> HackerOneProgram:
        session.assert_human()
        program = self.programs.get(handle)
        if program is None:
            raise HackerOneError("Import the program first", code="no_program")
        program.active_testing_approved = True
        return program

    def untrusted_instructions(self, handle: str) -> str:
        program = self.programs.get(handle)
        if program is None:
            return ""
        return wrap_untrusted("hackerone-program", program.instructions)

    def render(
        self,
        findings: Sequence[SecurityFinding],
        *,
        evidence: EvidenceBundle | None = None,
    ) -> SecurityReport:
        from app.adapters.reports.local import LocalReportProvider

        local = LocalReportProvider().render(findings, evidence=evidence)
        local.destination = "hackerone-local-preview"
        local.submitted_remotely = False
        return local

    def submit(self, report: SecurityReport) -> str:
        raise AdapterNotImplementedError(
            "HackerOneProvider.submit() does not auto-submit. Use the report "
            "workflow: LOCAL_DRAFT → READY_FOR_REVIEW → HUMAN_APPROVED → submit()."
        )

    def draft_from_finding(
        self, finding: SecurityFinding, handle: str, **kwargs: Any
    ) -> HackerOneReportDraft:
        program = self.programs.get(handle)
        if program is None:
            raise HackerOneError(
                "Sync the HackerOne program before drafting a report", code="no_program"
            )
        return self.reports.draft_from_finding(finding, program, **kwargs)

    def _structured_scopes(self, handle: str) -> tuple[list[StructuredScopeRecord], dict[str, Any]]:
        rows, meta = self.client.paginate_with_meta(f"hackers/programs/{handle}/structured_scopes")
        records: list[StructuredScopeRecord] = []
        for row in rows:
            attrs = attributes(row)
            identifier = str(attrs.get("asset_identifier") or "")
            if not identifier:
                continue
            asset_raw = str(attrs.get("asset_type") or "other")
            ident = node_id(row)
            records.append(
                StructuredScopeRecord(
                    id=ident,
                    asset_type_raw=asset_raw,
                    asset_type=asset_type_from_hackerone(asset_raw),
                    asset_identifier=identifier,
                    instruction=str(attrs.get("instruction") or ""),
                    eligible_for_bounty=bool(attrs.get("eligible_for_bounty")),
                    eligible_for_submission=bool(
                        attrs.get("eligible_for_submission")
                        if attrs.get("eligible_for_submission") is not None
                        else True
                    ),
                    reference=str(attrs.get("reference") or "") or None,
                    original={
                        "id": ident,
                        "asset_type": asset_raw,
                        "asset_identifier": identifier,
                        "instruction": attrs.get("instruction"),
                        "eligible_for_bounty": attrs.get("eligible_for_bounty"),
                        "eligible_for_submission": attrs.get("eligible_for_submission"),
                        "reference": attrs.get("reference"),
                    },
                )
            )
        return records, meta

    def _exclusions(self, handle: str) -> list[ScopeExclusionRecord]:
        rows = self.client.paginate(f"hackers/programs/{handle}/scope_exclusions")
        records: list[ScopeExclusionRecord] = []
        for row in rows:
            attrs = attributes(row)
            records.append(
                ScopeExclusionRecord(
                    id=node_id(row),
                    category=str(attrs.get("category") or ""),
                    details=str(attrs.get("details") or ""),
                    created_at=str(attrs.get("created_at") or "") or None,
                    updated_at=str(attrs.get("updated_at") or "") or None,
                    original={
                        "id": node_id(row),
                        "category": attrs.get("category"),
                        "details": attrs.get("details"),
                        "created_at": attrs.get("created_at"),
                        "updated_at": attrs.get("updated_at"),
                    },
                )
            )
        return records

    def _weaknesses(self, handle: str) -> list[WeaknessRecord]:
        rows = self.client.paginate(f"hackers/programs/{handle}/weaknesses")
        records: list[WeaknessRecord] = []
        for row in rows:
            attrs = attributes(row)
            numeric = parse_hackerone_id(node_id(row))
            if numeric is None:
                continue
            records.append(
                WeaknessRecord(
                    id=numeric,
                    name=str(attrs.get("name") or ""),
                    description=str(attrs.get("description") or ""),
                    external_id=str(attrs.get("external_id") or attrs.get("cwe") or ""),
                    original={
                        "id": numeric,
                        "name": attrs.get("name"),
                        "description": attrs.get("description"),
                        "external_id": attrs.get("external_id") or attrs.get("cwe"),
                    },
                )
            )
        return records


def _program_url(attrs: dict[str, Any], handle: str) -> str:
    url = attrs.get("url") or attrs.get("web_url")
    if isinstance(url, str) and url.startswith("http"):
        return url
    return f"https://hackerone.com/{handle}"


def _scope_mode(attrs: dict[str, Any]) -> ScopeMode:
    raw = str(attrs.get("scope_type") or attrs.get("scope") or "").lower()
    if attrs.get("open_scope") is True or raw in {"open", "open_scope"}:
        return ScopeMode.OPEN
    return ScopeMode.CLOSED


def _bool(value: object, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return None


def _copy_program(program: HackerOneProgram) -> HackerOneProgram:
    return HackerOneProgram(
        handle=program.handle,
        name=program.name,
        program_id=program.program_id,
        program_url=program.program_url,
        fetched_at=program.fetched_at,
        sync_status=program.sync_status,
        error=program.error,
        scope_mode=program.scope_mode,
        offers_bounties=program.offers_bounties,
        requires_severity=program.requires_severity,
        instructions=program.instructions,
        structured_scopes=program.structured_scopes,
        exclusions=program.exclusions,
        weaknesses=program.weaknesses,
        open_scope_policy=program.open_scope_policy,
        open_scope_acknowledged=program.open_scope_acknowledged,
        active_testing_approved=program.active_testing_approved,
        scope_version=program.scope_version,
        scope_sync_complete=program.scope_sync_complete,
        scope_count=program.scope_count,
        last_scope_id=program.last_scope_id,
        continuation_state=program.continuation_state,
        scope_content_hash=program.scope_content_hash,
        weaknesses_synced_at=program.weaknesses_synced_at,
        scope_pages_fetched=program.scope_pages_fetched,
    )
