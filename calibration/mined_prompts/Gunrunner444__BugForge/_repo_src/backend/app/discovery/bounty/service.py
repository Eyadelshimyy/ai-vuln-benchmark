"""Operational bounty-campaign service (Phase 51, hardened).

This is the operational layer that makes the Phase 49 orchestrator and the
Phase 50 bounty engines reachable from the API and the Cursor MCP server. It
*reuses* the existing stack and introduces no parallel session, evidence store,
or authority model:

* identity is :class:`CampaignIdentity` derived from :class:`BountyManifest`;
* execution is the Phase 49 :class:`Orchestrator` over :class:`DiscoveryScheduler`;
* state is persisted through the Phase 49 ``SqlStore`` (via
  :mod:`app.discovery.bounty.persistence`), so a campaign survives a restart;
* the gate is :class:`BountyGate` (which wraps the Phase 49 ``DefaultGate``);
* pause/stop are real Phase 49 state-machine stops (``operator_paused`` is
  resumable, ``operator_stopped`` is final) taken under the campaign lock, so a
  concurrent execute cannot run an engine past them;
* expensive work runs as a bounded background job on a worker thread;
* findings, VFCS plans, advisories, and the report pack are the Phase 50 modules.

The service never calls a model, never marks a finding verified, never raises a
budget, and never grants an approval to itself. Fork testing stays gated on an
operator-granted approval that only the human-only API path can set. Missing
optional tools surface as UNAVAILABLE and are never faked.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from app.discovery.bounty.campaign import BountyManifest, ManifestError

if TYPE_CHECKING:
    from app.discovery.bounty.analysis import CampaignAnalysis
    from app.discovery.bounty.compiler_diff import DifferentialResult
    from app.discovery.bounty.report_pack import ReportPack
from app.discovery.bounty.source_selection import (
    DEFAULT_LIMIT,
    SourceSelection,
    select_named,
    select_sources,
)
from app.discovery.engine import AnalysisRequest, DiscoveryEngine
from app.discovery.orchestration import CampaignIdentity, Orchestrator
from app.discovery.orchestration.model import (
    TERMINAL,
    OrchestratorState,
    PersistenceError,
    ResearchState,
    ResumeError,
    StopReason,
)
from app.discovery.scheduler import DiscoveryScheduler
from app.services.bounty_campaign_store import (
    CampaignConflictError,
    CampaignPersistenceError,
    CampaignRecord,
    CampaignRecordStore,
    RunnerSqlStore,
    SessionRunner,
    ensure_schema,
    runner_from_settings,
)

# The research engine and campaign records carry this version. Records written by
# earlier phases keep the version they were written with; nothing rewrites them.
ENGINE_VERSION = "phase51.1"
MAX_CAMPAIGNS = 256
MAX_FILES_PER_REQUEST = 64
DEFAULT_MAX_ROUNDS = 16
DEFAULT_MAX_ENGINES = 16
MAX_JOB_STEPS = 64
JOB_WORKERS = 2
# Only an operator may ever approve these. The AI and the MCP server cannot.
APPROVABLE_CAPABILITIES = frozenset({"fork_validation"})
CONTROL_ACTIVE = "active"
CONTROL_PAUSED = "paused"
CONTROL_STOPPED = "stopped"
JOB_IDLE = "idle"
JOB_ACTIVE = frozenset({"queued", "running"})


class CampaignError(ValueError):
    """The campaign request is malformed or refers to something unavailable."""


class UnknownCampaignError(KeyError):
    """No campaign exists for the identifier."""


class CampaignControlError(CampaignError):
    """The campaign's control state (paused, stopped, busy) refuses the action."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class CampaignSpec:
    manifest: BountyManifest
    repo_root: Path
    target: str = ""
    contract: str = ""
    function: str = ""
    source_file: str = ""
    files: tuple[str, ...] = ()
    language: str = "solidity"
    max_rounds: int = DEFAULT_MAX_ROUNDS
    max_engines: int = DEFAULT_MAX_ENGINES
    # Target deployment (Phase 51 hardening): the address and chain the operator
    # is researching. Never inferred from the source.
    address: str = ""
    chain_id: str = ""
    source_limit: int = DEFAULT_LIMIT
    # Typed constructor configuration. None keeps the fail-closed constructor builder.
    constructor_profile: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_root": str(self.repo_root),
            "target": self.target,
            "contract": self.contract,
            "function": self.function,
            "source_file": self.source_file,
            "files": list(self.files),
            "language": self.language,
            "max_rounds": self.max_rounds,
            "max_engines": self.max_engines,
            "address": self.address,
            "chain_id": self.chain_id,
            "source_limit": self.source_limit,
            "constructor_profile": self.constructor_profile,
        }

    @staticmethod
    def from_record(manifest: BountyManifest, data: dict[str, Any]) -> CampaignSpec:
        return CampaignSpec(
            manifest=manifest,
            repo_root=Path(str(data.get("repo_root", ""))),
            target=str(data.get("target", "")),
            contract=str(data.get("contract", "")),
            function=str(data.get("function", "")),
            source_file=str(data.get("source_file", "")),
            files=tuple(str(item) for item in data.get("files", [])),
            language=str(data.get("language", "solidity")),
            max_rounds=int(data.get("max_rounds", DEFAULT_MAX_ROUNDS)),
            max_engines=int(data.get("max_engines", DEFAULT_MAX_ENGINES)),
            address=str(data.get("address", "")),
            chain_id=str(data.get("chain_id", "")),
            source_limit=int(data.get("source_limit", DEFAULT_LIMIT)),
            constructor_profile=_profile_record(data.get("constructor_profile")),
        )


@dataclass
class BountyCampaign:
    """One live campaign: the orchestrator, the derived identity, and its record."""

    campaign_id: str
    spec: CampaignSpec
    identity: CampaignIdentity
    request: AnalysisRequest
    orchestrator: Orchestrator
    record: CampaignRecord
    selection: SourceSelection
    lock: threading.RLock = field(default_factory=threading.RLock)
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def manifest(self) -> BountyManifest:
        return self.spec.manifest

    @property
    def state(self) -> ResearchState:
        return self.orchestrator.state

    @property
    def operator_identity(self) -> str:
        return self.record.operator_identity

    @property
    def approvals(self) -> frozenset[str]:
        return frozenset(str(item.get("capability", "")) for item in self.record.approvals)

    @property
    def control(self) -> str:
        return self.record.control


def _sanitize_files(files: tuple[str, ...]) -> tuple[str, ...]:
    chosen: list[str] = []
    for name in files:
        text = str(name).strip()
        if not text or ".." in text or text.startswith("/"):
            raise CampaignError(f"unsafe file reference: {name!r}")
        chosen.append(text)
        if len(chosen) >= MAX_FILES_PER_REQUEST:
            break
    return tuple(chosen)


def _validate_target(spec: CampaignSpec) -> None:
    import re

    if spec.address and not re.fullmatch(r"0x[0-9a-fA-F]{40}", spec.address):
        raise CampaignError(f"target address {spec.address!r} is not an address")
    if spec.chain_id and not re.fullmatch(r"[0-9]{1,12}", spec.chain_id):
        raise CampaignError(f"chain id {spec.chain_id!r} is not a decimal chain id")
    if spec.chain_id and not spec.address:
        raise CampaignError("a chain id without an address names no deployment")


def _profile_record(raw: Any) -> dict[str, Any] | None:
    """A stored profile document, or None. Parsing happens later and fails closed."""
    if not isinstance(raw, dict) or not raw:
        return None
    return dict(raw)


def _accepted_profile(campaign: BountyCampaign) -> tuple[dict[str, Any] | None, Any]:
    """The profile analysis accepted, parsed again for the executor.

    A rejected or missing profile does not invent constructor arguments.
    """
    from app.discovery.bounty.constructor_profile import ProfileError, parse_profile

    cached = campaign.cache.get("constructor_profile")
    record = dict(cached) if isinstance(cached, dict) else None
    raw = campaign.spec.constructor_profile
    if record is None or record.get("validation") != "accepted" or not isinstance(raw, dict):
        return record, None
    try:
        return record, parse_profile(raw)
    except ProfileError as exc:
        rejected = {"validation": "rejected", "reason_code": exc.code, "reason": exc.reason}
        campaign.cache["constructor_profile"] = rejected
        return rejected, None


class BountyCampaignService:
    """Persistent registry of campaigns. One instance is shared by the API."""

    def __init__(
        self,
        runner: SessionRunner | None = None,
        *,
        engine_factory: Callable[[CampaignSpec], tuple[DiscoveryEngine, ...]] | None = None,
        job_workers: int = JOB_WORKERS,
    ) -> None:
        self._runner = runner or runner_from_settings()
        self._schema_ready = False
        self._store = RunnerSqlStore(self._runner)
        self._records = CampaignRecordStore(self._runner)
        self._campaigns: dict[str, BountyCampaign] = {}
        self._guard = threading.Lock()
        self._engine_factory = engine_factory
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, job_workers), thread_name_prefix="bounty-campaign"
        )
        self._jobs: dict[str, Future[None]] = {}
        self._halt: dict[str, StopReason] = {}

    # ---- persistence plumbing ---------------------------------------------------------

    def _ready(self) -> None:
        if not self._schema_ready:
            ensure_schema(self._runner)
            self._schema_ready = True

    @property
    def persistence(self) -> str:
        return self._runner.describe()

    def _save_record(self, campaign: BountyCampaign) -> None:
        try:
            campaign.record = self._records.update(campaign.record)
        except CampaignConflictError:
            self._evict(campaign.campaign_id)
            raise

    def _evict(self, campaign_id: str) -> None:
        with self._guard:
            self._campaigns.pop(campaign_id, None)
        self._store.forget(campaign_id)

    # ---- lifecycle ----------------------------------------------------------------------

    def create(self, spec: CampaignSpec, *, operator_identity: str = "") -> BountyCampaign:
        self._ready()
        root = spec.repo_root.resolve()
        if not root.is_dir():
            raise CampaignError(f"repository root is not a directory: {root}")
        _validate_target(spec)
        spec = CampaignSpec(**{**spec.__dict__, "repo_root": root})
        identity = self._identity(spec)
        existing = self._records.get(identity.campaign_id)
        if existing is not None:
            if existing.operator_identity and existing.operator_identity != operator_identity:
                raise CampaignConflictError("campaign exists and belongs to another operator")
            return self.get(identity.campaign_id)
        with self._guard:
            if len(self._campaigns) >= MAX_CAMPAIGNS:
                raise CampaignError("the campaign registry is full")
        campaign = self._build(spec, identity, record=None, operator_identity=operator_identity)
        with self._guard:
            self._campaigns[campaign.campaign_id] = campaign
        return campaign

    def _identity(self, spec: CampaignSpec) -> CampaignIdentity:
        identity = spec.manifest.to_campaign_identity(
            contract=spec.contract,
            function=spec.function,
            source_file=spec.source_file,
            repository_root=str(spec.repo_root),
        )
        if spec.address:
            deployment = f"{spec.chain_id}:{spec.address.lower()}"
            base = CampaignIdentity(**{**identity.__dict__, "campaign_id": ""})
            base = CampaignIdentity(**{**base.__dict__, "deployment": deployment})
            identity = CampaignIdentity(
                **{**base.__dict__, "campaign_id": f"cp_{base.target_digest()}"}
            )
        return identity

    def _selection(self, spec: CampaignSpec, files: tuple[str, ...]) -> SourceSelection:
        if files:
            return select_named(spec.repo_root, files)
        return select_sources(
            spec.repo_root,
            manifest=spec.manifest,
            focus_contract=spec.contract,
            focus_file=spec.source_file,
            limit=max(1, min(spec.source_limit, MAX_FILES_PER_REQUEST)),
        )

    def _build(
        self,
        spec: CampaignSpec,
        identity: CampaignIdentity,
        *,
        record: CampaignRecord | None,
        operator_identity: str = "",
    ) -> BountyCampaign:
        files = _sanitize_files(spec.files)
        selection = self._selection(spec, files)
        chosen = files or selection.selected
        extra = {"project_id": spec.manifest.program_id, **spec.manifest.request_extra()}
        extra.update(selection.request_extra())
        if spec.address:
            extra["deployment_address"] = spec.address.lower()
            extra["deployment_chain_id"] = spec.chain_id
        request = AnalysisRequest(
            repo_root=spec.repo_root,
            language=spec.language,
            target=spec.target or spec.contract or spec.manifest.program_id,
            contract=spec.contract,
            function=spec.function,
            source_file=spec.source_file,
            files=chosen,
            campaign_id=identity.campaign_id,
            extra=extra,
        )
        approvals = (
            frozenset(str(a.get("capability", "")) for a in record.approvals)
            if record
            else frozenset()
        )
        from app.discovery.bounty.gate import BountyGate

        orchestrator = Orchestrator(
            self._build_scheduler(spec),
            request,
            identity=identity,
            gate=BountyGate(spec.manifest, approvals=approvals),
            store=self._store,
            max_rounds=spec.max_rounds,
        )
        try:
            orchestrator.start()
        except ResumeError as exc:
            raise CampaignError(f"stored campaign cannot be restored: {exc}") from exc
        if record is None:
            record = self._records.insert(
                CampaignRecord(
                    campaign_id=identity.campaign_id,
                    operator_identity=operator_identity,
                    engine_version=ENGINE_VERSION,
                    program_context=spec.manifest.identity_digest(),
                    manifest=spec.manifest.to_dict(),
                    spec=spec.to_dict(),
                    artifacts={"source_selection": selection.as_dict()},
                )
            )
        elif record.job.get("status") in JOB_ACTIVE and identity.campaign_id not in self._jobs:
            # A job that was running when the process stopped did not finish.
            record.job = {**record.job, "status": "interrupted", "finished_at": _now()}
            record = self._records.update(record)
        return BountyCampaign(
            campaign_id=identity.campaign_id,
            spec=spec,
            identity=identity,
            request=request,
            orchestrator=orchestrator,
            record=record,
            selection=selection,
        )

    def _build_scheduler(self, spec: CampaignSpec) -> DiscoveryScheduler:
        if self._engine_factory is not None:
            engines = self._engine_factory(spec)
        else:
            from app.discovery.bounty.engine import build_bounty_engines
            from app.discovery.builtin import BugforgeStaticEngine

            engines = (BugforgeStaticEngine(), *build_bounty_engines(spec.manifest))
        from app.discovery.corpus import DiscoveryCorpus

        return DiscoveryScheduler(
            engines=tuple(engines),
            max_engines=spec.max_engines,
            max_rounds=spec.max_rounds,
            corpus=DiscoveryCorpus(),
        )

    def get(self, campaign_id: str) -> BountyCampaign:
        with self._guard:
            cached = self._campaigns.get(campaign_id)
        if cached is not None:
            return cached
        self._ready()
        record = self._records.get(campaign_id)
        if record is None:
            raise UnknownCampaignError(campaign_id)
        try:
            manifest = BountyManifest.from_dict(record.manifest)
        except ManifestError as exc:
            raise CampaignError(f"stored manifest is invalid: {exc}") from exc
        spec = CampaignSpec.from_record(manifest, record.spec)
        identity = self._identity(spec)
        if identity.campaign_id != campaign_id:
            raise CampaignError("stored campaign no longer derives the same identity")
        campaign = self._build(spec, identity, record=record)
        with self._guard:
            existing = self._campaigns.setdefault(campaign_id, campaign)
        return existing

    def list_ids(self, operator_identity: str | None = None) -> list[str]:
        self._ready()
        return self._records.list_ids(operator_identity)

    def reload(self, campaign_id: str) -> BountyCampaign:
        """Drop the in-process copy and rebuild from storage (used after a conflict)."""
        self._evict(campaign_id)
        return self.get(campaign_id)

    # ---- advancing ----------------------------------------------------------------------

    def _check_runnable(self, campaign: BountyCampaign, *, from_job: bool = False) -> None:
        if campaign.control == CONTROL_STOPPED:
            raise CampaignControlError("campaign is stopped by the operator")
        if campaign.control == CONTROL_PAUSED:
            raise CampaignControlError("campaign is paused; resume it first")
        if not from_job and campaign.record.job.get("status") in JOB_ACTIVE:
            raise CampaignControlError("a background job is advancing this campaign")

    def _step_locked(self, campaign: BountyCampaign) -> bool:
        try:
            progressed = campaign.orchestrator.step()
        except CampaignConflictError:
            self._evict(campaign.campaign_id)
            raise
        campaign.cache.clear()
        return progressed

    def analyze(self, campaign_id: str, *, max_steps: int = MAX_JOB_STEPS) -> dict[str, Any]:
        """Run the bounded campaign synchronously (callers run this off the event loop)."""
        campaign = self.get(campaign_id)
        steps = 0
        while steps < max_steps:
            if self._halt.get(campaign_id) is not None:
                break
            with campaign.lock:
                self._check_runnable(campaign, from_job=True)
                if campaign.record.job.get("status") in JOB_ACTIVE and not self._is_job_thread(
                    campaign_id
                ):
                    raise CampaignControlError("a background job is advancing this campaign")
                progressed = self._step_locked(campaign)
            steps += 1
            if not progressed:
                break
        self._after_run(campaign)
        return self.report(campaign_id)

    def step(self, campaign_id: str, *, capability: str = "", reason: str = "") -> dict[str, Any]:
        campaign = self.get(campaign_id)
        with campaign.lock:
            self._check_runnable(campaign)
            if capability:
                campaign.orchestrator.suggest(capability, reason=reason or "operator execute")
            progressed = self._step_locked(campaign)
        self._after_run(campaign)
        result = self.report(campaign_id)
        result["progressed"] = progressed
        return result

    def _after_run(self, campaign: BountyCampaign) -> None:
        """Hook for post-orchestrator campaign work (VFCS stateful loop, artifacts)."""
        return None

    def suggest(
        self, campaign_id: str, capability: str, *, engine: str = "", reason: str = ""
    ) -> bool:
        """Queue a Cursor/operator suggestion. It is validated like any candidate.

        A suggestion is a hint only; it cannot widen scope, raise budget, or
        enable a gated capability.
        """
        campaign = self.get(campaign_id)
        with campaign.lock:
            if campaign.control != CONTROL_ACTIVE:
                return False
            return campaign.orchestrator.suggest(capability, engine=engine, reason=reason)

    # ---- control: pause / resume / stop ------------------------------------------------

    def pause(self, campaign_id: str, *, reason: str = "operator") -> dict[str, Any]:
        return self._halt_campaign(campaign_id, StopReason.OPERATOR_PAUSED, reason)

    def stop(self, campaign_id: str, *, reason: str = "operator") -> dict[str, Any]:
        return self._halt_campaign(campaign_id, StopReason.OPERATOR_STOPPED, reason)

    def _halt_campaign(self, campaign_id: str, reason: StopReason, detail: str) -> dict[str, Any]:
        campaign = self.get(campaign_id)
        if campaign.control == CONTROL_STOPPED:
            raise CampaignControlError("campaign is already stopped")
        # Signal a running job first so it halts at the next round boundary.
        self._halt[campaign_id] = reason
        try:
            with campaign.lock:  # waits for any in-flight round to finish
                changed = campaign.orchestrator.halt(reason, detail=detail[:200])
                campaign.record.control = (
                    CONTROL_PAUSED if reason is StopReason.OPERATOR_PAUSED else CONTROL_STOPPED
                )
                campaign.record.control_reason = detail[:200]
                self._save_record(campaign)
        finally:
            self._halt.pop(campaign_id, None)
        state = self.report(campaign_id)
        state["state_changed"] = changed
        return state

    def resume(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.get(campaign_id)
        with campaign.lock:
            if campaign.control == CONTROL_STOPPED:
                raise CampaignControlError("a stopped campaign is final and cannot be resumed")
            if campaign.record.job.get("status") in JOB_ACTIVE:
                raise CampaignControlError("a background job is advancing this campaign")
            try:
                campaign.orchestrator.start(explicit_resume=True)
            except CampaignConflictError:
                self._evict(campaign_id)
                raise
            except ResumeError as exc:
                raise CampaignError(f"campaign cannot be resumed: {exc}") from exc
            campaign.record.control = CONTROL_ACTIVE
            campaign.record.control_reason = ""
            self._save_record(campaign)
            campaign.cache.clear()
        return self.report(campaign_id)

    # ---- background jobs -----------------------------------------------------------------

    def _is_job_thread(self, campaign_id: str) -> bool:
        return threading.current_thread().name.startswith("bounty-campaign") and bool(
            self._jobs.get(campaign_id)
        )

    def start_job(self, campaign_id: str, *, max_steps: int = MAX_JOB_STEPS) -> dict[str, Any]:
        """Queue the bounded campaign run on a worker thread and return immediately."""
        campaign = self.get(campaign_id)
        with campaign.lock:
            self._check_runnable(campaign)
            if campaign.state.state in TERMINAL:
                raise CampaignControlError(
                    f"campaign is {campaign.state.state.value}; nothing remains to run"
                )
            job_id = f"job_{uuid.uuid4().hex[:12]}"
            campaign.record.job = {
                "job_id": job_id,
                "status": "queued",
                "steps": 0,
                "max_steps": max(1, min(max_steps, MAX_JOB_STEPS)),
                "queued_at": _now(),
                "started_at": "",
                "finished_at": "",
                "error": "",
            }
            self._save_record(campaign)
            future = self._executor.submit(self._run_job, campaign_id, job_id)
            self._jobs[campaign_id] = future
        return self.progress(campaign_id)

    def _run_job(self, campaign_id: str, job_id: str) -> None:
        campaign = self.get(campaign_id)
        try:
            with campaign.lock:
                if campaign.record.job.get("job_id") != job_id:
                    return
                campaign.record.job.update({"status": "running", "started_at": _now()})
                self._save_record(campaign)
            limit = int(campaign.record.job.get("max_steps", MAX_JOB_STEPS))
            outcome = "completed"
            for _ in range(limit):
                if self._halt.get(campaign_id) is not None:
                    outcome = "halted"
                    break
                with campaign.lock:
                    if campaign.control != CONTROL_ACTIVE:
                        outcome = "halted"
                        break
                    progressed = self._step_locked(campaign)
                    campaign.record.job["steps"] = int(campaign.record.job.get("steps", 0)) + 1
                    campaign.record.job["round"] = campaign.state.round
                    campaign.record.job["state"] = campaign.state.state.value
                    self._save_record(campaign)
                if not progressed:
                    break
            if outcome == "completed":
                self._after_run(campaign)
            with campaign.lock:
                campaign.record.job.update({"status": outcome, "finished_at": _now()})
                self._save_record(campaign)
        except Exception as exc:  # recorded, never swallowed silently
            try:
                fresh = self.reload(campaign_id)
                with fresh.lock:
                    fresh.record.job.update(
                        {
                            "status": "failed",
                            "finished_at": _now(),
                            "error": f"{type(exc).__name__}: {exc}"[:300],
                        }
                    )
                    self._save_record(fresh)
            except Exception:
                pass
        finally:
            self._jobs.pop(campaign_id, None)

    def wait_for_job(self, campaign_id: str, timeout: float = 300.0) -> dict[str, Any]:
        future = self._jobs.get(campaign_id)
        if future is not None:
            future.result(timeout=timeout)
        return self.progress(campaign_id)

    def progress(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.get(campaign_id)
        state = campaign.state
        job = dict(campaign.record.job) or {"status": JOB_IDLE}
        return {
            "campaign_id": campaign_id,
            "control": campaign.control,
            "control_reason": campaign.record.control_reason,
            "job": job,
            "state": state.state.value,
            "stop_reason": state.stop_reason,
            "round": state.round,
            "decisions": state.decision_number,
            "budget_remaining": {
                name: state.budget.remaining(name) for name in sorted(state.budget.limits)
            },
            "verified": False,
        }

    # ---- operator approvals (human only) ------------------------------------------------

    def grant_approval(
        self, campaign_id: str, capability: str, *, granted_by: str = ""
    ) -> frozenset[str]:
        """Record an operator approval for a gated capability.

        The caller must already have proven it is a human operator; this method
        makes no authority decision of its own beyond refusing capabilities that
        are not approvable at all. The approval is persisted with its grantor.
        """
        if capability not in APPROVABLE_CAPABILITIES:
            raise CampaignError(f"capability is not approvable: {capability!r}")
        campaign = self.get(campaign_id)
        with campaign.lock:
            if capability not in campaign.approvals:
                campaign.record.approvals.append(
                    {"capability": capability, "granted_by": granted_by, "granted_at": _now()}
                )
                self._save_record(campaign)
            from app.discovery.bounty.gate import BountyGate

            campaign.orchestrator.gate = BountyGate(campaign.manifest, approvals=campaign.approvals)
        return campaign.approvals

    # ---- reads --------------------------------------------------------------------------

    def report(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.get(campaign_id)
        report = dict(campaign.orchestrator.report())
        report["program_context"] = campaign.manifest.identity_digest()
        report["scope_summary"] = self._scope_summary(campaign)
        report["manifest_gaps"] = list(campaign.manifest.gaps())
        report["approvals"] = sorted(campaign.approvals)
        report["control"] = campaign.control
        report["control_reason"] = campaign.record.control_reason
        report["paused"] = campaign.control == CONTROL_PAUSED
        report["stopped_by_operator"] = campaign.control == CONTROL_STOPPED
        report["job"] = dict(campaign.record.job) or {"status": JOB_IDLE}
        report["engine_version"] = campaign.record.engine_version
        report["persistence"] = self.persistence
        report["source_selection"] = {
            "truncated": campaign.selection.truncated,
            "selected": len(campaign.selection.selected),
            "considered": campaign.selection.considered,
            "mode": campaign.selection.mode,
        }
        report["verified"] = False
        report["submitted"] = False
        report["llm_invoked"] = False
        return report

    def next_action(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.get(campaign_id)
        state = campaign.state
        return {
            "campaign_id": campaign_id,
            "state": state.state.value,
            "control": campaign.control,
            "next_capability": state.next_capability,
            "uncertainties": list(state.uncertainties),
            "stop_reason": state.stop_reason,
            "recommend_verification_review": state.recommend_verification_review,
            "note": "a recommendation only; it verifies nothing and grants no approval",
        }

    def decisions(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.get(campaign_id)
        records = self._store.decisions(campaign_id)
        return {
            "campaign_id": campaign_id,
            "decisions": [
                {
                    "sequence": item.sequence,
                    "round": item.round,
                    "source": item.source.value,
                    "phase": item.phase.value,
                    "selected_capability": item.selected_capability,
                    "selected_engine": item.selected_engine,
                    "stop_reason": item.stop_reason,
                    "rationale": item.rationale,
                    "record_hash": item.record_hash,
                }
                for item in records
            ],
            "count": len(records),
            "state": campaign.state.state.value,
        }

    def _scope_summary(self, campaign: BountyCampaign) -> dict[str, str]:
        scope = campaign.manifest.scope_of(
            contract=campaign.spec.contract,
            file=campaign.spec.source_file,
            address=campaign.spec.address,
            chain_id=campaign.spec.chain_id,
        )
        return {"status": scope.status.value, "reason": scope.reason}

    def source_selection(self, campaign_id: str) -> dict[str, Any]:
        """Truncation-aware ranking of the repo's sources. Dropped != safe."""
        campaign = self.get(campaign_id)
        return {"campaign_id": campaign_id, **campaign.selection.as_dict()}

    # ---- analysis-backed reads ----------------------------------------------------------

    def _snapshot_token(self, campaign: BountyCampaign) -> str:
        """Content and inventory identity for the campaign's current source snapshot."""
        import hashlib

        from app.discovery.bounty.analysis import read_sources
        from app.discovery.orchestration.codec import digest

        root = campaign.spec.repo_root
        listing: list[str] = []
        skip = {
            ".git",
            "node_modules",
            "lib",
            "out",
            "cache",
            "artifacts",
            "broadcast",
            "script",
            "test",
            "tests",
        }
        if root.is_dir():
            for path in sorted(root.rglob("*.sol")):
                rel = path.relative_to(root)
                if set(rel.parts) & skip:
                    continue
                listing.append(rel.as_posix())
                if len(listing) >= 2000:
                    break
        sources = read_sources(root, campaign.request.files)
        hashes = tuple(
            sorted(
                (path, hashlib.sha256(text.encode("utf-8")).hexdigest())
                for path, text in sources.items()
            )
        )
        return digest(
            {
                "listing": listing,
                "hashes": hashes,
                "compiler": campaign.manifest.compiler.fingerprint(),
                "contract": campaign.spec.contract,
                "address": campaign.spec.address.lower(),
                "chain_id": campaign.spec.chain_id,
                "selected": list(campaign.selection.selected),
                "mode": campaign.selection.mode,
            }
        )

    def analysis_stale(self, campaign: BountyCampaign) -> bool:
        """True when a cached analysis no longer matches the files on disk."""
        if campaign.cache.get("analysis") is None:
            return False
        return bool(campaign.cache.get("analysis_stale"))

    def _analysis(self, campaign: BountyCampaign) -> CampaignAnalysis:
        from app.discovery.bounty.analysis import analyze_campaign

        token = self._snapshot_token(campaign)
        cached = campaign.cache.get("analysis")
        if cached is not None:
            # Keep the plan that was built. A changed snapshot is stale and must
            # not be restamped; execution fails closed until reanalyze().
            campaign.cache["analysis_stale"] = campaign.cache.get("analysis_token") != token
            return cast("CampaignAnalysis", cached)
        analysis = analyze_campaign(
            manifest=campaign.manifest,
            identity=campaign.identity,
            repo_root=campaign.spec.repo_root,
            files=campaign.request.files,
            selection=campaign.selection,
            contract=campaign.spec.contract,
            address=campaign.spec.address,
            chain_id=campaign.spec.chain_id,
        )
        from app.discovery.bounty.constructor_profile import attach_profile_seed

        attach_profile_seed(campaign, analysis)
        campaign.cache["analysis"] = analysis
        campaign.cache["analysis_token"] = token
        campaign.cache["analysis_stale"] = False
        return analysis

    def reanalyze(self, campaign_id: str) -> CampaignAnalysis:
        """Build a new plan from the current snapshot. Old executions are not copied onto it."""
        from dataclasses import replace

        from app.discovery.bounty.source_selection import select_sources

        campaign = self.get(campaign_id)
        if campaign.selection.mode != "explicit":
            fresh = select_sources(
                campaign.spec.repo_root,
                manifest=campaign.manifest,
                focus_contract=campaign.spec.contract,
                focus_file=campaign.spec.source_file,
                limit=max(1, min(campaign.spec.source_limit, MAX_FILES_PER_REQUEST)),
            )
            campaign.request = replace(campaign.request, files=fresh.selected)
        else:
            fresh = select_named(campaign.spec.repo_root, campaign.request.files)
        campaign.selection = fresh
        campaign.cache.pop("analysis", None)
        campaign.cache.pop("analysis_token", None)
        campaign.cache.pop("analysis_stale", None)
        return self._analysis(campaign)

    def findings(self, campaign_id: str) -> dict[str, Any]:
        from app.discovery.bounty.findings import build_findings

        campaign = self.get(campaign_id)
        analysis = self._analysis(campaign)
        findings = build_findings(
            analysis.candidates,
            manifest=campaign.manifest,
            identity=campaign.identity,
            state=campaign.state,
            sequences=analysis.sequences,
            deployment_for=analysis.deployment_for,
            ambiguous_contracts=analysis.ambiguous_contracts,
            target_address=campaign.spec.address,
            target_chain_id=campaign.spec.chain_id,
            executions=self.stored_executions(campaign, analysis),
        )
        return {
            "campaign_id": campaign_id,
            "verified": False,
            "deployment": analysis.deployment.as_dict(),
            "findings": [_finding_dict(item) for item in findings],
        }

    def evidence(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.get(campaign_id)
        state = campaign.state
        return {
            "campaign_id": campaign_id,
            "verified": False,
            "evidence": [_evidence_dict(item) for item in state.evidence],
            "contradictions": [
                {
                    "id": item.contradiction_id,
                    "kind": item.kind,
                    "identity": item.identity_key,
                    "status": item.status,
                }
                for item in state.contradictions
            ],
        }

    def repro(self, campaign_id: str) -> dict[str, Any]:
        from app.discovery.bounty.vfcs import instance_identities

        campaign = self.get(campaign_id)
        analysis = self._analysis(campaign)
        return {
            "campaign_id": campaign_id,
            "note": "call-sequence plans from static candidates; plans are not executions",
            "verified": False,
            "deployment": analysis.deployment.key,
            "sequences": [
                {
                    "sequence_id": seq.sequence_id,
                    "template": seq.template,
                    "derived_from": seq.derived_from,
                    "deployment": seq.identity.deployment,
                    "calls": [call.identity for call in seq.calls],
                    "instances": list(instance_identities(seq)),
                }
                for seq in analysis.sequences
            ],
        }

    def vfcs_feedback(self, campaign_id: str, signals: list[dict[str, Any]]) -> dict[str, Any]:
        from app.discovery.bounty.vfcs import FeedbackSignal, incorporate_feedback

        campaign = self.get(campaign_id)
        analysis = self._analysis(campaign)
        available = frozenset(
            engine.engine_id
            for engine in campaign.orchestrator.scheduler.engines
            if engine.availability().value == "available"
        )
        # fuzzers are discovery engines named foundry/echidna/medusa/ityfuzz
        fuzzers = frozenset(
            name
            for name in ("foundry", "echidna", "medusa", "ityfuzz")
            if any(
                e.engine_id == name and e.availability().value == "available"
                for e in campaign.orchestrator.scheduler.engines
            )
        )
        parsed = [
            FeedbackSignal(
                kind=str(item.get("kind", "")),
                sequence_id=str(item.get("sequence_id", "")),
                call_index=int(item.get("call_index", -1)),
                values=tuple((str(k), str(v)) for k, v in dict(item.get("values", {})).items()),
                engine=str(item.get("engine", "")),
                call_instance=str(item.get("call_instance", "")),
            )
            for item in signals
        ]
        outcome = incorporate_feedback(
            analysis.sequences, parsed, available_engines=fuzzers or available
        )
        return {"campaign_id": campaign_id, **outcome.as_dict()}

    def advisories(self, campaign_id: str) -> dict[str, Any]:
        from app.discovery.bounty.advisories import (
            advisory_dict,
            match_advisories,
            precondition_graphs,
        )

        campaign = self.get(campaign_id)
        analysis = self._analysis(campaign)
        report = advisory_dict(match_advisories(campaign.manifest.compiler, analysis.sources))
        report["precondition_graphs"] = [
            graph.as_dict()
            for graph in precondition_graphs(campaign.manifest.compiler, analysis.sources)
        ]
        return report

    def scope_identity(self, campaign_id: str) -> dict[str, Any]:
        from app.discovery.bounty.deployment_identity import resolve_scope_identity

        campaign = self.get(campaign_id)
        analysis = self._analysis(campaign)
        decision = resolve_scope_identity(
            campaign.manifest,
            contract=campaign.spec.contract,
            file=campaign.spec.source_file,
            address=campaign.spec.address,
            chain_id=campaign.spec.chain_id,
            model=analysis.model,
        )
        result = {"campaign_id": campaign_id, **decision.as_dict()}
        result["deployment_identity"] = analysis.deployment.as_dict()
        result["followup"] = analysis.followup
        return result

    def tool_availability(self, campaign: BountyCampaign) -> dict[str, str]:
        from app.discovery.bounty.stateful import execution_enabled, tool_status

        diff_engine = next(
            (
                e
                for e in campaign.orchestrator.scheduler.engines
                if e.engine_id == "bugforge-compiler-diff"
            ),
            None,
        )
        solc = (
            "available"
            if diff_engine and diff_engine.availability().value == "available"
            else "unavailable"
        )
        tools = {
            "bugforge-research": "available",
            "bugforge-static": "available",
            "bugforge-compiler-diff": solc,
            "solc": solc,
        }
        fuzz = tool_status()
        tools["forge"] = fuzz.forge
        tools["stateful_execution"] = (
            "available" if (execution_enabled() and fuzz.available) else "unavailable"
        )
        for name in ("echidna", "medusa", "ityfuzz"):
            engine = next(
                (e for e in campaign.orchestrator.scheduler.engines if e.engine_id == name), None
            )
            tools[name] = engine.availability().value if engine is not None else "unavailable"
        return tools

    def report_pack(self, campaign_id: str) -> ReportPack:
        from app.discovery.bounty.advisories import match_advisories
        from app.discovery.bounty.findings import build_findings
        from app.discovery.bounty.priority import prioritize
        from app.discovery.bounty.report_pack import build_report_pack

        campaign = self.get(campaign_id)
        analysis = self._analysis(campaign)
        findings = build_findings(
            analysis.candidates,
            manifest=campaign.manifest,
            identity=campaign.identity,
            state=campaign.state,
            sequences=analysis.sequences,
            deployment_for=analysis.deployment_for,
            ambiguous_contracts=analysis.ambiguous_contracts,
            target_address=campaign.spec.address,
            target_chain_id=campaign.spec.chain_id,
            executions=self.stored_executions(campaign, analysis),
        )
        advisories = match_advisories(campaign.manifest.compiler, analysis.sources)
        differential = self._differential(campaign, analysis.sources)
        model = analysis.model
        priority = (
            prioritize(model, manifest=campaign.manifest, candidates=analysis.candidates)
            if model is not None
            else None
        )
        return build_report_pack(
            manifest=campaign.manifest,
            identity=campaign.identity,
            findings=findings,
            sequences=analysis.sequences,
            state=campaign.state,
            advisories=advisories,
            differential=differential,
            priority=priority,
            tools=self.tool_availability(campaign),
        )

    def _differential(
        self, campaign: BountyCampaign, sources: dict[str, str]
    ) -> DifferentialResult:
        """Real compiler differential when solc is installed and host compilation is on,
        else the stored prior result, else UNAVAILABLE. Never a hard-coded backend."""
        from app.discovery.bounty.compiler_diff import HostSolcBackend, run_differential

        backend = HostSolcBackend()
        if backend.available():
            result = run_differential(sources, backend=backend, expected=campaign.manifest.compiler)
            summary = {
                "status": result.status,
                "reason": result.reason,
                "backend": result.backend,
                "compiler_version": result.compiler_version,
                "differences": len(result.differences),
            }

            def put(artifacts: dict[str, Any]) -> list[str]:
                artifacts["compiler_differential"] = summary
                return []

            # Fail closed: the persistence state is recorded, never swallowed.
            state = self._persist_artifacts(campaign, put)
            campaign.cache["differential_persistence"] = state
            return result
        return run_differential(sources, backend=backend)

    # ---- fail-closed artifact persistence ----------------------------------------------

    def _persist_artifacts(
        self,
        campaign: BountyCampaign,
        merge: Callable[[dict[str, Any]], list[str]],
    ) -> dict[str, Any]:
        """Merge into the campaign's persisted artifacts and save, failing closed.

        ``merge`` must be additive and idempotent (it adds keyed, immutable evidence and
        returns the keys it refused to overwrite). Returns one of:

        * ``persisted`` -- written (possibly after reloading and reconciling);
        * ``persistence_conflict`` -- another writer kept winning; nothing overwritten;
        * ``persistence_unavailable`` -- the database refused or is down.

        Errors are never swallowed silently: the state is always returned to the caller.
        """
        with campaign.lock:
            draft = campaign.record.copy()
            refused = merge(draft.artifacts)
            try:
                campaign.record = self._records.update(draft)
                return {
                    "status": "persisted",
                    "revision": campaign.record.revision,
                    "reconciled": False,
                    "refused_overwrites": refused,
                }
            except CampaignConflictError:
                pass
            except CampaignPersistenceError as exc:
                return {"status": "persistence_unavailable", "reason": str(exc)[:200]}
            except PersistenceError as exc:
                return {"status": "persistence_unavailable", "reason": str(exc)[:200]}
            # Conflict: reload the latest record, reconcile immutable evidence, retry once.
            try:
                latest = self._records.get(campaign.campaign_id)
            except (CampaignPersistenceError, PersistenceError) as exc:
                return {"status": "persistence_unavailable", "reason": str(exc)[:200]}
            if latest is None:
                return {"status": "persistence_unavailable", "reason": "the record disappeared"}
            refused = merge(latest.artifacts)
            try:
                saved = self._records.update(latest)
            except CampaignConflictError:
                self._evict(campaign.campaign_id)
                return {
                    "status": "persistence_conflict",
                    "reason": "the campaign record kept changing; nothing was overwritten",
                }
            except (CampaignPersistenceError, PersistenceError) as exc:
                return {"status": "persistence_unavailable", "reason": str(exc)[:200]}
            campaign.record = saved
            return {
                "status": "persisted",
                "revision": saved.revision,
                "reconciled": True,
                "refused_overwrites": refused,
            }

    # ---- Phase 52 hardened stateful execution -------------------------------------------

    def _identity_context(self, campaign: BountyCampaign, analysis: CampaignAnalysis) -> Any:
        from app.discovery.bounty.stateful import IdentityContext

        deployment = analysis.deployment
        return IdentityContext(
            campaign_id=campaign.campaign_id,
            deployment=deployment.key,
            chain_id=deployment.chain_id,
            address=deployment.address,
            runtime_digest=deployment.runtime_digest,
            proxy_kind=deployment.proxy_kind or "none",
            implementation=deployment.implementation,
            compiler_configuration=campaign.manifest.compiler.fingerprint(),
            source_snapshot=campaign.manifest.source_commit,
            expected_hashes=dict(analysis.source_hashes),
        )

    def _current_sources(
        self, campaign: BountyCampaign, analysis: CampaignAnalysis, sequence_id: str
    ) -> dict[str, str]:
        """Re-read a sequence's exact source set from disk (so a changed file is caught)."""
        from app.discovery.bounty.analysis import read_sources

        recorded = analysis.sources_for(sequence_id)
        if not recorded:
            return {}
        current = read_sources(campaign.spec.repo_root, tuple(sorted(recorded)))
        # a file that vanished keeps its analyzed text out of the set: identity fails closed
        return current

    def stateful_execute(self, campaign_id: str, *, rounds: int = 3) -> dict[str, Any]:
        """Phase 52 (hardened): the executable closed loop over the campaign's VFCS plans.

        Every sequence runs against the exact source set it was built from (re-read
        and hash-checked), with its declared property. Results are structured; a
        sequence that ran without an oracle is execution only. Durable reproduction
        bundles go into the campaign's persisted artifacts, and persistence fails
        closed. Refused for stopped/paused campaigns and out-of-scope targets.
        """
        from app.discovery.bounty.campaign import ScopeStatus
        from app.discovery.bounty.engine_adapters import (
            PropertyEngine,
            engine_registry,
            engine_status_map,
            usable_property_engines,
        )
        from app.discovery.bounty.properties import build_property
        from app.discovery.bounty.solc_resolve import exact_version, resolve_solc
        from app.discovery.bounty.stateful import (
            StatefulExecutor,
            execution_enabled,
            run_feedback_loop,
            target_contract,
            tool_status,
        )
        from app.discovery.process import tool_path

        campaign = self.get(campaign_id)
        self._check_runnable(campaign)
        # Policy refusals come before tool availability: they hold whatever is installed.
        target_scope = campaign.manifest.scope_of(
            contract=campaign.spec.contract, file=campaign.spec.source_file
        )
        if target_scope.status is ScopeStatus.OUT_OF_SCOPE:
            raise CampaignControlError(f"the target is out of scope: {target_scope.reason}")
        tools = tool_status()
        unavailable = {
            "campaign_id": campaign_id,
            "available": False,
            "tools": tools.as_dict(),
            "verified": False,
        }
        if not execution_enabled():
            return {
                **unavailable,
                "reason": "local stateful execution is disabled",
                "blocker": "local stateful execution is disabled",
                "dynamic_foundry_execution_demonstrated": False,
            }
        required = exact_version(campaign.manifest.compiler.version)
        compiler = resolve_solc(required) if required else None
        if tool_path("forge") is None or (
            compiler is None and not required and tools.solc != "available"
        ):
            missing = "forge" if tool_path("forge") is None else "solc"
            return {
                **unavailable,
                "reason": f"{missing} is not installed; unavailable",
                "blocker": "Foundry or solc is unavailable",
                "dynamic_foundry_execution_demonstrated": False,
            }
        analysis = self._analysis(campaign)
        context = self._identity_context(campaign, analysis)
        runnable: list[Any] = []
        skipped: list[dict[str, str]] = []
        sources: dict[str, dict[str, str]] = {}
        specs: dict[str, Any] = {}
        for sequence in analysis.sequences:
            contract = target_contract(sequence)
            candidate = analysis.candidate_for(sequence.derived_from)
            source_file = candidate.file if candidate is not None else ""
            scope = campaign.manifest.scope_of(contract=contract, file=source_file)
            refusal = _execution_refusal(campaign, candidate, scope)
            if refusal is not None:
                code, detail = refusal
                skipped.append(
                    {
                        "sequence_id": sequence.sequence_id,
                        "derived_from": sequence.derived_from,
                        "contract": contract,
                        "file": source_file,
                        "detector": candidate.detector if candidate is not None else "",
                        "scope_status": scope.status.value,
                        "outcome": "skipped",
                        "reason": detail,
                        "reason_code": code,
                        "disposition": "skipped",
                        "forge_invoked": "false",
                        "campaign_id": campaign_id,
                    }
                )
                continue
            sources[sequence.sequence_id] = self._current_sources(
                campaign, analysis, sequence.sequence_id
            )
            model = analysis.models[sequence.sequence_id]
            specs[sequence.sequence_id] = build_property(sequence, model, candidate)
            runnable.append(sequence)
        supporting = self._supporting_detectors(analysis)
        registry = engine_registry()
        engine_status = engine_status_map(registry)
        engines = [PropertyEngine(name) for name in usable_property_engines(registry)]
        project = _project_settings(campaign.manifest.compiler) if compiler is not None else None
        profile_record, profile = _accepted_profile(campaign)
        executor = StatefulExecutor(
            compiler=compiler,
            project=project,
            compiler_required=required,
            profile=profile,
        )
        result = run_feedback_loop(
            executor,
            tuple(runnable),
            dict(analysis.models),
            sources,
            specs=specs,
            context=context,
            rounds=max(1, min(rounds, 5)),
            engine_status=engine_status,
            supporting=supporting,
            property_engines=engines,
        )
        bundles, blobs, executions = self._bundles(result, sources)
        registry_snapshot = [entry.as_dict() for entry in registry]
        persistence = self._persist_artifacts(
            campaign,
            lambda artifacts: _merge_stateful(
                artifacts,
                executions=executions,
                bundles=bundles,
                blobs=blobs,
                feedback=result.feedback,
                engines=registry_snapshot,
                skipped={item["sequence_id"]: item for item in skipped},
                run={
                    "at": _now(),
                    "outcomes": result.outcome_counts(),
                    "observations": len(result.observations),
                    "violations": len(result.violations),
                    "mechanism_observations": len(result.mechanism_observations),
                    "mutated": len(result.mutated),
                    "forge_runs": executor.runs,
                    "tools": _run_tools(tools, compiler, executor, required),
                    "constructor_profile": profile_record,
                },
            ),
        )
        campaign.cache.pop("coverage", None)
        from app.discovery.bounty.research_ledger import campaign_coverage

        summary = campaign_coverage(
            **self._coverage_facts(campaign, analysis),
            executions={**{item["sequence_id"]: item for item in skipped}, **executions},
            forge_runs=executor.runs,
            engines=registry_snapshot,
            persistence_status=str(persistence.get("status", "")),
            analysis_stale=self.analysis_stale(campaign),
            constructor_profile=profile_record,
        )
        return {
            "campaign_id": campaign_id,
            "available": True,
            "deployment": analysis.deployment.as_dict(),
            "scope_status": target_scope.status.value,
            "skipped": skipped,
            "forge_runs": executor.runs,
            "bundles": sorted(bundles),
            "persistence": persistence,
            "engines": registry_snapshot,
            "campaign_summary": summary,
            "blocker": summary["blocker"],
            "dynamic_foundry_execution_demonstrated": summary[
                "dynamic_foundry_execution_demonstrated"
            ],
            **result.as_dict(),
        }

    def _supporting_detectors(self, analysis: CampaignAnalysis) -> dict[str, tuple[str, ...]]:
        """Other detectors (different family) at the same function: supporting only."""
        by_site: dict[str, list[Any]] = {}
        for item in analysis.candidates:
            by_site.setdefault(f"{item.contract}.{item.function}", []).append(item)
        found: dict[str, tuple[str, ...]] = {}
        for site, items in by_site.items():
            for item in items:
                others = sorted({o.detector for o in items if o.family != item.family})
                if others:
                    found[f"{item.detector}@{site}"] = tuple(others[:4])
        return found

    def _bundles(
        self, result: Any, sources: dict[str, dict[str, str]]
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        from app.discovery.bounty.stateful import BUNDLED, build_bundle, parent_of

        bundles: dict[str, Any] = {}
        blobs: dict[str, str] = {}
        executions: dict[str, Any] = {}
        for obs in result.observations:
            sequence = result.sequences.get(obs.sequence_id)
            spec = result.specs.get(obs.sequence_id)
            if sequence is None or spec is None:
                continue
            exact = sources.get(obs.sequence_id) or sources.get(parent_of(sequence), {})
            bundle_id = ""
            if (
                obs.outcome in BUNDLED
                and obs.harness is not None
                and len(bundles) < MAX_BUNDLES_PER_RUN
            ):
                bundle = build_bundle(
                    obs,
                    sequence,
                    spec,
                    exact,
                    minimized=result.minimized.get(obs.sequence_id),
                    corroboration=result.independent.get(obs.sequence_id),
                    independent_runs=result.independent_runs.get(obs.sequence_id, ()),
                    created_at=_now(),
                )
                bundle_id = bundle["bundle_id"]
                bundles[bundle_id] = bundle
                for text in exact.values():
                    import hashlib

                    blobs[hashlib.sha256(text.encode("utf-8")).hexdigest()] = text
            corroboration = result.independent.get(obs.sequence_id)
            identity = obs.identity
            executions[obs.sequence_id] = {
                "sequence_id": obs.sequence_id,
                "derived_from": sequence.derived_from,
                "origin": sequence.origin,
                "outcome": obs.outcome.value,
                "reason": obs.reason[:200],
                "reason_code": obs.reason_code,
                "declaration": obs.declaration,
                "verdict": obs.verdict,
                "oracle_kind": obs.oracle_kind,
                "property_id": obs.property_id,
                "family": spec.family.value,
                "strong_candidate": obs.strong_candidate,
                "identity_status": identity.status if identity else "unknown",
                "source_set_digest": identity.source_set_digest if identity else "",
                "replay_mode": identity.replay_mode.value if identity else "",
                "corroboration": corroboration.status if corroboration else "not_applicable",
                "independence": (corroboration.independence if corroboration else "not_applicable"),
                "agreeing_paths": list(corroboration.agreeing) if corroboration else [],
                "check_paths": (
                    [
                        {
                            "path": path.path,
                            "engine": path.engine,
                            "material": path.material,
                            "verdict": path.verdict,
                        }
                        for path in corroboration.paths
                    ]
                    if corroboration
                    else []
                ),
                "harness_contracts": list(obs.harness.deployed) if obs.harness else [],
                "relationships": (
                    [item.as_dict() for item in obs.harness.relationships] if obs.harness else []
                ),
                "minimized_calls": (
                    len(result.minimized[obs.sequence_id].minimized)
                    if obs.sequence_id in result.minimized
                    else None
                ),
                "bundle_id": bundle_id,
                "process_status": obs.result.process_status if obs.result else None,
                "compiler_status": obs.result.compiler_status if obs.result else None,
                "test_status": obs.result.test_status if obs.result else None,
                "sequence_status": obs.result.sequence_status if obs.result else None,
                "property_status": obs.result.property_status if obs.result else None,
                "property_checked": obs.property_checked,
                "verified": False,
            }
        return bundles, blobs, executions

    def stateful_status(self, campaign_id: str) -> dict[str, Any]:
        """Persisted stateful results (durable across restarts). Nothing verified."""
        from app.discovery.bounty.stateful import replay_modes

        campaign = self.get(campaign_id)
        stored = dict(campaign.record.artifacts.get("stateful", {}))
        bundles = dict(campaign.record.artifacts.get("repro_bundles", {}))
        return {
            "campaign_id": campaign_id,
            "runs": list(stored.get("runs", [])),
            "executions": dict(stored.get("executions", {})),
            "skipped_sequences": dict(stored.get("skipped_sequences", {})),
            "feedback": dict(stored.get("feedback", {})),
            "engines": list(stored.get("engines", [])),
            "replay_modes": replay_modes(),
            "bundles": {
                bid: {
                    "schema": b.get("schema"),
                    "created_at": b.get("created_at"),
                    "sequence_id": dict(b.get("sequence", {})).get("sequence_id"),
                    "outcome": dict(b.get("execution", {})).get("outcome"),
                    "property_id": dict(b.get("property", {})).get("property_id"),
                    "artifact_hashes": b.get("artifact_hashes", {}),
                    "source_hashes": b.get("source_hashes", {}),
                }
                for bid, b in sorted(bundles.items())
            },
            "verified": False,
        }

    def repro_bundle(self, campaign_id: str, bundle_id: str) -> dict[str, Any]:
        """A full stored bundle plus the source blobs it references."""
        campaign = self.get(campaign_id)
        bundle = dict(campaign.record.artifacts.get("repro_bundles", {})).get(bundle_id)
        if bundle is None:
            raise CampaignError(f"no reproduction bundle {bundle_id!r}")
        blobs = dict(campaign.record.artifacts.get("source_blobs", {}))
        referenced = {
            digest: blobs.get(digest, "")
            for digest in dict(bundle.get("source_hashes", {})).values()
        }
        return {
            "campaign_id": campaign_id,
            "bundle": bundle,
            "sources": referenced,
            "self_contained": all(referenced.values()),
            "verified": False,
        }

    def _coverage_facts(
        self, campaign: BountyCampaign, analysis: CampaignAnalysis
    ) -> dict[str, Any]:
        """Selection, read, and model facts shared by the execute summary and the ledger."""
        from app.discovery.bounty.source_selection import Priority

        selection = campaign.selection
        model = analysis.model
        selected = set(selection.selected)
        scope = {item.path for item in selection.ranked if item.priority == Priority.SCOPE}
        unreadable: list[dict[str, str]] = []
        seen: set[str] = set()
        for record in (*analysis.file_reads, *selection.reads):
            if record.status == "read" or record.reason == "excluded" or record.path in seen:
                continue
            if record.path not in selected and record.path not in scope:
                continue
            seen.add(record.path)
            unreadable.append(record.as_dict())
        read_selected = sum(
            1
            for record in analysis.file_reads
            if record.path in selected and record.status == "read"
        )
        if not analysis.file_reads:
            read_selected = len(analysis.sources)
        selected_roles = {item.role: 0 for item in selection.ranked if item.path in selected}
        for item in selection.ranked:
            if item.path in selected:
                selected_roles[item.role] = selected_roles.get(item.role, 0) + 1
        return {
            "files_discovered": selection.considered,
            "target_files": selected_roles.get("target", 0),
            "compile_dependencies": selected_roles.get("compile_dependency", 0),
            "analysis_only_files": selected_roles.get("analysis_context", 0),
            "scope_excluded": len(selection.scope_excluded),
            "files_selected": len(selection.selected),
            "files_read": read_selected,
            "truncated": bool(analysis.truncated or selection.truncated),
            "selection_truncated": selection.truncated,
            "parser_limited": bool(model is not None and model.truncated),
            "truncation_notes": tuple(model.notes) if model is not None else (),
            "dependency_status": selection.dependency_status(),
            "selection_mode": selection.mode,
            "repository_candidates": selection.repository_candidates,
            "excluded_count": selection.excluded_count,
            "enumeration_capped": selection.enumeration_capped,
            "unreadable_files": unreadable,
            "contracts_modeled": len(model.contracts) if model is not None else 0,
            "functions_modeled": len(model.all_functions()) if model is not None else 0,
            "candidates": len(analysis.candidates),
            "sequences": len(analysis.sequences),
            "skipped": analysis.sequence_skipped,
        }

    def research_ledger(self, campaign_id: str) -> dict[str, Any]:
        """RESEARCH_COVERAGE + RESEARCH_GAPS from typed state. Read-only; nothing verified."""
        from app.discovery.bounty.research_ledger import build_ledger

        campaign = self.get(campaign_id)
        analysis = self._analysis(campaign)
        specs = self._property_specs(analysis)
        executions = self.stored_executions(campaign, analysis)
        ledger = build_ledger(
            models=analysis.all_models(),
            candidates=analysis.candidates,
            sequences=analysis.sequences,
            specs=specs,
            skipped=analysis.sequence_skipped,
            executions=executions,
            engines=self._engine_snapshot(campaign),
            manifest=campaign.manifest,
        )
        from app.discovery.bounty.research_ledger import campaign_coverage

        stored = dict(campaign.record.artifacts.get("stateful", {}))
        runs = list(stored.get("runs", []))
        forge_runs = sum(int(item.get("forge_runs") or 0) for item in runs) if runs else None
        stored_profile = runs[-1].get("constructor_profile") if runs else None
        summary = campaign_coverage(
            **self._coverage_facts(campaign, analysis),
            executions=executions,
            forge_runs=forge_runs,
            engines=self._engine_snapshot(campaign),
            persistence_status="persisted" if stored else "not_recorded",
            analysis_stale=self.analysis_stale(campaign),
            constructor_profile=stored_profile if isinstance(stored_profile, dict) else None,
        )
        return {"campaign_id": campaign_id, "campaign_summary": summary, **ledger}

    def evidence_graph(self, campaign_id: str) -> dict[str, Any]:
        """The derived evidence graph: provenance on every edge, contradictions visible."""
        from app.discovery.bounty.evidence_graph import build_evidence_graph

        campaign = self.get(campaign_id)
        analysis = self._analysis(campaign)
        specs = self._property_specs(analysis)
        graph = build_evidence_graph(
            candidates=analysis.candidates,
            sequences=analysis.sequences,
            specs=specs,
            executions=self.stored_executions(campaign, analysis),
            phase49_contradictions=self.evidence(campaign_id)["contradictions"],
        )
        return {"campaign_id": campaign_id, **graph}

    def research_plan(self, campaign_id: str) -> dict[str, Any]:
        """Cost-aware next research actions over the ledger (a recommendation only)."""
        from app.discovery.bounty.research_ledger import plan_actions
        from app.discovery.bounty.stateful import MAX_RUNS

        campaign = self.get(campaign_id)
        ledger = self.research_ledger(campaign_id)
        budget = campaign.state.budget
        plan = plan_actions(
            ledger,
            budget_remaining={name: budget.remaining(name) for name in sorted(budget.limits)},
            local_runs_remaining=MAX_RUNS,
        )
        return {
            "campaign_id": campaign_id,
            "ledger_digest": ledger["ledger_digest"],
            **plan,
        }

    def _property_specs(self, analysis: Any) -> dict[str, Any]:
        from app.discovery.bounty.properties import build_property

        return {
            sequence.sequence_id: build_property(
                sequence,
                analysis.models[sequence.sequence_id],
                analysis.candidate_for(sequence.derived_from),
            )
            for sequence in analysis.sequences
        }

    def research_properties(self, campaign_id: str) -> dict[str, Any]:
        """Each sequence's declared property and oracle, with its latest observation.

        Read-only. A property stays a hypothesis; an observation is a local execution
        result for the stated inputs, never a verification.
        """
        campaign = self.get(campaign_id)
        analysis = self._analysis(campaign)
        specs = self._property_specs(analysis)
        executions = self.stored_executions(campaign, analysis)
        properties = []
        for sequence in sorted(analysis.sequences, key=lambda item: item.sequence_id):
            spec = specs[sequence.sequence_id]
            record = dict(executions.get(sequence.sequence_id) or {})
            properties.append(
                {
                    "sequence_id": sequence.sequence_id,
                    "derived_from": sequence.derived_from,
                    **spec.as_dict(),
                    "observation": (
                        {
                            "outcome": record.get("outcome"),
                            "identity_status": record.get("identity_status"),
                            "replay_mode": record.get("replay_mode"),
                            "stale": bool(record.get("stale")),
                            "check_paths": list(record.get("check_paths", []) or []),
                            "bundle_id": record.get("bundle_id", ""),
                        }
                        if record
                        else None
                    ),
                    "verified": False,
                }
            )
        return {
            "campaign_id": campaign_id,
            "properties": properties,
            "skipped": [
                {"candidate": key, "reason": reason}
                for key, reason in sorted(analysis.sequence_skipped.items())
            ],
            "verified": False,
        }

    def research_modules(self, campaign_id: str) -> dict[str, Any]:
        """Which evidence-triggered protocol modules ran, with the triggering evidence."""
        from app.parsing.solidity_modules import module_triggers

        campaign = self.get(campaign_id)
        analysis = self._analysis(campaign)
        merged: dict[str, dict[str, Any]] = {}
        for model in analysis.all_models():
            for status in module_triggers(model):
                entry = merged.setdefault(
                    status.module,
                    {
                        "module": status.module,
                        "triggered": False,
                        "status": "not_triggered",
                        "detectors": list(status.detectors),
                        "evidence": [],
                    },
                )
                if status.triggered:
                    entry["triggered"] = True
                    entry["status"] = status.status
                    for file, line, what in status.evidence:
                        item = {"file": file, "line": line, "evidence": what}
                        if item not in entry["evidence"]:
                            entry["evidence"].append(item)
        return {
            "campaign_id": campaign_id,
            "modules": [merged[name] for name in sorted(merged)],
            "note": "a module runs only on evidence; triggered_no_detector makes no claim",
            "verified": False,
        }

    def research_engines(self, campaign_id: str) -> dict[str, Any]:
        """Engine availability: the snapshot the last run used, and the box's current view.

        The current view is not smoke-checked here (a GET runs no engine), so an
        engine that is present reads ``installed``; ``usable`` comes only from a run's
        smoke-checked snapshot.
        """
        from app.discovery.bounty.engine_adapters import engine_registry

        campaign = self.get(campaign_id)
        stored = dict(campaign.record.artifacts.get("stateful", {}))
        return {
            "campaign_id": campaign_id,
            "last_run": list(stored.get("engines", [])),
            "current": [entry.as_dict() for entry in engine_registry(smoke=False)],
            "verified": False,
        }

    def _engine_snapshot(self, campaign: BountyCampaign) -> list[dict[str, Any]]:
        from app.discovery.bounty.engine_adapters import engine_registry

        stored = list(dict(campaign.record.artifacts.get("stateful", {})).get("engines", []))
        if stored:
            return stored
        return [entry.as_dict() for entry in engine_registry(smoke=False)]

    def stored_executions(
        self, campaign: BountyCampaign, analysis: CampaignAnalysis
    ) -> dict[str, dict[str, Any]]:
        """Persisted executions whose source set still matches the analyzed snapshot."""
        stateful = dict(campaign.record.artifacts.get("stateful", {}))
        stored = dict(stateful.get("executions", {}))
        skipped = dict(stateful.get("skipped_sequences", {}))
        current = {sid: _digest_of(analysis.sources_for(sid)) for sid in analysis.sequence_sources}
        valid_digests = set(current.values())
        found: dict[str, dict[str, Any]] = {}
        for sid, item in {**skipped, **stored}.items():
            record = dict(item)
            if record.get("disposition") == "skipped" or record.get("outcome") == "skipped":
                record["stale"] = False
            else:
                record["stale"] = record.get("source_set_digest") not in valid_digests
            found[sid] = record
        return found


MAX_BUNDLES_PER_RUN = 12
MAX_STORED_BUNDLES = 24
MAX_BLOB_BYTES = 600_000
MAX_STORED_RUNS = 16
MAX_STORED_FEEDBACK = 256


def _run_tools(ambient: Any, compiler: Any, executor: Any, required: str) -> dict[str, str]:
    """The compiler this run bound, kept distinct from whatever ``solc`` is on PATH."""
    project = executor.project
    selected = str(compiler.reported_version) if compiler is not None else ""
    bound_version = str(executor.tools.solc_version) if selected else str(ambient.solc_version)

    def flag(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    return {
        **ambient.as_dict(),
        "solc_path": str(compiler.path) if compiler is not None else "",
        "solc_selected_version": selected,
        "solc_version": bound_version,
        "solc_ambient_version": str(ambient.solc_version),
        "compiler_required": required,
        "evm_requested": str(executor.evm_requested),
        "evm_exact": "true" if executor.evm_exact else "false",
        "evm_version": "" if project is None else str(project.evm_version),
        "optimizer": "" if project is None else flag(project.optimizer),
        "optimizer_runs": "" if project is None else flag(project.optimizer_runs),
        "via_ir": "" if project is None else flag(project.via_ir),
    }


def _project_settings(compiler: Any) -> Any:
    from app.discovery.bounty.stateful import ProjectSettings

    optimizer = {"enabled": True, "disabled": False}.get(compiler.optimizer)
    via_ir = {"true": True, "false": False}.get(compiler.via_ir)
    runs = int(compiler.optimizer_runs) if str(compiler.optimizer_runs).isdigit() else None
    evm = compiler.evm_version if compiler.evm_version not in {"", "unknown"} else ""
    return ProjectSettings(
        optimizer=optimizer,
        optimizer_runs=runs,
        via_ir=via_ir,
        evm_version=evm,
        evm_exact=True,
        evm_requested=evm,
    )


def _execution_refusal(campaign: Any, candidate: Any, scope: Any) -> tuple[str, str] | None:
    """Why a generated sequence must not reach Forge, or None when it may."""
    from app.discovery.bounty.campaign import ScopeStatus
    from app.discovery.bounty.findings import match_known_issue

    if scope.status is ScopeStatus.OUT_OF_SCOPE:
        return "blocked_by_policy:out_of_scope", scope.reason or "excluded by the manifest"
    if campaign.manifest.in_scope and scope.status is not ScopeStatus.IN_SCOPE:
        role = campaign.selection.role_of(candidate.file if candidate is not None else "")
        if role == "compile_dependency":
            return (
                "blocked_by_policy:dependency_only",
                "the file is a compile dependency and is not an eligible target",
            )
        if role == "analysis_context":
            return (
                "blocked_by_policy:analysis_only",
                "the file was read as additional context and is not an eligible target",
            )
        return (
            "blocked_by_policy:unknown_scope",
            "scope for this candidate is not established",
        )
    if candidate is not None:
        known = match_known_issue(campaign.manifest.known_issues, candidate)
        if known is not None:
            return (
                "blocked_by_policy:known_issue",
                f"matches known issue {known.issue_id}",
            )
        if str(candidate.detector).startswith("oracle."):
            return (
                "blocked_by_policy:oracle_testing_prohibited",
                "dynamic oracle testing is prohibited for this campaign",
            )
    return None


def _digest_of(sources: dict[str, str]) -> str:
    from app.discovery.bounty.stateful import sha256_text, source_set_digest

    return source_set_digest({path: sha256_text(text) for path, text in sources.items()})


def _merge_stateful(
    artifacts: dict[str, Any],
    *,
    executions: dict[str, Any],
    bundles: dict[str, Any],
    blobs: dict[str, str],
    feedback: tuple[dict[str, Any], ...],
    run: dict[str, Any],
    engines: list[dict[str, Any]] | None = None,
    skipped: dict[str, Any] | None = None,
) -> list[str]:
    """Additive, idempotent merge of one stateful run into persisted artifacts.

    Bundles and blobs are content-addressed and immutable: an existing key is never
    overwritten. Executions record the latest observation per sequence and keep the
    earlier one in ``history`` when the outcome changed. Feedback is keyed and
    idempotent. Every collection is bounded.
    """
    refused: list[str] = []
    stateful = artifacts.setdefault("stateful", {})
    runs = list(stateful.get("runs", []))
    if run not in runs:
        runs.append(run)
    stateful["runs"] = runs[-MAX_STORED_RUNS:]
    stored_exec = dict(stateful.get("executions", {}))
    for sid, record in executions.items():
        previous = stored_exec.get(sid)
        if previous is not None and previous.get("outcome") != record.get("outcome"):
            history = list(previous.get("history", []))[-4:]
            history.append({k: previous.get(k) for k in ("outcome", "verdict", "bundle_id")})
            record = {**record, "history": history}
        stored_exec[sid] = record
    stateful["executions"] = stored_exec
    if skipped is not None:
        stored_skips = dict(stateful.get("skipped_sequences", {}))
        for sid, record in skipped.items():
            stored_skips[sid] = dict(record)
        stateful["skipped_sequences"] = stored_skips
    stored_feedback = dict(stateful.get("feedback", {}))
    for item in feedback:
        key = str(item.get("key", ""))
        if key and key not in stored_feedback and len(stored_feedback) < MAX_STORED_FEEDBACK:
            stored_feedback[key] = dict(item)
    stateful["feedback"] = stored_feedback
    if engines is not None:
        # the registry snapshot the latest run used (status is per box, so latest wins)
        stateful["engines"] = list(engines)
    stored_blobs = dict(artifacts.get("source_blobs", {}))
    size = sum(len(text) for text in stored_blobs.values())
    stored_bundles = dict(artifacts.get("repro_bundles", {}))
    for bundle_id, bundle in sorted(bundles.items()):
        if bundle_id in stored_bundles:
            if stored_bundles[bundle_id].get("artifact_hashes") != bundle.get("artifact_hashes"):
                refused.append(bundle_id)
            continue
        if len(stored_bundles) >= MAX_STORED_BUNDLES:
            refused.append(f"{bundle_id}:capacity")
            continue
        needed = {
            digest: blobs[digest]
            for digest in dict(bundle.get("source_hashes", {})).values()
            if digest in blobs and digest not in stored_blobs
        }
        extra = sum(len(text) for text in needed.values())
        if size + extra > MAX_BLOB_BYTES:
            refused.append(f"{bundle_id}:source_capacity")
            continue
        stored_blobs.update(needed)
        size += extra
        stored_bundles[bundle_id] = bundle
    artifacts["source_blobs"] = stored_blobs
    artifacts["repro_bundles"] = stored_bundles
    return refused


def _evidence_dict(item: Any) -> dict[str, Any]:
    return {
        "evidence_id": item.evidence_id,
        "kind": item.kind,
        "quality": item.quality.value,
        "engine": item.engine,
        "capability": item.capability,
        "identity_key": item.identity_key,
        "polarity": item.polarity,
        "verified": item.attrs.get("verified", "false"),
    }


def _finding_dict(item: Any) -> dict[str, Any]:
    from app.discovery.bounty.findings import quality_of

    return {
        "finding_id": item.finding_id,
        "title": item.title,
        "detector": item.detector,
        "family": item.family,
        "contract": item.contract,
        "function": item.function,
        "file": item.file,
        "line": item.line,
        "scope_status": item.scope_status,
        "severity_candidate": item.severity_candidate,
        "confirmed_severity": item.confirmed_severity,
        "poc_status": item.poc_status,
        "evidence_quality": item.evidence_quality,
        "sequence_ids": list(item.sequence_ids),
        "open_contradictions": list(item.open_contradictions),
        "known_issue": item.known_issue.issue_id if item.known_issue else "",
        "duplicate_of": item.duplicate_of,
        "duplicate_basis": item.duplicate_basis,
        "root_cause_key": item.root_cause_key,
        "qualification": item.qualification.status,
        "deployment": dict(item.deployment),
        "execution_status": item.execution_status,
        "candidate_strength": item.candidate_strength,
        "property_ids": list(item.property_ids),
        "bundle_ids": list(item.bundle_ids),
        "corroboration": item.corroboration,
        "quality": quality_of(item.candidate_strength, item.corroboration),
        "verified": item.verified,
        "submitted": item.submitted,
    }


_SERVICE: BountyCampaignService | None = None
_SERVICE_LOCK = threading.Lock()


def get_bounty_service() -> BountyCampaignService:
    """Return the process-wide campaign service (lazily created from settings)."""
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = BountyCampaignService()
        return _SERVICE


__all__ = [
    "APPROVABLE_CAPABILITIES",
    "BountyCampaign",
    "BountyCampaignService",
    "CampaignConflictError",
    "CampaignControlError",
    "CampaignError",
    "CampaignPersistenceError",
    "CampaignSpec",
    "ENGINE_VERSION",
    "OrchestratorState",
    "PersistenceError",
    "UnknownCampaignError",
    "get_bounty_service",
]
