"""Propagate the CV outward: local artifacts + permanent portal profiles.

Planning here is read-only and browser-free so a dry-run costs nothing. Writes stay
in the existing adapters, behind ``--apply`` and per-destination confirmation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum

from jobbot.adapters.diff_engine import build_sync_plan, load_snapshot
from jobbot.browser.sessions import SessionState, SessionStatus, session_for
from jobbot.config import JobbotConfig
from jobbot.models.candidate import Candidate


class PropagationTarget(StrEnum):
    CV = "cv"
    GETONBOARD = "getonboard"
    INDEED = "indeed"
    LINKEDIN = "linkedin"


# Local artifacts first: the portals reuse the CV that this step rebuilds.
# These are permanent profiles (your standing presence), not company career sites.
DEFAULT_TARGETS: tuple[PropagationTarget, ...] = (
    PropagationTarget.CV,
    PropagationTarget.GETONBOARD,
    PropagationTarget.INDEED,
    PropagationTarget.LINKEDIN,
)

# Names that look like "push my CV to every ATS we know". Writes for those land via
# `cv sync` (issue #43) / `companies signup` (#44), not via `cv propagate --targets`.
_NOT_PROPAGATION: frozenset[str] = frozenset(
    {
        "companies",
        "company",
        "portals",
        "portal",
        "ats",
        "greenhouse",
        "lever",
        "ashby",
        "workday",
        "all-companies",
        "all_companies",
        "everywhere",
    }
)


@dataclass(frozen=True)
class TargetPlan:
    """What propagation would do for one destination, and what blocks it."""

    target: PropagationTarget
    operations: list[str] = field(default_factory=list)
    blocked_reason: str | None = None
    hint: str | None = None
    note: str | None = None
    session: SessionState | None = None

    @property
    def ready(self) -> bool:
        return self.blocked_reason is None

    @property
    def actionable(self) -> bool:
        return self.ready and bool(self.operations)


class UnknownTargetError(ValueError):
    """Raised for a --targets value JobBot cannot propagate to."""


def parse_targets(raw: str) -> list[PropagationTarget]:
    """Parse ``all`` / ``permanent`` or a comma-separated list, in canonical order.

    ``all`` and ``permanent`` are the same thing: the standing profiles (CV, Get on
    Board, Indeed, LinkedIn). They are not the company career sites in
    ``data/companies.yaml`` — those are reached per job via ``application apply``.
    """
    text = (raw or "").strip().casefold()
    if not text or text in {"all", "permanent"}:
        return list(DEFAULT_TARGETS)
    wanted: set[PropagationTarget] = set()
    for chunk in text.split(","):
        name = chunk.strip()
        if not name:
            continue
        if name in _NOT_PROPAGATION:
            msg = (
                f"{name!r} is not a `cv propagate` target. "
                "`cv propagate` updates permanent profiles only "
                "(all|permanent|cv,getonboard,indeed,linkedin). "
                "For active company portals use `jobbot cv sync` "
                "(HITL --apply for active sites; signup sheet is #44) "
                "or `jobbot companies signup NOMBRE` / `jobbot application apply Jxxxx`."
            )
            raise UnknownTargetError(msg)
        try:
            wanted.add(PropagationTarget(name))
        except ValueError as exc:
            valid = ", ".join(t.value for t in DEFAULT_TARGETS)
            msg = (
                f"unknown propagation target {name!r} "
                f"(use all|permanent or: {valid})"
            )
            raise UnknownTargetError(msg) from exc
    return [target for target in DEFAULT_TARGETS if target in wanted]


def plan_cv(config: JobbotConfig, candidate: Candidate) -> TargetPlan:
    """Local CV artifacts every portal step reuses."""
    if not config.templates_dir.is_dir():
        return TargetPlan(
            target=PropagationTarget.CV,
            blocked_reason=f"templates dir not found: {config.templates_dir}",
            hint="check paths.templates in .jobbot.toml",
        )
    base = config.output_dir / "base"
    return TargetPlan(
        target=PropagationTarget.CV,
        operations=[
            f"rebuild {base / 'cv.tex'} + {base / 'cv.pdf'}",
            f"rebuild {base / 'cv_ats.txt'}",
        ],
        note=None if (base / "cv.pdf").is_file() else "no PDF yet; this is the first build",
    )


def plan_indeed(
    config: JobbotConfig,
    candidate: Candidate,
    *,
    section: str = "all",
) -> TargetPlan:
    """Diff profile.yaml against the stored Indeed snapshot (no browser)."""
    external = load_snapshot(config.output_dir, "indeed")
    if external is None:
        return TargetPlan(
            target=PropagationTarget.INDEED,
            blocked_reason="no Indeed snapshot stored, so nothing to diff against",
            hint="jobbot indeed pull",
        )
    sec = None if section in {None, "", "all"} else section
    plan = build_sync_plan("indeed", candidate, external, section=sec)
    return TargetPlan(
        target=PropagationTarget.INDEED,
        operations=[
            f"{op.op.value} {op.section}.{op.field} — {op.label}" for op in plan.actionable
        ],
        note=f"snapshot captured {external.captured_at.date().isoformat()}",
    )


def plan_linkedin(config: JobbotConfig, candidate: Candidate) -> TargetPlan:
    """Publications are the only writable LinkedIn section."""
    from jobbot.adapters.linkedin.package import build_linkedin_publication_items

    items = build_linkedin_publication_items(candidate)
    if not items:
        return TargetPlan(
            target=PropagationTarget.LINKEDIN,
            note="no publications in profile.yaml",
        )
    return TargetPlan(
        target=PropagationTarget.LINKEDIN,
        operations=[f"add publication — {item.title}" for item in items],
        note="assumes remote empty; --apply fetches your remote titles first",
    )


def plan_getonboard(config: JobbotConfig, candidate: Candidate) -> TargetPlan:
    """Permanent Get on Board profile: cumulative refine, then HITL paste."""
    from jobbot.adapters.getonboard.draft import load_permanent_profile

    previous = load_permanent_profile(config.output_dir)
    out = config.output_dir / "getonboard"
    first_time = previous is None
    return TargetPlan(
        target=PropagationTarget.GETONBOARD,
        operations=[
            (
                f"create {out / 'profile_permanent.md'}"
                if first_time
                else f"refine {out / 'profile_permanent.md'} (cumulative, keeps prior text)"
            ),
            f"rewrite {out / 'sync_package.md'}",
            "write profile + experience + education on getonbrd.com/webpros/edit",
        ],
        note="--apply reads what the portal holds first; empty drafts never wipe it",
    )


BROWSER_TARGETS: frozenset[PropagationTarget] = frozenset(
    {
        PropagationTarget.INDEED,
        PropagationTarget.LINKEDIN,
        PropagationTarget.GETONBOARD,
    }
)


def with_session(
    plan: TargetPlan,
    state: SessionState | None,
    *,
    cdp_url: str | None = None,
) -> TargetPlan:
    """Fold the session preflight into a plan: suggest an endpoint, never attach to it."""
    if state is None:
        return plan
    if plan.blocked_reason is not None or cdp_url:
        return replace(plan, session=state)
    if state.profile_busy:
        return replace(plan, blocked_reason=state.evidence, hint=state.hint, session=state)
    if state.status in {SessionStatus.READY, SessionStatus.NEEDS_LOGIN}:
        return replace(
            plan,
            note=_join_notes(plan.note, state.evidence),
            hint=state.hint,
            session=state,
        )
    return replace(plan, session=state)


def plan_propagation(
    config: JobbotConfig,
    candidate: Candidate,
    *,
    targets: list[PropagationTarget] | None = None,
    section: str = "all",
    sessions: Sequence[SessionState] | None = None,
    cdp_url: str | None = None,
) -> list[TargetPlan]:
    wanted = targets if targets is not None else list(DEFAULT_TARGETS)
    plans: list[TargetPlan] = []
    for target in wanted:
        if target == PropagationTarget.CV:
            plans.append(plan_cv(config, candidate))
        elif target == PropagationTarget.INDEED:
            plans.append(plan_indeed(config, candidate, section=section))
        elif target == PropagationTarget.LINKEDIN:
            plans.append(plan_linkedin(config, candidate))
        elif target == PropagationTarget.GETONBOARD:
            plans.append(plan_getonboard(config, candidate))
    if sessions is None:
        return plans
    return [
        with_session(plan, session_for(sessions, plan.target.value), cdp_url=cdp_url)
        if plan.target in BROWSER_TARGETS
        else plan
        for plan in plans
    ]


def _join_notes(*notes: str | None) -> str | None:
    parts = [note for note in notes if note]
    return "; ".join(parts) if parts else None


def summarize_plans(plans: list[TargetPlan]) -> str:
    """One line for the narration: which destinations have work waiting."""
    actionable = [p for p in plans if p.actionable]
    blocked = [p for p in plans if not p.ready]
    if not actionable and not blocked:
        return "nothing to propagate"
    bits = [
        f"{plan.target.value}: {len(plan.operations)}" for plan in actionable
    ]
    if blocked:
        bits.append("blocked: " + ", ".join(plan.target.value for plan in blocked))
    return "; ".join(bits)
