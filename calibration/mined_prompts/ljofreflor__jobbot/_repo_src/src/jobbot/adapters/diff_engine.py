"""Diff Candidate vs ExternalProfile."""

from __future__ import annotations

import json
from pathlib import Path

from jobbot.branding import MARK, stamp_description, strip_mark
from jobbot.models.candidate import Candidate
from jobbot.models.external_profile import ExternalProfile
from jobbot.models.sync import SyncOperation, SyncOpType, SyncPlan


def load_snapshot(output_dir: Path, source: str) -> ExternalProfile | None:
    path = output_dir / "snapshots" / source / "profile.json"
    if not path.is_file():
        return None
    return ExternalProfile.model_validate_json(path.read_text(encoding="utf-8"))


def save_snapshot(profile: ExternalProfile, output_dir: Path) -> Path:
    path = output_dir / "snapshots" / profile.source / "profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return path


def build_diff_operations(
    candidate: Candidate,
    external: ExternalProfile,
    *,
    section: str | None = None,
) -> list[SyncOperation]:
    ops: list[SyncOperation] = []
    sections = {section} if section else {"headline", "summary", "experience", "skills"}

    if "headline" in sections:
        ops.append(
            _cmp(
                "headline",
                "headline",
                external.headline,
                candidate.personal.headline,
            )
        )
    if "summary" in sections:
        ops.append(
            _cmp(
                "summary",
                "summary",
                external.summary,
                _published_summary(external.summary, candidate.summary),
            )
        )
    if "skills" in sections:
        local_skills = {s.casefold(): s for s in candidate.skills.all_skills()}
        remote = {s.casefold(): s for s in external.skills}
        for key, label in local_skills.items():
            if key in remote:
                ops.append(
                    SyncOperation(
                        op=SyncOpType.SAME,
                        section="skills",
                        field=label,
                        before=remote[key],
                        after=label,
                        label=label,
                    )
                )
            else:
                ops.append(
                    SyncOperation(
                        op=SyncOpType.ADD,
                        section="skills",
                        field=label,
                        after=label,
                        label=label,
                    )
                )
        for key, label in remote.items():
            if key not in local_skills:
                ops.append(
                    SyncOperation(
                        op=SyncOpType.REMOVE,
                        section="skills",
                        field=label,
                        before=label,
                        label=label,
                    )
                )
    if "experience" in sections:
        local_keys = {
            (e.company.casefold(), e.title.casefold()): e for e in candidate.experience
        }
        remote_keys = {
            ((e.company or "").casefold(), (e.title or "").casefold())
            for e in external.experience
        }
        for (_company_key, _title_key), exp in local_keys.items():
            label = f"{exp.company} / {exp.title}"
            if (_company_key, _title_key) in remote_keys:
                ops.append(
                    SyncOperation(
                        op=SyncOpType.SAME,
                        section="experience",
                        field="role",
                        before=label,
                        after=label,
                        label=exp.company,
                    )
                )
            else:
                ops.append(
                    SyncOperation(
                        op=SyncOpType.ADD,
                        section="experience",
                        field="role",
                        after=label,
                        label=exp.company,
                    )
                )
    return ops


def build_sync_plan(
    target: str,
    candidate: Candidate,
    external: ExternalProfile,
    *,
    section: str | None = None,
    allow_removals: bool = False,
) -> SyncPlan:
    ops = build_diff_operations(candidate, external, section=section)
    if not allow_removals:
        ops = [o for o in ops if o.op != SyncOpType.REMOVE]
    # Sync only actionable non-SAME
    actionable = [o for o in ops if o.op in {SyncOpType.ADD, SyncOpType.CHANGE}]
    return SyncPlan(target=target, operations=actionable)


def render_profile_diff(
    candidate: Candidate,
    output_dir: Path,
    source: str,
    *,
    section: str | None = None,
) -> str:
    external = load_snapshot(output_dir, source)
    if external is None:
        return f"No {source} snapshot found. Run: jobbot {source} pull"
    ops = build_diff_operations(candidate, external, section=section)
    lines = [f"{source.upper()} PROFILE DIFF"]
    if section:
        relevant = [o for o in ops if o.section == section]
        if all(o.op == SyncOpType.SAME for o in relevant) and relevant:
            return f"No difference for {section}."
    for op in ops:
        mark = {
            SyncOpType.SAME: "=",
            SyncOpType.ADD: "+",
            SyncOpType.REMOVE: "-",
            SyncOpType.CHANGE: "!",
            SyncOpType.UNKNOWN: "?",
        }[op.op]
        if op.op == SyncOpType.CHANGE:
            lines.append(f"{op.section} / {op.field}")
            lines.append(f"- {source}: {op.before}")
            lines.append(f"+ Local: {op.after}")
        elif op.op == SyncOpType.SAME:
            lines.append(f"= {op.label or op.field}")
        else:
            lines.append(f"{mark} {op.section} {op.label or op.field}: {op.after or op.before}")
    if not ops:
        lines.append("No differences detected.")
    return "\n".join(lines)


def _published_summary(remote: str | None, local: str | None) -> str:
    """Summary Jobbot would leave: same words, plus the mark once."""
    stamped = stamp_description(local)
    remote_text = (remote or "").strip()
    same_words = strip_mark(remote_text) == strip_mark(local)
    if same_words and MARK.casefold() in remote_text.casefold():
        return remote_text
    return stamped or (local or "").strip()


def _cmp(
    section: str,
    field: str,
    before: str | None,
    after: str | None,
    label: str = "",
) -> SyncOperation:
    b = (before or "").strip()
    a = (after or "").strip()
    if not b and a:
        op = SyncOpType.ADD
    elif b and not a:
        op = SyncOpType.REMOVE
    elif b == a:
        op = SyncOpType.SAME
    elif not b and not a:
        op = SyncOpType.UNKNOWN
    else:
        op = SyncOpType.CHANGE
    return SyncOperation(
        op=op,
        section=section,
        field=field,
        before=before,
        after=after,
        label=label or field,
    )


def dump_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "model_dump_json"):
        path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
