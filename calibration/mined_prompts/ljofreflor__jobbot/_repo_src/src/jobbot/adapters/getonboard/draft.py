"""Permanent Get on Board profile fields from Candidate (facts only; Spanish).

GoB stores these texts on the professional account and reuses them on later
applications. JobBot is the maintainer: regenerate from profile.yaml, HITL paste
into https://www.getonbrd.com/webpros/edit, never invent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from jobbot.branding import MARK, stamp_description
from jobbot.models.candidate import Candidate
from jobbot.models.experience import Experience
from jobbot.models.job import JobPosting

EXPERIENCE_MIN = 300
EXPERIENCE_MAX = 2000
EDUCATION_MIN = 100
EDUCATION_MAX = 2000

JOBBOT_SIGNATURE = MARK


@dataclass(frozen=True)
class PermanentProfileFields:
    """Fields GoB keeps on the account for reuse across applications."""

    experiencia_y_perfil: str
    formacion_academica: str
    headline: str
    skills: list[str]
    signature: str = JOBBOT_SIGNATURE

    def to_dict(self) -> dict[str, object]:
        return {
            "experiencia_y_perfil": self.experiencia_y_perfil,
            "formacion_academica": self.formacion_academica,
            "headline": self.headline,
            "skills": list(self.skills),
            "signature": self.signature,
            "limits": {
                "experiencia_chars": len(self.experiencia_y_perfil),
                "formacion_chars": len(self.formacion_academica),
                "experiencia_max": EXPERIENCE_MAX,
                "formacion_max": EDUCATION_MAX,
            },
        }


def build_permanent_profile_fields(candidate: Candidate) -> PermanentProfileFields:
    """Derive permanent GoB blurbs from profile.yaml facts only."""
    experience = stamp_description(
        _fit(_draft_experience(candidate), EXPERIENCE_MAX),
        max_len=EXPERIENCE_MAX,
    )
    education = _fit(_draft_education(candidate), EDUCATION_MAX)
    return PermanentProfileFields(
        experiencia_y_perfil=experience,
        formacion_academica=education,
        headline=candidate.personal.headline,
        skills=list(candidate.skills.all_skills())[:10],
    )


def draft_getonboard_fields(
    candidate: Candidate,
    job: JobPosting | None = None,
    *,
    permanent: PermanentProfileFields | None = None,
) -> dict[str, str]:
    """Application-form fields; prefer permanent profile when available.

    A stored permanent profile keeps its wording, but it may predate profile.yaml
    (a new role, an end date), so it is refined against the profile before use.
    """
    _ = job
    if permanent is None:
        fields = build_permanent_profile_fields(candidate)
    else:
        from jobbot.nlp.refine import refine_permanent_profile

        fields = refine_permanent_profile(candidate, permanent).fields
    return {
        "experiencia_y_perfil": fields.experiencia_y_perfil,
        "formacion_academica": fields.formacion_academica,
        "signature": fields.signature,
    }


def render_permanent_profile_markdown(fields: PermanentProfileFields) -> str:
    return "\n".join(
        [
            "# Get on Board — perfil permanente (mantenedor)",
            "",
            "Fuente: data/profile.yaml + iteración previa (computación acumulativa).",
            "No se descarta el texto anterior: se refina con hechos actuales.",
            "La descripción cierra con la firma. Pegala junto con el texto.",
            f"Firma: {fields.signature}",
            "",
            "URL: https://www.getonbrd.com/webpros/edit",
            "",
            "Orden HITL:",
            "1. Abre Editar perfil y reemplaza textos viejos (Matlab/VTK/Ceamos, etc.).",
            "2. Pega los bloques de abajo (GoB los guarda para futuras postulaciones).",
            "3. Tus CVs: sube output/base/cv.pdf y márcalo default.",
            "4. Recién entonces postula a un cargo.",
            "",
            f"## Headline ({len(fields.headline)} chars)",
            "",
            fields.headline,
            "",
            f"## experiencia_y_perfil ({len(fields.experiencia_y_perfil)} chars; "
            f"límite {EXPERIENCE_MIN}–{EXPERIENCE_MAX})",
            "",
            fields.experiencia_y_perfil,
            "",
            f"## formacion_academica ({len(fields.formacion_academica)} chars; "
            f"límite {EDUCATION_MIN}–{EDUCATION_MAX})",
            "",
            fields.formacion_academica,
            "",
            "## Skills (máx. 10 en GoB)",
            "",
            ", ".join(fields.skills) if fields.skills else "_(none)_",
            "",
        ]
    )


def render_getonboard_markdown(fields: dict[str, str], *, job: JobPosting) -> str:
    """Per-job paste sheet (reuses permanent body)."""
    return "\n".join(
        [
            "# Get on Board — textos para postulación (español)",
            f"# Job: {job.id} {job.title} @ {job.company}",
            "# Preferir perfil permanente actualizado en webpros/edit",
            f"# La descripción ya cierra con la firma: {fields.get('signature', JOBBOT_SIGNATURE)}",
            "",
            f"## experiencia_y_perfil ({len(fields['experiencia_y_perfil'])} chars)",
            "",
            fields["experiencia_y_perfil"],
            "",
            f"## formacion_academica ({len(fields['formacion_academica'])} chars)",
            "",
            fields["formacion_academica"],
            "",
        ]
    )


def permanent_profile_dir(root_output: Path) -> Path:
    return root_output / "getonboard"


def permanent_profile_yaml_path(root_output: Path) -> Path:
    return permanent_profile_dir(root_output) / "profile_permanent.yaml"


def permanent_profile_md_path(root_output: Path) -> Path:
    return permanent_profile_dir(root_output) / "profile_permanent.md"


def save_permanent_profile(
    fields: PermanentProfileFields,
    root_output: Path,
    *,
    archive_previous: bool = True,
) -> Path:
    out = permanent_profile_dir(root_output)
    out.mkdir(parents=True, exist_ok=True)
    yaml_path = permanent_profile_yaml_path(root_output)
    md_path = permanent_profile_md_path(root_output)
    if archive_previous and yaml_path.is_file():
        history = out / "history"
        history.mkdir(parents=True, exist_ok=True)
        from datetime import UTC, datetime

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archived = history / f"profile_permanent_{stamp}.yaml"
        archived.write_text(yaml_path.read_text(encoding="utf-8"), encoding="utf-8")
    yaml_path.write_text(
        yaml.safe_dump(fields.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    md_path.write_text(render_permanent_profile_markdown(fields), encoding="utf-8")
    return md_path


def fields_from_seed_text(
    *,
    experiencia: str,
    formacion: str,
    candidate: Candidate,
) -> PermanentProfileFields:
    """Seed a PermanentProfileFields from pasted portal text (accumulated capital)."""
    return PermanentProfileFields(
        experiencia_y_perfil=stamp_description(experiencia, max_len=EXPERIENCE_MAX),
        formacion_academica=formacion.strip() or _fit(_draft_education(candidate), EDUCATION_MAX),
        headline=candidate.personal.headline,
        skills=list(candidate.skills.all_skills())[:10],
    )


def load_permanent_profile(root_output: Path) -> PermanentProfileFields | None:
    path = permanent_profile_yaml_path(root_output)
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return None
    exp = str(raw.get("experiencia_y_perfil") or "").strip()
    edu = str(raw.get("formacion_academica") or "").strip()
    if not exp or not edu:
        return None
    skills = raw.get("skills") or []
    if not isinstance(skills, list):
        skills = []
    return PermanentProfileFields(
        experiencia_y_perfil=exp,
        formacion_academica=edu,
        headline=str(raw.get("headline") or ""),
        skills=[str(s) for s in skills][:10],
        signature=str(raw.get("signature") or JOBBOT_SIGNATURE),
    )


def recent_roles(candidate: Candidate, limit: int = 2) -> list[Experience]:
    """Current roles first, then by end and start date; the YAML order is not a fact."""
    ordered = sorted(
        candidate.experience,
        key=lambda exp: (exp.current, exp.end_date or "", exp.start_date),
        reverse=True,
    )
    return ordered[:limit]


def role_label(exp: Experience) -> str:
    """How a draft names a role, before its period."""
    return f"{exp.title} en {exp.company}"


def role_head(exp: Experience) -> str:
    """Opening sentence of a role paragraph: who, where and the period profile.yaml states."""
    end = "actualidad" if exp.current else (exp.end_date or "")
    period = ""
    if exp.start_date:
        period = f" ({exp.start_date}–{end})" if end else f" ({exp.start_date})"
    return f"{role_label(exp)}{period}."


def _draft_experience(candidate: Candidate) -> str:
    paragraphs: list[str] = []

    for exp in recent_roles(candidate):
        head = role_head(exp)
        # One achievement per role keeps two roles under the GoB character cap.
        achs = [a.text.strip() for a in exp.achievements[:1] if a.text.strip()]
        if achs:
            paragraphs.append(f"{head} {' '.join(achs)}")
        elif exp.description:
            paragraphs.append(f"{head} {exp.description.strip()}")
        else:
            paragraphs.append(head)

    skills = candidate.skills.all_skills()
    if skills:
        # Which skills matter is a question only this profile can answer: the ones
        # its own headline, summary and achievements already talk about come first.
        emphasis = " ".join(
            [
                candidate.personal.headline or "",
                candidate.summary or "",
                *(
                    f"{exp.title} {exp.description or ''} "
                    + " ".join(a.text for a in exp.achievements)
                    for exp in candidate.experience
                ),
                *candidate.specialties,
            ]
        ).casefold()
        highlighted = [skill for skill in skills if skill.casefold() in emphasis]
        shown = (highlighted + [s for s in skills if s not in highlighted])[:8]
        paragraphs.append("Stack habitual: " + ", ".join(shown) + ".")

    if candidate.summary:
        # The summary leads, but the roles own their space: it gets what is left of
        # the GoB cap, cut on a sentence so the portal never shows a dangling clause.
        room = EXPERIENCE_MAX - len("\n\n".join(paragraphs)) - 2
        summary = _sentence_prefix(candidate.summary.strip(), room)
        if summary:
            paragraphs.insert(0, summary)

    return "\n\n".join(paragraphs).strip()


def _sentence_prefix(text: str, maximum: int) -> str:
    """Longest prefix of whole sentences that fits, '' when not even one does."""
    if maximum <= 0:
        return ""
    if len(text) <= maximum:
        return text
    window = text[:maximum]
    cut = max(window.rfind(". "), window.rfind(".\n"))
    if cut > 0:
        return window[: cut + 1]
    return window if window.endswith(".") else ""


def _draft_education(candidate: Candidate) -> str:
    parts: list[str] = []
    edu_lines: list[str] = []
    for edu in candidate.education:
        line = f"{edu.degree} — {edu.institution}"
        dates = "–".join(d for d in (edu.start_date, edu.end_date) if d)
        if dates:
            line += f" ({dates})"
        if edu.details:
            line += f". {edu.details.strip()}"
        edu_lines.append(line)
    if edu_lines:
        parts.append(". ".join(edu_lines) + ".")

    teaching: list[str] = []
    for exp in candidate.experience:
        title_l = exp.title.casefold()
        if any(
            k in title_l
            for k in (
                "profesor",
                "docente",
                "ayudante",
                "coordinador académico",
                "investigador",
            )
        ):
            teaching.append(f"{exp.title} en {exp.company}")
    if teaching:
        parts.append("He sido " + "; ".join(teaching) + ".")

    pubs: list[str] = []
    for pub in candidate.publications[:3]:
        bits = [pub.title]
        if pub.journal:
            bits.append(pub.journal)
        if pub.year:
            bits.append(str(pub.year))
        if pub.doi:
            bits.append(f"DOI {pub.doi}")
        pubs.append(" — ".join(bits))
    if pubs:
        parts.append("Publicaciones relevantes: " + "; ".join(pubs) + ".")

    return "\n\n".join(parts).strip()


def _fit(text: str, maximum: int) -> str:
    """Keep under max by dropping trailing paragraphs (never invent)."""
    text = text.strip()
    if len(text) <= maximum:
        return text
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    kept: list[str] = []
    for part in parts:
        trial = "\n\n".join([*kept, part])
        if len(trial) <= maximum:
            kept.append(part)
        else:
            break
    if kept:
        return "\n\n".join(kept)
    # Single long paragraph: hard cut at word boundary
    cut = text[: maximum - 1].rsplit(" ", 1)[0]
    return cut
