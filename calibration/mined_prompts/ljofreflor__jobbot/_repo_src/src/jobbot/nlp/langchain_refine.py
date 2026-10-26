"""Optional LangChain-backed cumulative refine (facts from Candidate only)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from jobbot.adapters.getonboard.draft import (
    EDUCATION_MAX,
    EXPERIENCE_MAX,
    PermanentProfileFields,
    build_permanent_profile_fields,
)
from jobbot.branding import stamp_description
from jobbot.models.candidate import Candidate
from jobbot.nlp.gateway import FACTS_ONLY_RULES

logger = logging.getLogger("jobbot.nlp.langchain")

_SYSTEM = (
    "Eres un editor de perfil profesional en español para Get on Board.\n"
    "Reglas:\n"
    "- El texto PREVIO es capital computacional: mejóralo, no lo reescribas "
    "desde cero.\n"
    f"- {FACTS_ONLY_RULES}\n"
    "- Si PREVIO menciona tecnologías/empresas obsoletas que no están en "
    "FACTS, reescríbelas o elimínalas.\n"
    "- Integra hechos nuevos de FACTS que falten en PREVIO.\n"
    "- Respeta límites de caracteres.\n"
    "- Responde SOLO JSON con keys: experiencia_y_perfil, formacion_academica.\n"
)


class LangChainProfileRefiner:
    """Refine previous GoB blurbs with an LLM via LangChain (optional extra)."""

    def refine(
        self,
        candidate: Candidate,
        previous: PermanentProfileFields | None,
    ) -> PermanentProfileFields:
        cold = build_permanent_profile_fields(candidate)
        if previous is None:
            previous = cold

        llm = _build_chat_model()
        facts = _facts_payload(candidate)
        user = (
            f"LIMITES: experiencia<={EXPERIENCE_MAX}, formacion<={EDUCATION_MAX}\n\n"
            f"FACTS:\n{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n"
            f"PREVIO experiencia_y_perfil:\n{previous.experiencia_y_perfil}\n\n"
            f"PREVIO formacion_academica:\n{previous.formacion_academica}\n\n"
            "Devuelve JSON mejorado (computación acumulativa sobre PREVIO)."
        )
        raw = llm.invoke(
            [
                ("system", _SYSTEM),
                ("human", user),
            ]
        )
        content = _message_content(raw)
        parsed = _parse_json_object(content)
        exp = str(parsed.get("experiencia_y_perfil") or previous.experiencia_y_perfil).strip()
        edu = str(parsed.get("formacion_academica") or previous.formacion_academica).strip()
        exp, edu = _ground_or_fallback(exp, edu, candidate, previous, cold)
        return PermanentProfileFields(
            experiencia_y_perfil=stamp_description(exp, max_len=EXPERIENCE_MAX),
            formacion_academica=edu[:EDUCATION_MAX],
            headline=candidate.personal.headline,
            skills=list(candidate.skills.all_skills())[:10],
            signature=previous.signature,
        )


def _build_chat_model() -> Any:
    from jobbot.nlp.gateway import build_chat_model

    return build_chat_model(temperature=0.2)


def _facts_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        "headline": candidate.personal.headline,
        "summary": candidate.summary,
        "skills": candidate.skills.all_skills(),
        "experience": [
            {
                "company": e.company,
                "title": e.title,
                "start": e.start_date,
                "end": e.end_date,
                "current": e.current,
                "achievements": [a.text for a in e.achievements[:3]],
            }
            for e in candidate.experience[:6]
        ],
        "education": [
            {
                "institution": e.institution,
                "degree": e.degree,
                "details": e.details,
                "start": e.start_date,
                "end": e.end_date,
            }
            for e in candidate.education
        ],
        "publications": [
            {"title": p.title, "journal": p.journal, "year": p.year, "doi": p.doi}
            for p in candidate.publications[:5]
        ],
    }


def _message_content(raw: Any) -> str:
    content = getattr(raw, "content", raw)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        msg = "LLM output is not a JSON object"
        raise ValueError(msg)
    return data


def _ground_or_fallback(
    exp: str,
    edu: str,
    candidate: Candidate,
    previous: PermanentProfileFields,
    cold: PermanentProfileFields,
) -> tuple[str, str]:
    """Reject LLM output that invents companies not in Candidate."""
    allowed = {e.company.casefold() for e in candidate.experience}
    for e in candidate.experience:
        for part in re.split(r"[/|,]", e.company):
            if part.strip():
                allowed.add(part.strip().casefold())
    # Also allow institution names
    for ed in candidate.education:
        allowed.add(ed.institution.casefold())

    def _ok(text: str) -> bool:
        # Heuristic: if text mentions a Capitalized multi-word Org-like token
        # not in allowed and not in previous/cold, treat as invention risk.
        # Keep simple: block known-bad leftovers only + require at least one allowed company.
        bad = ("ceamos", "adexus", "adacom", "mmgeo")
        lower = text.casefold()
        if any(b in lower for b in bad):
            return False
        return any(a in lower for a in allowed if len(a) >= 4)

    if not _ok(exp):
        logger.warning("LLM experience failed grounding; keeping cumulative/cold mix")
        exp = (
            previous.experiencia_y_perfil
            if previous.experiencia_y_perfil
            else cold.experiencia_y_perfil
        )
    if not edu.strip():
        edu = previous.formacion_academica or cold.formacion_academica
    return exp, edu
