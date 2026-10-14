"""OpenAI evidence annotation for checklist items the marker attributor satisfied.

The model is told which items are already observed and asked only to quote the
note excerpts that support each one, through a JSON schema. It never decides
whether an item is satisfied: items the markers left unobserved are not sent, and
any item key or excerpt it returns that is not an observed item or a verbatim
substring of the note is dropped. Provider or payload failures raise so the
caller can fall back to marker evidence.
"""

import json
from typing import Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gvas.config import OpenAISettings
from gvas.domain.reporting import (
    ChecklistEvidence,
    ChecklistEvidenceAnnotation,
    ChecklistEvidenceRequest,
    ChecklistOutcome,
)

CHAT_COMPLETIONS_PATH: Final = "/chat/completions"
SCHEMA_NAME: Final = "checklist_evidence_excerpts"
MAX_EXCERPTS_PER_ITEM: Final = 5
MAX_EXCERPT_CHARS: Final = 400

SYSTEM_PROMPT: Final = """You attach supporting evidence to checklist items behind a field-note
capture workflow that turns a field technician's dictated notes into an inspection
field-note report (asbestos / lead / mold / PCB / IAQ / pre-demo inspections).

You receive the transcript of what the technician said and a list of checklist
items that a separate step has ALREADY confirmed are satisfied, each with the
evidence it already cites.

Your only job: for each listed item, quote the transcript passage(s) that show
the item was addressed, so a reader of the report can see where the finding came
from.

Rules:
- Quote exactly. Every excerpt must be copied character-for-character from the
  transcript; do not paraphrase, merge, correct spelling, or add words.
- Never add an item that is not in the list and never drop one because you doubt
  it; whether an item is satisfied is not your decision.
- Prefer one or two short, complete phrases per item over long passages.
- Return an empty excerpt list for an item when nothing beyond the cited
  evidence supports it."""

RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["item_key", "excerpts"],
                "properties": {
                    "item_key": {"type": "string"},
                    "excerpts": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}


class ChecklistEvidenceAnnotationError(RuntimeError):
    """Raised when the annotation pass failed and marker evidence should stand alone."""


class _AnnotatedItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    item_key: str
    excerpts: tuple[str, ...] = Field(default_factory=tuple)


class _AnnotatedItems(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    items: tuple[_AnnotatedItem, ...] = Field(default_factory=tuple)


class OpenAIChecklistEvidenceAnnotator:
    """Runs the evidence annotation pass through the chat-completions endpoint."""

    def __init__(self, settings: OpenAISettings, client: httpx.AsyncClient) -> None:
        if not settings.is_configured:
            raise ChecklistEvidenceAnnotationError("openai api key is not configured")
        self._settings = settings
        self._client = client

    async def annotate(
        self, request: ChecklistEvidenceRequest, attributed: tuple[ChecklistEvidence, ...]
    ) -> tuple[ChecklistEvidenceAnnotation, ...]:
        observed = tuple(item for item in attributed if item.outcome is ChecklistOutcome.OBSERVED)
        if not observed:
            return ()
        try:
            response = await self._client.post(
                f"{self._settings.api_base_url}{CHAT_COMPLETIONS_PATH}",
                json=_request_body(self._settings.review_model, request, observed),
                headers={"Authorization": f"Bearer {self._settings.api_key}"},
                timeout=self._settings.timeout_seconds,
            )
        except httpx.HTTPError as error:
            raise ChecklistEvidenceAnnotationError("openai was unreachable") from error
        return _annotations(response, request, observed)


def _request_body(
    model: str, request: ChecklistEvidenceRequest, observed: tuple[ChecklistEvidence, ...]
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_content(request, observed)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": RESPONSE_SCHEMA},
        },
    }


def _user_content(
    request: ChecklistEvidenceRequest, observed: tuple[ChecklistEvidence, ...]
) -> str:
    return json.dumps(
        {
            "satisfied_items": [
                {
                    "item_key": item.item_key,
                    "prompt": item.prompt,
                    "cited_evidence": list(item.evidence),
                }
                for item in observed
            ],
            "transcript": request.canonical_transcript,
        },
        ensure_ascii=False,
    )


def _annotations(
    response: httpx.Response,
    request: ChecklistEvidenceRequest,
    observed: tuple[ChecklistEvidence, ...],
) -> tuple[ChecklistEvidenceAnnotation, ...]:
    if response.status_code >= 400:
        raise ChecklistEvidenceAnnotationError(f"openai returned http {response.status_code}")
    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("message content is not text")
        reported = _AnnotatedItems.model_validate_json(content)
    except (KeyError, IndexError, TypeError, ValueError, ValidationError) as error:
        raise ChecklistEvidenceAnnotationError(
            "openai returned an unreadable evidence annotation"
        ) from error
    observed_keys = {item.item_key for item in observed}
    transcript = request.canonical_transcript
    annotations: list[ChecklistEvidenceAnnotation] = []
    seen: set[str] = set()
    for item in reported.items:
        key = item.item_key.strip()
        if key in seen or key not in observed_keys:
            continue
        seen.add(key)
        excerpts = tuple(
            dict.fromkeys(
                excerpt
                for raw in item.excerpts
                if (excerpt := raw.strip())
                and len(excerpt) <= MAX_EXCERPT_CHARS
                and excerpt in transcript
            )
        )[:MAX_EXCERPTS_PER_ITEM]
        if excerpts:
            annotations.append(ChecklistEvidenceAnnotation(item_key=key, excerpts=excerpts))
    return tuple(annotations)
