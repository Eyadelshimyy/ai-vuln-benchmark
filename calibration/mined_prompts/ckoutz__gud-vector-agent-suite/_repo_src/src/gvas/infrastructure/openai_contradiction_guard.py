"""OpenAI contradiction pass over a review the primary reviewer found complete.

The model is asked one narrow question -- does the transcript contain a hard
contradiction that would leave the report wrong as-is -- and answers through a
JSON schema. It never decides which checklist items are missing; that stays with
the primary reviewer. Provider or payload failures raise so the dispatcher
retries the review command instead of marking the note ready.
"""

import json
from datetime import UTC, datetime
from typing import Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gvas.config import OpenAISettings
from gvas.domain.completeness import (
    ChecklistItemKey,
    CompletenessReviewRequest,
    ContradictionGuardOutcome,
    DetectedContradiction,
)
from gvas.domain.usage import UsageKind, UsageLedgerPort

CHAT_COMPLETIONS_PATH: Final = "/chat/completions"
SCHEMA_NAME: Final = "field_note_contradictions"
MAX_QUESTION_CHARS: Final = 400

SYSTEM_PROMPT: Final = """You are the contradiction check behind a field-note capture workflow that
turns a field technician's dictated notes into an inspection field-note report
(asbestos / lead / mold / PCB / IAQ / pre-demo inspections).

You receive the checklist items the report is built around, the transcript of what
the technician said, and any answers the technician already gave to follow-up
questions. A separate step has already confirmed every required item is present.

Your only job: report a hard, clear contradiction that would make the report
actively wrong if left as-is. Example: the stated inspection type is "Mold" while
every finding is explicitly about asbestos material.

Bias strongly toward reporting nothing:
- Minor ambiguity, a loose end, or an unresolved-but-plausible reference is NOT a
  contradiction.
- Optional detail that was never mentioned is NOT a contradiction.
- A later statement that corrects an earlier one is a correction, not a
  contradiction, when the technician clearly meant to update it.
- A follow-up answer the technician already gave resolves the conflict it answers.

For each hard contradiction, name the checklist item it falls under (using the
item key exactly as given), write one short question phrased the way a colleague
would ask over radio, and quote the transcript spans that conflict as the detail.
Do not invent facts. Return an empty list when there is nothing to report."""

RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["contradictions"],
    "properties": {
        "contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["item_key", "question", "detail"],
                "properties": {
                    "item_key": {"type": "string"},
                    "question": {"type": "string"},
                    "detail": {"type": "string"},
                },
            },
        }
    },
}


class ContradictionGuardError(RuntimeError):
    """Raised when the contradiction pass should be retried by the dispatcher."""


class _ReportedContradiction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    item_key: str
    question: str
    detail: str


class _ReportedContradictions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    contradictions: tuple[_ReportedContradiction, ...] = Field(default_factory=tuple)


class OpenAIContradictionGuard:
    """Runs the contradiction pass through the chat-completions endpoint."""

    def __init__(
        self,
        settings: OpenAISettings,
        client: httpx.AsyncClient,
        usage_ledger: UsageLedgerPort | None = None,
    ) -> None:
        if not settings.is_configured:
            raise ContradictionGuardError("openai api key is not configured")
        self._settings = settings
        self._client = client
        self._usage_ledger = usage_ledger

    async def detect(self, request: CompletenessReviewRequest) -> ContradictionGuardOutcome:
        try:
            response = await self._client.post(
                f"{self._settings.api_base_url}{CHAT_COMPLETIONS_PATH}",
                json=_request_body(self._settings.review_model, request),
                headers={"Authorization": f"Bearer {self._settings.api_key}"},
                timeout=self._settings.timeout_seconds,
            )
        except httpx.HTTPError as error:
            raise ContradictionGuardError("openai was unreachable") from error
        outcome = _outcome(response, request)
        if self._usage_ledger is not None:
            await self._usage_ledger.record(
                request.business_id,
                UsageKind.REVIEW_TOKENS,
                _total_tokens(response),
                at=datetime.now(UTC),
            )
        return outcome


def _total_tokens(response: httpx.Response) -> int:
    """Input plus output tokens as reported by the provider; 0 when it reports none."""

    try:
        usage = response.json().get("usage")
    except ValueError:
        return 0
    if not isinstance(usage, dict):
        return 0
    total = 0
    for key in ("prompt_tokens", "completion_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            total += value
    return total


def _request_body(model: str, request: CompletenessReviewRequest) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_content(request)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": RESPONSE_SCHEMA},
        },
    }


def _user_content(request: CompletenessReviewRequest) -> str:
    return json.dumps(
        {
            "checklist": [
                {
                    "item_key": item.key,
                    "prompt": item.prompt,
                    "requirement": item.requirement.value,
                }
                for item in request.checklist.items
            ],
            "transcript": request.transcript_text,
            "answers": [
                {"item_key": answer.item_key, "text": answer.text} for answer in request.answers
            ],
        },
        ensure_ascii=False,
    )


def _outcome(
    response: httpx.Response, request: CompletenessReviewRequest
) -> ContradictionGuardOutcome:
    if response.status_code >= 400:
        raise ContradictionGuardError(f"openai returned http {response.status_code}")
    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("message content is not text")
        reported = _ReportedContradictions.model_validate_json(content)
    except (KeyError, IndexError, TypeError, ValueError, ValidationError) as error:
        raise ContradictionGuardError("openai returned an unreadable contradiction pass") from error
    contradictions: list[DetectedContradiction] = []
    seen: set[str] = set()
    for item in reported.contradictions:
        key = item.item_key.strip()
        question = " ".join(item.question.split())[:MAX_QUESTION_CHARS]
        detail = item.detail.strip()
        if (
            key in seen
            or request.checklist.item(ChecklistItemKey(key)) is None
            or not question
            or not detail
        ):
            continue
        seen.add(key)
        contradictions.append(
            DetectedContradiction(item_key=ChecklistItemKey(key), question=question, detail=detail)
        )
    return ContradictionGuardOutcome(contradictions=tuple(contradictions))
