import json
import logging
from uuid import uuid4

import httpx
import pytest

from gvas.application.checklist_evidence import MarkerChecklistEvidenceAttributor
from gvas.application.guarded_checklist_evidence import GuardedChecklistEvidenceAttributor
from gvas.config import OpenAISettings
from gvas.domain.identifiers import BusinessId
from gvas.domain.reporting import (
    ChecklistEvidence,
    ChecklistEvidenceAnnotation,
    ChecklistEvidenceRequest,
    ChecklistOutcome,
    CorrelatedAnswer,
)
from gvas.infrastructure.openai_checklist_evidence import (
    ChecklistEvidenceAnnotationError,
    OpenAIChecklistEvidenceAnnotator,
)
from test_completeness import checklist

OPENAI_KEY = "sk-test"
TRANSCRIPT = "site: north depot, boiler room. Fibrous pipe insulation seen, sampled two spots."


class FakeAnnotator:
    def __init__(self, *annotations: ChecklistEvidenceAnnotation, error: Exception | None = None):
        self._annotations = annotations
        self._error = error
        self.calls: list[tuple[ChecklistEvidenceRequest, tuple[ChecklistEvidence, ...]]] = []

    async def annotate(
        self, request: ChecklistEvidenceRequest, attributed: tuple[ChecklistEvidence, ...]
    ) -> tuple[ChecklistEvidenceAnnotation, ...]:
        self.calls.append((request, attributed))
        if self._error is not None:
            raise self._error
        return self._annotations


def request(transcript: str = TRANSCRIPT, *answers: CorrelatedAnswer) -> ChecklistEvidenceRequest:
    business_id = BusinessId(uuid4())
    return ChecklistEvidenceRequest(
        business_id=business_id,
        case_id=uuid4(),
        checklist=checklist(business_id),
        canonical_transcript=transcript,
        correlated_answers=answers,
    )


def openai_settings() -> OpenAISettings:
    return OpenAISettings(api_key=OPENAI_KEY, review_model="review-model")


def completion(content: object) -> dict[str, object]:
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps(content)}}]}


def by_key(evidence: tuple[ChecklistEvidence, ...]) -> dict[str, ChecklistEvidence]:
    return {item.item_key: item for item in evidence}


@pytest.mark.asyncio
async def test_annotation_excerpts_are_appended_to_marker_evidence() -> None:
    annotator = FakeAnnotator(
        ChecklistEvidenceAnnotation(item_key="site", excerpts=("boiler room",)),
    )
    attributor = GuardedChecklistEvidenceAttributor(MarkerChecklistEvidenceAttributor(), annotator)

    evidence = by_key(await attributor.attribute(request()))

    assert evidence["site"].outcome is ChecklistOutcome.OBSERVED
    assert evidence["site"].evidence == ("site: north depot, boiler room.", "boiler room")
    assert evidence["work"].outcome is ChecklistOutcome.NOT_OBSERVED
    assert evidence["work"].evidence == ()
    _, attributed = annotator.calls[0]
    assert by_key(attributed)["site"].evidence == ("site: north depot, boiler room.",)


@pytest.mark.asyncio
async def test_annotations_never_flip_outcomes_or_add_evidence_to_unsatisfied_items() -> None:
    annotator = FakeAnnotator(
        ChecklistEvidenceAnnotation(item_key="work", excerpts=("sampled two spots",)),
        ChecklistEvidenceAnnotation(item_key="unknown", excerpts=("boiler room",)),
    )
    attributor = GuardedChecklistEvidenceAttributor(MarkerChecklistEvidenceAttributor(), annotator)

    guarded = await attributor.attribute(request())

    assert guarded == await MarkerChecklistEvidenceAttributor().attribute(request(TRANSCRIPT))


@pytest.mark.asyncio
async def test_hallucinated_excerpts_are_rejected_in_the_application_layer() -> None:
    annotator = FakeAnnotator(
        ChecklistEvidenceAnnotation(
            item_key="site",
            excerpts=("Boiler Room", "north depot boiler", "  boiler room  ", "boiler room"),
        ),
    )
    attributor = GuardedChecklistEvidenceAttributor(MarkerChecklistEvidenceAttributor(), annotator)

    evidence = by_key(await attributor.attribute(request()))

    assert evidence["site"].evidence == ("site: north depot, boiler room.", "boiler room")


@pytest.mark.asyncio
async def test_annotator_error_falls_back_to_marker_evidence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    annotator = FakeAnnotator(error=ChecklistEvidenceAnnotationError("openai returned http 500"))
    attributor = GuardedChecklistEvidenceAttributor(MarkerChecklistEvidenceAttributor(), annotator)

    with caplog.at_level(logging.WARNING, logger="gvas.application.guarded_checklist_evidence"):
        guarded = await attributor.attribute(request())

    assert guarded == await MarkerChecklistEvidenceAttributor().attribute(request())
    assert any("annotation failed" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_openai_annotator_sends_only_observed_items_and_keeps_verbatim_excerpts() -> None:
    seen: list[httpx.Request] = []

    def handle(http_request: httpx.Request) -> httpx.Response:
        seen.append(http_request)
        return httpx.Response(
            200,
            json=completion(
                {
                    "items": [
                        {
                            "item_key": " site ",
                            "excerpts": [
                                "boiler room",
                                "Boiler Room",
                                "north depot, boiler room",
                                "boiler room",
                                "the technician sampled",
                            ],
                        },
                        {"item_key": "work", "excerpts": ["sampled two spots"]},
                        {"item_key": "site", "excerpts": ["Fibrous"]},
                    ]
                }
            ),
        )

    attributed = await MarkerChecklistEvidenceAttributor().attribute(request())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        annotations = await OpenAIChecklistEvidenceAnnotator(openai_settings(), client).annotate(
            request(), attributed
        )

    assert annotations == (
        ChecklistEvidenceAnnotation(
            item_key="site", excerpts=("boiler room", "north depot, boiler room")
        ),
    )
    http_request = seen[0]
    assert http_request.url.path.endswith("/chat/completions")
    assert http_request.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    body = json.loads(http_request.read())
    assert body["model"] == "review-model"
    assert body["response_format"]["json_schema"]["strict"] is True
    content = json.loads(body["messages"][1]["content"])
    assert [item["item_key"] for item in content["satisfied_items"]] == ["site"]
    assert content["transcript"] == TRANSCRIPT


@pytest.mark.asyncio
async def test_openai_annotator_skips_the_call_when_nothing_is_observed() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    attributed = await MarkerChecklistEvidenceAttributor().attribute(request("nothing marked"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        annotations = await OpenAIChecklistEvidenceAnnotator(openai_settings(), client).annotate(
            request("nothing marked"), attributed
        )

    assert annotations == ()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"error": {"message": f"bad key {OPENAI_KEY}"}}),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json=completion({"items": "not a list"})),
        httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}),
    ],
)
@pytest.mark.asyncio
async def test_openai_annotator_failures_raise_sanitized_errors(response: httpx.Response) -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        return response

    attributed = await MarkerChecklistEvidenceAttributor().attribute(request())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ChecklistEvidenceAnnotationError) as raised:
            await OpenAIChecklistEvidenceAnnotator(openai_settings(), client).annotate(
                request(), attributed
            )

    assert OPENAI_KEY not in str(raised.value)


@pytest.mark.asyncio
async def test_openai_annotator_network_failure_raises() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    attributed = await MarkerChecklistEvidenceAttributor().attribute(request())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ChecklistEvidenceAnnotationError, match="unreachable"):
            await OpenAIChecklistEvidenceAnnotator(openai_settings(), client).annotate(
                request(), attributed
            )


def test_openai_annotator_requires_a_configured_key() -> None:
    with pytest.raises(ChecklistEvidenceAnnotationError):
        OpenAIChecklistEvidenceAnnotator(OpenAISettings(api_key=""), httpx.AsyncClient())
