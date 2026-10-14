"""Generate candidate API tests. Execution requires the authorization chain."""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.evidence import Evidence, EvidenceKind
from app.security_testing.api_spec import APISpec
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.errors import AuthorizationDeniedError
from app.security_testing.failures import ToolExecutionResult, ToolExecutionState


@dataclass(frozen=True)
class CandidateAPITest:
    title: str
    method: str
    url: str
    hypothesis: str
    category: str
    headers: tuple[tuple[str, str], ...] = ()
    body: str | None = None
    active: bool = True


class APITestGenerator:
    def generate(self, spec: APISpec, *, base_url: str | None = None) -> list[CandidateAPITest]:
        root = (base_url or spec.base_url).rstrip("/")
        tests: list[CandidateAPITest] = []
        for endpoint in spec.endpoints:
            url = _join(root, endpoint.path)
            tests.append(
                CandidateAPITest(
                    title=f"Auth omitted on {endpoint.method} {endpoint.path}",
                    method=endpoint.method,
                    url=url,
                    hypothesis="Endpoint may respond without authentication",
                    category="authentication",
                )
            )
            if endpoint.method in {"GET", "HEAD"}:
                tests.append(
                    CandidateAPITest(
                        title=f"Unexpected method POST on {endpoint.path}",
                        method="POST",
                        url=url,
                        hypothesis="Endpoint may accept an undocumented method",
                        category="unexpected_methods",
                    )
                )
            if (
                any(p.location == "path" and "id" in p.name.lower() for p in endpoint.parameters)
                or "{id}" in endpoint.path
                or "{userId}" in endpoint.path
            ):
                other = endpoint.path.replace("{id}", "2").replace("{userId}", "2")
                tests.append(
                    CandidateAPITest(
                        title=f"Object identifier swap on {endpoint.path}",
                        method=endpoint.method,
                        url=_join(root, other),
                        hypothesis="Authorization may not bind the object to the caller",
                        category="authorization",
                    )
                )
            tests.append(
                CandidateAPITest(
                    title=f"Invalid input on {endpoint.method} {endpoint.path}",
                    method=endpoint.method,
                    url=url,
                    hypothesis="Input validation may be missing",
                    category="input_validation",
                    body='{"fuzz":"\' OR 1=1"}',
                )
            )
        return tests


class APITestRunner:
    def __init__(self, engine: SecurityTestEngine) -> None:
        self.engine = engine
        self.generator = APITestGenerator()

    async def execute(
        self, tests: list[CandidateAPITest], *, require_human: bool = False
    ) -> list[Evidence] | ToolExecutionResult:
        evidence: list[Evidence] = []
        client = self.engine.http("api_test")
        for test in tests:
            if require_human:
                from app.security_testing.approvals import ApprovalKind

                if not self.engine.approvals.is_granted(ApprovalKind.SEND_POC_REQUEST):
                    return ToolExecutionResult(
                        tool="api_test",
                        state=ToolExecutionState.APPROVAL_REQUIRED,
                        detail="PoC-class API tests require SEND_POC_REQUEST approval",
                    )
            try:
                result = await client.request(test.method, test.url, content=test.body)
            except AuthorizationDeniedError as exc:
                evidence.append(
                    Evidence(
                        kind=EvidenceKind.API_TEST,
                        source="api_test",
                        summary=f"Denied {test.method} {test.url}: {exc.reason}",
                    )
                )
                continue
            if isinstance(result, ToolExecutionResult):
                if result.state is ToolExecutionState.DRY_RUN:
                    evidence.append(
                        Evidence(
                            kind=EvidenceKind.API_TEST,
                            source="api_test",
                            summary=f"Dry-run {test.method} {test.url}: {test.hypothesis}",
                        )
                    )
                    continue
                continue
            evidence.append(
                Evidence(
                    kind=EvidenceKind.API_TEST,
                    source="api_test",
                    summary=(
                        f"{test.title}: status={result.response_status} "
                        f"(hypothesis, not a verified finding)"
                    ),
                    details=test.hypothesis,
                    metadata={
                        "category": test.category,
                        "status": str(result.response_status or ""),
                    },
                )
            )
        self.engine.add_evidence(evidence)
        return evidence


def _join(root: str, path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not root:
        return path
    if not path.startswith("/"):
        path = "/" + path
    return root.rstrip("/") + path
