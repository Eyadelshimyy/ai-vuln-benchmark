"""Conservative authentication, authorization, and control-missing indicators."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.analyzers.framework_detector import FrameworkInfo
from app.domain.security import VulnerabilityClass
from app.parsing.model import SyntaxGraph
from app.security.rules.base import RuleDocumentation, SecurityObservation, SecurityRule

JWT_DOC = RuleDocumentation(
    detects="JWT or token decoding with signature verification disabled.",
    evidence="Matching source line.",
    limitations="Does not prove the token is attacker-controlled.",
    false_positives="Local test helpers that decode unsigned fixtures.",
)
AUTH_CMP_DOC = RuleDocumentation(
    detects="Direct equality comparison against a password or secret.",
    evidence="Matching comparison line.",
    limitations="Does not prove the comparison is on a login path.",
    false_positives="Comparisons against empty-string guards or dummy values.",
)
CSRF_DOC = RuleDocumentation(
    detects="CSRF protection explicitly disabled on a handler.",
    evidence="Decorator or assignment line.",
    limitations="Framework defaults are not modeled.",
    false_positives="Internal APIs that use another anti-CSRF control.",
)
AUTHZ_DOC = RuleDocumentation(
    detects="Object lookup by request id without an adjacent ownership check.",
    evidence="Lookup call site.",
    limitations=(
        "Intra-procedural only. Middleware and authorization in another file "
        "are not modeled. A function name containing admin is not evidence."
    ),
    false_positives="Lookups filtered by the current principal on the same statement.",
)

_PASSWORD_EQ = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(password|passwd|secret_key|api_secret)(?![A-Za-z0-9_])"
    r"\s*==\s*([A-Za-z_][A-Za-z0-9_]*)"
)
_HASH_NAME = re.compile(r"(?i)hash|digest|checksum")
_CSRF_NAMES = frozenset({"WTF_CSRF_ENABLED", "csrf_protect", "csrf_protection", "CSRF_ENABLED"})
_CSRF_DECORATOR = re.compile(r"(?i)(?:^|\.)csrf_exempt$")
_ADMIN_DECORATOR = re.compile(
    r"(?i)admin_required|require_admin|staff_member_required|permission_required"
)
_IDOR = re.compile(
    r"(get_object_or_404|find_by_id|findById|objects\.get)\s*\([^)\n]*(?:id|pk)\s*=",
    re.IGNORECASE,
)
_OWNER = re.compile(
    r"request\.user|current_user|g\.user|principal|owner\s*==|user_id\s*==",
    re.IGNORECASE,
)
_GUARD = re.compile(r"\b(return|raise)\b")


class JwtUnverifiedRule(SecurityRule):
    rule_id = "sec.auth.jwt_unverified"
    vulnerability_class = VulnerabilityClass.AUTHENTICATION
    documentation = JWT_DOC

    def check(
        self,
        graph: SyntaxGraph,
        *,
        frameworks: Sequence[FrameworkInfo] = (),
    ) -> list[SecurityObservation]:
        del frameworks
        return _jwt_hits(self, graph)


class PasswordCompareRule(SecurityRule):
    rule_id = "sec.auth.password_compare"
    vulnerability_class = VulnerabilityClass.AUTHENTICATION
    documentation = AUTH_CMP_DOC

    def check(
        self,
        graph: SyntaxGraph,
        *,
        frameworks: Sequence[FrameworkInfo] = (),
    ) -> list[SecurityObservation]:
        del frameworks
        return _password_hits(self, graph)


class CsrfDisabledRule(SecurityRule):
    rule_id = "sec.authz.csrf_disabled"
    vulnerability_class = VulnerabilityClass.MISSING_SECURITY_CONTROL
    documentation = CSRF_DOC

    def check(
        self,
        graph: SyntaxGraph,
        *,
        frameworks: Sequence[FrameworkInfo] = (),
    ) -> list[SecurityObservation]:
        del frameworks
        return _csrf_hits(self, graph)


class UnscopedLookupRule(SecurityRule):
    rule_id = "sec.authz.unscoped_lookup"
    vulnerability_class = VulnerabilityClass.IDOR
    documentation = AUTHZ_DOC

    def check(
        self,
        graph: SyntaxGraph,
        *,
        frameworks: Sequence[FrameworkInfo] = (),
    ) -> list[SecurityObservation]:
        del frameworks
        observations: list[SecurityObservation] = []
        for i, line in enumerate(graph.lines, start=1):
            if _IDOR.search(line) is None:
                continue
            if _explicit_admin(graph, i) or _lookup_constrained(graph, i):
                continue
            observations.append(
                SecurityObservation(
                    rule_id=self.rule_id,
                    vulnerability_class=self.vulnerability_class,
                    title="Potential insecure direct object reference",
                    summary="Object fetched by id without an ownership constraint on the lookup",
                    file_path=graph.file_path,
                    line=i,
                    evidence_text=line.strip()[:400],
                    confidence="low",
                    language=graph.language,
                    documentation=self.documentation,
                )
            )
        return observations


def _explicit_admin(graph: SyntaxGraph, line: int) -> bool:
    """True when a decorator marks the enclosing callable as an admin route."""
    for entity in graph.entities:
        if not (entity.start_line <= line <= entity.end_line):
            continue
        if any(_ADMIN_DECORATOR.search(decorator) for decorator in entity.decorators):
            return True
    return False


def _lookup_constrained(graph: SyntaxGraph, line: int) -> bool:
    """True when this lookup is filtered or dominated by an authorization guard.

    An ownership comparison in another statement of the same function does not
    count. Middleware and checks in another file stay a documented gap.
    """
    if line < 1 or line > len(graph.lines):
        return False
    lookup = graph.lines[line - 1]
    if _OWNER.search(lookup):
        return True
    span = _function_span(graph, line)
    if span is None:
        return False
    lookup_indent = _indent(lookup)
    for index in range(span[0], line):
        raw = graph.lines[index - 1]
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if _indent(raw) != lookup_indent or not raw.lstrip().startswith("if "):
            continue
        if _OWNER.search(raw) is None:
            continue
        body: list[str] = []
        for follow in range(index + 1, line):
            nxt = graph.lines[follow - 1]
            if nxt.strip() and _indent(nxt) <= lookup_indent:
                break
            body.append(nxt)
        if _GUARD.search("\n".join(body)):
            return True
    return False


def _function_span(graph: SyntaxGraph, line: int) -> tuple[int, int] | None:
    chosen: tuple[int, int, int] | None = None
    for scope in graph.scopes:
        span = scope.span
        if span is None or not (span.start_line <= line <= span.end_line):
            continue
        if scope.kind.value not in {"function", "method"}:
            continue
        width = span.end_line - span.start_line
        if chosen is None or width < chosen[0]:
            chosen = (width, span.start_line, span.end_line)
    if chosen is None:
        return None
    return chosen[1], chosen[2]


def _indent(text: str) -> int:
    return len(text) - len(text.lstrip(" "))


def _jwt_hits(rule: SecurityRule, graph: SyntaxGraph) -> list[SecurityObservation]:
    observations: list[SecurityObservation] = []
    for call in graph.calls:
        if not _is_jwt_decode(graph, call) or not _verification_disabled(graph, call):
            continue
        line = (
            graph.lines[call.line - 1] if 0 < call.line <= len(graph.lines) else call.argument_text
        )
        observations.append(
            _observation(rule, graph, call.line, line, "JWT decoded without signature verification")
        )
    return observations


def _is_jwt_decode(graph: SyntaxGraph, call: object) -> bool:
    qualified = str(getattr(call, "qualified", "") or "")
    name = str(getattr(call, "name", "") or "")
    if name != "decode" and not qualified.endswith(".decode"):
        return False
    root = qualified.split(".", 1)[0]
    if root in {"jwt", "jose"}:
        return True
    for item in graph.imports:
        imported = item.name or ""
        module = item.module or ""
        bound = item.alias or (imported if item.is_from_import else module.split(".", 1)[0])
        if bound != root:
            continue
        if module in {"jwt", "jose"} or imported == "jwt" or module.endswith(".jwt"):
            return True
    return False


def _verification_disabled(graph: SyntaxGraph, call: object) -> bool:
    """True when this decode call disables signature verification.

    The parser drops keyword arguments from the argument list, so the call's
    source line is part of the check. The match is the ``verify=False`` keyword
    or a ``verify_signature: False`` option, not an arbitrary substring.
    """
    texts = [
        str(getattr(argument, "text", "") or "")
        for argument in (getattr(call, "arguments", ()) or ())
    ]
    line_no = int(getattr(call, "line", 0) or 0)
    if 0 < line_no <= len(graph.lines):
        texts.append(graph.lines[line_no - 1])
    for text in texts:
        compact = "".join(text.split()).lower().rstrip(",")
        if compact in {"verify=false"}:
            return True
        if re.search(r"(?:^|[\s,(])verify\s*=\s*False\b", text):
            return True
        if re.search(r"""(?i)['"]verify_signature['"]\s*:\s*False\b""", text):
            return True
    return False


def _password_hits(rule: SecurityRule, graph: SyntaxGraph) -> list[SecurityObservation]:
    observations: list[SecurityObservation] = []
    for index, line in enumerate(graph.lines, start=1):
        stripped = line.strip()
        if stripped.startswith(("#", "//")):
            continue
        match = _PASSWORD_EQ.search(line)
        if match is None or _HASH_NAME.search(match.group(2)):
            continue
        observations.append(
            _observation(rule, graph, index, line, "Password compared with == instead of a digest")
        )
    return observations


def _csrf_hits(rule: SecurityRule, graph: SyntaxGraph) -> list[SecurityObservation]:
    observations: list[SecurityObservation] = []
    seen: set[int] = set()
    for entity in graph.entities:
        if not any(_CSRF_DECORATOR.search(decorator.strip()) for decorator in entity.decorators):
            continue
        observations.append(
            _observation(
                rule,
                graph,
                entity.start_line,
                entity.decorators[0],
                "CSRF protection disabled",
            )
        )
        seen.add(entity.start_line)
    for binding in graph.bindings:
        if binding.name not in _CSRF_NAMES or not re.search(r"\bFalse\b", binding.rhs):
            continue
        if binding.line in seen:
            continue
        line = (
            graph.lines[binding.line - 1] if 0 < binding.line <= len(graph.lines) else binding.rhs
        )
        observations.append(
            _observation(rule, graph, binding.line, line, "CSRF protection disabled")
        )
    return observations


def _observation(
    rule: SecurityRule, graph: SyntaxGraph, line: int, raw: str, summary: str
) -> SecurityObservation:
    return SecurityObservation(
        rule_id=rule.rule_id,
        vulnerability_class=rule.vulnerability_class,
        title=f"Potential {rule.vulnerability_class.value.replace('_', ' ')}",
        summary=summary,
        file_path=graph.file_path,
        line=line,
        evidence_text=raw.strip()[:400],
        confidence="medium",
        language=graph.language,
        documentation=rule.documentation,
    )
