"""Conservative secret, crypto, and configuration indicators."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.analyzers.framework_detector import FrameworkInfo
from app.domain.security import VulnerabilityClass
from app.parsing.model import SyntaxGraph
from app.security.language_vocab import vocab_for
from app.security.rules.base import RuleDocumentation, SecurityObservation, SecurityRule
from app.security.taint import compile_patterns, matches_any

_SECRET_ASSIGN = re.compile(
    r"(?i)(?<![A-Za-z0-9])['\"]?"
    r"(api[_-]?key|secret|password|passwd|token|private[_-]?key)"
    r"['\"]?\s*[=:]\s*['\"]([^'\"]{8,})['\"]"
)
_PLACEHOLDER_VALUE = re.compile(
    r"(?i)^(change_me|changeme|placeholder|example|todo|xxx+|your[_-]?api[_-]?key|"
    r"dummy|fake|sample|notasecret|redacted|test[-_ ]?secret|password123|"
    r"<[^>]+>|\$\{[^}]+\})[._-]*$"
)
_CONN = re.compile(
    r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:\s'\"]+:[^@\s'\"]+@"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}")
_CHECKSUM_NAME = re.compile(r"(?i)checksum|etag|digest|sha256|crc32|md5sum")
_AWS_KEY = re.compile(r"AKIA[0-9A-Z]{16}")
_PEM = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")

SECRET_DOC = RuleDocumentation(
    detects="Hard-coded credentials, cloud keys, connection strings, bearer tokens, or PEM private keys.",
    evidence="Assignment snippet with the secret value redacted.",
    limitations=(
        "Files named test_*.py or under /tests/ are skipped. "
        "A placeholder must be the whole literal, not a substring. "
        "Secrets built at runtime are not detected."
    ),
    false_positives="Whole-value placeholders and checksum or digest names.",
)
CRYPTO_DOC = RuleDocumentation(
    detects="Use of MD5/SHA1/DES/RC4/ECB which is often inappropriate for secrets.",
    evidence="Matching source line.",
    limitations="Hashing for non-security checksums is common and not a vuln.",
    false_positives="Checksums, ETags, cache keys.",
)
DEBUG_DOC = RuleDocumentation(
    detects="Debug or insecure TLS/app settings left enabled.",
    evidence="Matching configuration line.",
    limitations="Local development settings are expected.",
    false_positives="Tests that intentionally enable debug.",
)

_PEM_MATERIAL = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)


def _redact_secrets(raw: str) -> str:
    """Remove secret material from one evidence line, including same-line tails."""
    redacted = _PEM_MATERIAL.sub("-----BEGIN PRIVATE KEY----- *** -----END PRIVATE KEY-----", raw)
    redacted = _SECRET_ASSIGN.sub(lambda match: f"{match.group(1)} = '***'", redacted)
    redacted = _CONN.sub("scheme://***:***@", redacted)
    redacted = _BEARER.sub("Bearer ***", redacted)
    redacted = _AWS_KEY.sub("AKIA***", redacted)
    return redacted


def _placeholder_value(value: str) -> bool:
    return _PLACEHOLDER_VALUE.fullmatch(value.strip().strip("'\"")) is not None


def _test_path(graph: SyntaxGraph) -> bool:
    path = graph.file_path.replace("\\", "/")
    return "/tests/" in path or path.rsplit("/", 1)[-1].startswith("test_")


def _cipher_tokens(token: str) -> tuple[str, ...]:
    """Exact algorithm names, or slash-separated cipher segments.

    ``not-md5`` is one value and does not match ``md5``. ``AES/ECB/PKCS5Padding``
    yields ``ecb`` because the cipher string names that mode.
    """
    text = token.strip().lower()
    if not text:
        return ()
    if "/" in text:
        return tuple(part for part in text.split("/") if part)
    return (text,)


def _crypto_literals(call: object) -> tuple[str, ...]:
    literals: list[str] = []
    for argument in getattr(call, "arguments", ()) or ():
        text = str(getattr(argument, "text", "") or "").strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            literals.append(text[1:-1].lower())
        elif getattr(argument, "is_literal", False):
            literals.append(text.strip("'\"").lower())
    return tuple(literals)


_DEBUG_PATTERNS = (
    r"DEBUG\s*=\s*True",
    r"ssl_verify\s*=\s*False",
    r"verify\s*=\s*False",
    r"InsecureRequestWarning",
    r"NODE_TLS_REJECT_UNAUTHORIZED\s*=\s*['\"]0['\"]",
    r"SECRET_KEY\s*=\s*['\"][^'\"]{0,8}['\"]",
)


class HardcodedSecretRule(SecurityRule):
    rule_id = "sec.secrets.hardcoded"
    vulnerability_class = VulnerabilityClass.HARDCODED_SECRET
    documentation = SECRET_DOC

    def check(
        self,
        graph: SyntaxGraph,
        *,
        frameworks: Sequence[FrameworkInfo] = (),
    ) -> list[SecurityObservation]:
        del frameworks
        if _test_path(graph):
            return []
        observations: list[SecurityObservation] = []
        for binding in graph.bindings:
            if not binding.rhs_is_literal:
                continue
            line = (
                graph.lines[binding.line - 1]
                if 0 < binding.line <= len(graph.lines)
                else binding.rhs
            )
            if (
                _CHECKSUM_NAME.search(binding.name)
                and not _PEM.search(line)
                and not _AWS_KEY.search(line)
            ):
                continue
            if _PEM.search(line):
                observations.append(
                    self._obs(graph, binding.line, "PEM private key in source", line)
                )
                continue
            aws = _AWS_KEY.search(line)
            if aws:
                observations.append(self._obs(graph, binding.line, "AWS-style access key id", line))
                continue
            assigned = _SECRET_ASSIGN.search(f"{binding.name} = {binding.rhs}")
            if assigned and not _placeholder_value(assigned.group(2)):
                observations.append(
                    self._obs(graph, binding.line, f"Hard-coded {assigned.group(1)}", line)
                )
            elif _CONN.search(binding.rhs):
                observations.append(
                    self._obs(graph, binding.line, "Hard-coded connection string", line)
                )
            elif _BEARER.search(binding.rhs):
                observations.append(self._obs(graph, binding.line, "Hard-coded bearer token", line))
        for i, line in enumerate(graph.lines, start=1):
            if line.strip().startswith(("#", "//", "/*", "*", "--")):
                continue
            if any(obs.line == i for obs in observations):
                continue
            if _CHECKSUM_NAME.search(line) and not _PEM.search(line) and not _AWS_KEY.search(line):
                continue
            assigned = _SECRET_ASSIGN.search(line)
            if assigned and not _placeholder_value(assigned.group(2)):
                observations.append(self._obs(graph, i, f"Hard-coded {assigned.group(1)}", line))
                continue
            if _CONN.search(line):
                observations.append(self._obs(graph, i, "Hard-coded connection string", line))
                continue
            if _BEARER.search(line):
                observations.append(self._obs(graph, i, "Hard-coded bearer token", line))
                continue
            if _PEM.search(line) or _AWS_KEY.search(line):
                kind = (
                    "PEM private key in source" if _PEM.search(line) else "AWS-style access key id"
                )
                observations.append(self._obs(graph, i, kind, line))
        return observations

    def _obs(self, graph: SyntaxGraph, line: int, summary: str, raw: str) -> SecurityObservation:
        redacted = _redact_secrets(raw)
        return SecurityObservation(
            rule_id=self.rule_id,
            vulnerability_class=self.vulnerability_class,
            title="Potential hard-coded secret",
            summary=summary,
            file_path=graph.file_path,
            line=line,
            evidence_text=redacted.strip()[:400],
            confidence="medium",
            language=graph.language,
            documentation=self.documentation,
        )


class WeakCryptoRule(SecurityRule):
    rule_id = "sec.crypto.weak"
    vulnerability_class = VulnerabilityClass.WEAK_CRYPTOGRAPHY
    documentation = CRYPTO_DOC

    def check(
        self,
        graph: SyntaxGraph,
        *,
        frameworks: Sequence[FrameworkInfo] = (),
    ) -> list[SecurityObservation]:
        del frameworks
        vocab = vocab_for(graph.language)
        algorithms = tuple(
            token.lower()
            for token in (vocab.crypto_names if vocab is not None else ())
            if token.lower() not in {"createhash", "hash", "digest", "new", "getinstance", "cipher"}
        )
        if not algorithms:
            return []
        observations: list[SecurityObservation] = []
        crypto_callees = {"new", "getinstance", "createhash", "cipher", "pbkdf2"}
        for call in graph.calls:
            simple = call.qualified.rsplit(".", 1)[-1].lower()
            literals = _crypto_literals(call)
            direct = simple in algorithms
            configured = simple in crypto_callees and any(
                part in algorithms for token in literals for part in _cipher_tokens(token)
            )
            if not direct and not configured:
                continue
            hay = f"{call.qualified}({call.argument_text})".lower()
            observations.append(
                SecurityObservation(
                    rule_id=self.rule_id,
                    vulnerability_class=self.vulnerability_class,
                    title="Potential weak cryptography",
                    summary="Weak hash or cipher used in a security-sensitive context",
                    file_path=graph.file_path,
                    line=call.line,
                    evidence_text=hay.strip()[:400],
                    confidence="low",
                    language=graph.language,
                    documentation=self.documentation,
                    parser_backend=graph.parser_backend,
                    node_id=call.node_id,
                )
            )
        return observations


class InsecureConfigRule(SecurityRule):
    rule_id = "sec.config.insecure"
    vulnerability_class = VulnerabilityClass.INSECURE_CONFIGURATION
    documentation = DEBUG_DOC

    def check(
        self,
        graph: SyntaxGraph,
        *,
        frameworks: Sequence[FrameworkInfo] = (),
    ) -> list[SecurityObservation]:
        del frameworks
        patterns = compile_patterns(_DEBUG_PATTERNS)
        observations: list[SecurityObservation] = []
        path = graph.file_path.replace("\\", "/")
        if "/tests/" in path or path.rsplit("/", 1)[-1].startswith("test_"):
            return []
        for i, line in enumerate(graph.lines, start=1):
            if matches_any(line, patterns) is None:
                continue
            observations.append(
                SecurityObservation(
                    rule_id=self.rule_id,
                    vulnerability_class=self.vulnerability_class,
                    title="Potential insecure configuration",
                    summary="Debug or TLS verification disabled in source",
                    file_path=graph.file_path,
                    line=i,
                    evidence_text=line.strip()[:400],
                    confidence="low",
                    language=graph.language,
                    documentation=self.documentation,
                )
            )
        return observations
