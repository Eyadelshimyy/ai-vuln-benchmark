"""Discovery corpus. Every seed records provenance and never stores raw secrets."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256

from app.security_testing.secrets import redact_text


class SeedSource(StrEnum):
    INITIAL = "initial"
    STATIC_FINDING = "static_finding"
    SYMBOLIC_EXECUTION = "symbolic_execution"
    LLM = "llm"
    COVERAGE = "coverage"
    CRASH = "crash"
    INTERESTING = "interesting"
    MINIMIZED = "minimized"
    STATE_TRANSITION = "state_transition"
    ITYFUZZ = "ityfuzz"
    OPERATOR = "operator"


@dataclass(frozen=True)
class Seed:
    seed_id: str
    source: SeedSource
    reason: str
    content_sha256: str
    preview: str
    language: str = ""
    target: str = ""
    project: str = ""
    source_snapshot: str = ""
    compiler_configuration: str = ""
    engine: str = ""
    engine_version: str = ""
    campaign: str = ""

    def snapshot(self) -> dict[str, str]:
        return {
            "seed_id": self.seed_id,
            "source": self.source.value,
            "reason": self.reason,
            "content_sha256": self.content_sha256,
            "preview": self.preview,
            "language": self.language,
            "target": self.target,
            "project": self.project,
            "source_snapshot": self.source_snapshot,
            "compiler_configuration": self.compiler_configuration,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "campaign": self.campaign,
        }


@dataclass
class DiscoveryCorpus:
    seeds: list[Seed] = field(default_factory=list)

    def add(
        self,
        content: str,
        *,
        source: SeedSource,
        reason: str,
        language: str = "",
        target: str = "",
        seed_id: str = "",
        project: str = "",
        source_snapshot: str = "",
        compiler_configuration: str = "",
        engine: str = "",
        engine_version: str = "",
        campaign: str = "",
    ) -> Seed:
        digest = sha256(content.encode("utf-8")).hexdigest()
        for existing in self.seeds:
            if (
                existing.content_sha256 == digest
                and existing.source is source
                and existing.target == target
                and existing.project == project
                and existing.source_snapshot == source_snapshot
                and existing.compiler_configuration == compiler_configuration
                and existing.engine == engine
                and existing.engine_version == engine_version
                and existing.campaign == campaign
            ):
                return existing
        redacted = redact_text(content)
        ident = seed_id or f"seed_{len(self.seeds) + 1:03d}"
        seed = Seed(
            seed_id=ident,
            source=source,
            reason=redact_text(reason)[:240],
            content_sha256=digest,
            preview=redacted[:180],
            language=language,
            target=target,
            project=project,
            source_snapshot=source_snapshot,
            compiler_configuration=compiler_configuration,
            engine=engine,
            engine_version=engine_version,
            campaign=campaign,
        )
        self.seeds.append(seed)
        return seed

    def by_source(self, source: SeedSource) -> tuple[Seed, ...]:
        return tuple(seed for seed in self.seeds if seed.source is source)

    def for_binding(
        self,
        *,
        project: str,
        target: str,
        source_snapshot: str,
        compiler_configuration: str = "",
        engine: str = "",
        engine_version: str = "",
    ) -> tuple[Seed, ...]:
        """Return seeds that match the minimum safe binding.

        The binding is project, target, source snapshot, compiler configuration,
        engine, and engine version. A different campaign may still match. A
        different compiler, engine, snapshot, target, or project does not.
        """
        found = [
            seed
            for seed in self.seeds
            if seed.project == project
            and seed.target == target
            and seed.source_snapshot == source_snapshot
            and seed.compiler_configuration == compiler_configuration
            and seed.engine == engine
            and seed.engine_version == engine_version
        ]
        return tuple(
            sorted(found, key=lambda item: (item.source.value, item.seed_id, item.content_sha256))
        )

    def snapshot(self) -> dict[str, list[dict[str, str]]]:
        return {"seeds": [seed.snapshot() for seed in self.seeds]}

    @classmethod
    def from_snapshot(cls, payload: dict[str, list[dict[str, str]]] | None) -> DiscoveryCorpus:
        corpus = cls()
        for raw in () if not payload else payload.get("seeds", []):
            corpus.seeds.append(
                Seed(
                    seed_id=raw["seed_id"],
                    source=SeedSource(raw["source"]),
                    reason=raw.get("reason", ""),
                    content_sha256=raw["content_sha256"],
                    preview=raw.get("preview", ""),
                    language=raw.get("language", ""),
                    target=raw.get("target", ""),
                    project=raw.get("project", ""),
                    source_snapshot=raw.get("source_snapshot", ""),
                    compiler_configuration=raw.get("compiler_configuration", ""),
                    engine=raw.get("engine", ""),
                    engine_version=raw.get("engine_version", ""),
                    campaign=raw.get("campaign", ""),
                )
            )
        return corpus
