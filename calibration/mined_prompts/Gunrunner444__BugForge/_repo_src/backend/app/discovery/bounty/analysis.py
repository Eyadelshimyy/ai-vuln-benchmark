"""Campaign analysis: the real bounty analysis path behind findings, VFCS, and reports.

Everything here derives from the campaign's *selected* sources (truncation-aware
:func:`select_sources`), never from the first files in alphabetical order:

* the primary pass reads the selected files;
* dropped high-priority files get a bounded follow-up pass (at most
  ``MAX_FOLLOWUP_BATCHES`` batches); what still was not read is listed, never
  implied safe;
* a contract name declared in several files is not bound to the first file:
  each additional declaring file is analysed on its own and its candidates stay
  bound to that file and flagged as name-ambiguous;
* the campaign's deployment identity (chain, address, contract, source commit,
  runtime digest, compiler, proxy/implementation/beacon/diamond/clone) is
  carried into findings, VFCS sequence identity, dedup keys, and the report.

Nothing here executes a transaction, compiles, or touches a network.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.discovery.bounty.campaign import BountyManifest
from app.discovery.bounty.deployment_identity import (
    DeploymentIdentity,
    deployment_identity,
)
from app.discovery.bounty.priority import prioritize
from app.discovery.bounty.source_selection import FileRead, SourceSelection, read_source_files
from app.discovery.bounty.vfcs import MAX_VFCS_CANDIDATES, SequenceIdentity, Vfcs, generate
from app.discovery.orchestration.model import CampaignIdentity
from app.parsing.solidity_research import (
    ResearchModel,
    SemanticCandidate,
    build_research_model,
)
from app.parsing.solidity_research_suite import run_suite

MAX_AMBIGUOUS_PASSES = 8
_DECLARATION = re.compile(r"\b(?:abstract\s+)?(?:contract|library|interface)\s+([A-Za-z_]\w*)")
MAX_TOTAL_SEQUENCES = 2 * MAX_VFCS_CANDIDATES


@dataclass
class CampaignAnalysis:
    sources: dict[str, str]
    model: ResearchModel | None
    candidates: tuple[SemanticCandidate, ...]
    sequences: tuple[Vfcs, ...]
    models: dict[str, ResearchModel]  # sequence_id -> the model it was built from
    deployment: DeploymentIdentity
    sequence_identity: SequenceIdentity
    ambiguous_contracts: frozenset[str]
    followup: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    _deployments: dict[str, DeploymentIdentity] = field(default_factory=dict)
    # sequence_id -> the exact source set its model was built from (never a generic
    # first file); and path -> sha256 of every file read at analysis time.
    sequence_sources: dict[str, dict[str, str]] = field(default_factory=dict)
    source_hashes: dict[str, str] = field(default_factory=dict)
    # "detector@Contract.signature" -> why no sequence template applied (coverage gaps)
    sequence_skipped: dict[str, str] = field(default_factory=dict)
    # Every file the analysis tried to read. Skipped files are not in ``sources``.
    file_reads: tuple[FileRead, ...] = ()

    def all_models(self) -> tuple[ResearchModel, ...]:
        """The main model first, then each distinct follow-up model (stable order)."""
        found: list[ResearchModel] = [self.model] if self.model is not None else []
        for sid in sorted(self.models):
            item = self.models[sid]
            if all(item is not other for other in found):
                found.append(item)
        return tuple(found)

    def sources_for(self, sequence_id: str) -> dict[str, str]:
        return self.sequence_sources.get(sequence_id, {})

    def candidate_for(self, derived_from: str) -> SemanticCandidate | None:
        for item in self.candidates:
            if f"{item.detector}@{item.contract}.{item.function}" == derived_from:
                return item
        return None

    def deployment_for(self, contract: str) -> DeploymentIdentity | None:
        return self._deployments.get(contract)


def read_sources(repo_root: Path, files: tuple[str, ...]) -> dict[str, str]:
    """Successfully read Solidity sources. Unreadable paths are omitted, not blanked."""
    sources, _reads = read_source_files(repo_root, files)
    return sources


def _merge_reads(existing: list[FileRead], extra: tuple[FileRead, ...]) -> list[FileRead]:
    by_path = {item.path: item for item in existing}
    for item in extra:
        by_path[item.path] = item
    return [by_path[path] for path in sorted(by_path)]


def _ordered(
    model: ResearchModel, manifest: BountyManifest, candidates: tuple[SemanticCandidate, ...]
) -> list[SemanticCandidate]:
    ranked = prioritize(model, manifest=manifest, candidates=candidates)
    order = {entry.identity: index for index, entry in enumerate(ranked.ranked)}
    return sorted(
        candidates,
        key=lambda c: (order.get(f"{c.contract}.{c.function}", len(order)), c.detector, c.line),
    )


def analyze_campaign(
    *,
    manifest: BountyManifest,
    identity: CampaignIdentity,
    repo_root: Path,
    files: tuple[str, ...],
    selection: SourceSelection,
    contract: str = "",
    address: str = "",
    chain_id: str = "",
) -> CampaignAnalysis:
    sources, primary_reads = read_source_files(repo_root, files)
    read_log: list[FileRead] = list(primary_reads)
    model = build_research_model(sources) if sources else None
    target = deployment_identity(
        manifest, contract=contract, address=address, chain_id=chain_id, model=model
    )
    seq_identity = SequenceIdentity(
        campaign_id=identity.campaign_id,
        source_snapshot=manifest.source_commit,
        compiler_configuration=manifest.compiler.fingerprint(),
        fork_reference=manifest.fork_reference(),
        program_context=manifest.identity_digest(),
        deployment=target.key if target.known else "",
        implementation=target.implementation,
    )
    candidates: list[SemanticCandidate] = []
    sequences: list[Vfcs] = []
    models: dict[str, ResearchModel] = {}
    sequence_sources: dict[str, dict[str, str]] = {}
    sequence_skipped: dict[str, str] = {}
    hashes: dict[str, str] = {path: _sha256(text) for path, text in sources.items()}
    ambiguous: set[str] = set()
    followup: dict[str, Any] = {
        "batches": [],
        "ambiguous_passes": [],
        "never_read": [],
    }

    def absorb(
        part: ResearchModel, found: tuple[SemanticCandidate, ...], part_sources: dict[str, str]
    ) -> None:
        candidates.extend(found)
        for path, text in part_sources.items():
            hashes.setdefault(path, _sha256(text))
        room = MAX_TOTAL_SEQUENCES - len(sequences)
        if room <= 0:
            return
        built = generate(
            part,
            _ordered(part, manifest, found),
            seq_identity,
            limit=min(room, MAX_VFCS_CANDIDATES),
            sources=part_sources,
        )
        for key, reason in built.skipped:
            sequence_skipped.setdefault(key, reason)
        for item in built.sequences:
            if item.sequence_id not in models:
                sequences.append(item)
                models[item.sequence_id] = part
                sequence_sources[item.sequence_id] = part_sources

    if model is not None:
        absorb(model, run_suite(model).candidates, sources)
        ambiguous |= set(model.ambiguous)

    # Bounded follow-up over dropped high-priority files.
    batches = selection.followup_batches()
    covered = set(sources)
    for batch in batches:
        batch_sources, batch_reads = read_source_files(
            repo_root, tuple(p for p in batch if p not in covered)
        )
        read_log = _merge_reads(read_log, batch_reads)
        if not batch_sources:
            continue
        covered |= set(batch_sources)
        part = build_research_model(batch_sources)
        found = tuple(c for c in run_suite(part).candidates if c.file in batch_sources)
        absorb(part, found, batch_sources)
        ambiguous |= set(part.ambiguous)
        followup["batches"].append({"files": sorted(batch_sources), "candidates": len(found)})
    followed = {path for batch in batches for path in batch}
    followup["never_read"] = [p for p in selection.dropped_high_priority if p not in followed]

    # A contract name declared in several files: analyse each other declaring file
    # alone so its candidates are bound to that file instead of being dropped.
    declared: dict[str, list[str]] = {}
    for name, declaring in selection.ambiguous_contracts:
        declared[name] = list(declaring)
        ambiguous.add(name)
    for path, text in sources.items():
        for name in _DECLARATION.findall(text):
            if name in ambiguous and path not in declared.setdefault(name, []):
                declared[name].append(path)
    for name in sorted(ambiguous):
        extra_files = [
            path
            for path in declared.get(name, [])
            if model is None
            or model.contracts.get(name) is None
            or model.contracts[name].file != path
        ]
        for path in extra_files:
            if len(followup["ambiguous_passes"]) >= MAX_AMBIGUOUS_PASSES:
                break
            alone, alone_reads = read_source_files(repo_root, (path,))
            read_log = _merge_reads(read_log, alone_reads)
            if not alone:
                continue
            part = build_research_model(alone)
            found = tuple(c for c in run_suite(part).candidates if c.contract == name)
            absorb(part, found, alone)
            followup["ambiguous_passes"].append(
                {"contract": name, "file": path, "candidates": len(found)}
            )

    unique: dict[tuple[str, str, str, str, int], SemanticCandidate] = {}
    for item in candidates:
        unique.setdefault((item.detector, item.file, item.contract, item.function, item.line), item)

    deployments: dict[str, DeploymentIdentity] = {}
    for item in unique.values():
        if item.contract in deployments:
            continue
        if contract and item.contract == contract:
            deployments[item.contract] = target
        else:
            deployments[item.contract] = deployment_identity(
                manifest, contract=item.contract, model=model
            )

    return CampaignAnalysis(
        sources=sources,
        model=model,
        candidates=tuple(
            sorted(unique.values(), key=lambda c: (c.file, c.line, c.contract, c.function))
        ),
        sequences=tuple(sequences),
        models=models,
        deployment=target,
        sequence_identity=seq_identity,
        ambiguous_contracts=frozenset(ambiguous),
        followup=followup,
        truncated=bool(selection.truncated or (model is not None and model.truncated)),
        _deployments=deployments,
        sequence_sources=sequence_sources,
        sequence_skipped=sequence_skipped,
        source_hashes=hashes,
        file_reads=tuple(read_log),
    )


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sources_digest(sources: Mapping[str, str]) -> str:
    from app.discovery.orchestration.codec import digest

    return digest(sorted((path, len(text)) for path, text in sources.items()))


__all__ = ["CampaignAnalysis", "analyze_campaign", "read_sources", "sources_digest"]
