"""Bounty campaign manifest.

The manifest describes a real bug-bounty investigation: the program, its rules,
what is in and out of scope, the source and compiler identity, and any pinned fork.
It is operator input. Unknown stays unknown, a field is never invented, and a
contract does not become in scope because it exists in a repository.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from app.discovery.orchestration.codec import CodecError, digest, from_jsonable, to_jsonable
from app.discovery.orchestration.model import CampaignIdentity

SCHEMA_VERSION = 1
UNKNOWN = "unknown"
MAX_ASSETS = 256
MAX_KNOWN_ISSUES = 256
MAX_DEPLOYMENTS = 64
MAX_IMPACT_CATEGORIES = 32
MAX_NOTES = 4000
MAX_FIELD_CHARS = 400
_SECRET_KEYS = re.compile(r"(private|secret|mnemonic|seed_phrase|api_?key|password|token)", re.I)
_SECRET_QUERY = re.compile(r"(key|token|secret|auth|sig)", re.I)
_HEX_KEY = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")


class ManifestError(ValueError):
    """The manifest is malformed, oversized, or carries something it must not."""


class ScopeStatus(StrEnum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    UNKNOWN = "unknown"


class TestingMode(StrEnum):
    LOCAL_ONLY = "local_only"
    FORK_ALLOWED = "fork_allowed"
    UNKNOWN = "unknown"


class PocRequirement(StrEnum):
    REQUIRED = "required"
    NOT_REQUIRED = "not_required"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AssetRef:
    kind: str  # contract | address | path | repository
    identifier: str
    chain_id: str = ""
    note: str = ""


@dataclass(frozen=True)
class Deployment:
    address: str
    chain_id: str
    contract_name: str = ""
    runtime_bytecode_digest: str = ""
    # Deployment topology as the program or operator declares it (Phase 51
    # hardening). Empty means unknown; nothing here is inferred from the source.
    proxy_kind: str = ""  # uups | transparent | beacon | diamond | clone | generic_proxy
    implementation: str = ""
    beacon: str = ""
    facets: tuple[str, ...] = ()
    clone_of: str = ""
    source_commit: str = ""
    compiler_version: str = ""


# Deployment fields added after Phase 50. Left out of the program-context digest
# while empty so manifests (and campaigns) from before keep their identity.
_DEPLOYMENT_EXTENSIONS = (
    "proxy_kind",
    "implementation",
    "beacon",
    "facets",
    "clone_of",
    "source_commit",
    "compiler_version",
)
_PROXY_KINDS = frozenset(
    {"", "uups", "transparent", "beacon", "diamond", "clone", "generic_proxy", "none"}
)


@dataclass(frozen=True)
class ForkReference:
    chain_id: str
    source_ref: str
    block: str
    state_snapshot: str


@dataclass(frozen=True)
class CompilerConfiguration:
    version: str = UNKNOWN
    optimizer: str = UNKNOWN  # enabled | disabled | unknown
    optimizer_runs: str = UNKNOWN
    via_ir: str = UNKNOWN  # true | false | unknown
    evm_version: str = UNKNOWN
    runtime_bytecode_digest: str = ""

    def known(self) -> bool:
        return self.version != UNKNOWN

    def fingerprint(self) -> str:
        """A stable string for identity binding. Empty when nothing is known."""
        parts = {
            "solc": self.version,
            "optimizer": self.optimizer,
            "runs": self.optimizer_runs,
            "via_ir": self.via_ir,
            "evm": self.evm_version,
        }
        if all(value == UNKNOWN for value in parts.values()):
            return ""
        return ";".join(f"{key}={value}" for key, value in parts.items())


@dataclass(frozen=True)
class ImpactCategory:
    """Program-provided policy. The core logic hard-codes no program's rewards."""

    name: str
    severity: str
    weight: int
    tags: tuple[str, ...]


@dataclass(frozen=True)
class KnownIssue:
    issue_id: str
    title: str
    source: str  # operator | audit | program_exclusion
    contracts: tuple[str, ...] = ()
    functions: tuple[str, ...] = ()
    detectors: tuple[str, ...] = ()
    root_cause_key: str = ""


@dataclass(frozen=True)
class BountyManifest:
    platform: str
    program_id: str
    program_name: str = ""
    program_url: str = ""
    rules_version: str = UNKNOWN
    in_scope: tuple[AssetRef, ...] = ()
    out_of_scope: tuple[AssetRef, ...] = ()
    known_issues: tuple[KnownIssue, ...] = ()
    poc_requirement: PocRequirement = PocRequirement.UNKNOWN
    testing_mode: TestingMode = TestingMode.UNKNOWN
    repository: str = ""
    source_commit: str = ""
    deployments: tuple[Deployment, ...] = ()
    contract_names: tuple[str, ...] = ()
    chain_ids: tuple[str, ...] = ()
    fork: ForkReference | None = None
    compiler: CompilerConfiguration = field(default_factory=CompilerConfiguration)
    impact_categories: tuple[ImpactCategory, ...] = ()
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    # ---- identity ----------------------------------------------------------------------

    def identity_digest(self) -> str:
        """Digest of every field that defines the program context.

        Deployment topology fields that are empty are left out, so a manifest that
        does not use them keeps the digest it had before they existed.
        """
        data = to_jsonable(self)
        for item in data.get("deployments", []):
            for name in _DEPLOYMENT_EXTENSIONS:
                if item.get(name) in ("", [], None):
                    item.pop(name, None)
        return f"pc_{digest(data, length=24)}"

    def to_campaign_identity(
        self,
        *,
        campaign_id: str = "",
        project: str = "",
        language: str = "solidity",
        contract: str = "",
        function: str = "",
        source_file: str = "",
        repository_root: str = "",
    ) -> CampaignIdentity:
        """Extend the existing campaign identity. No second identity model exists."""
        base = CampaignIdentity(
            campaign_id="",
            project=project or self.program_id,
            source_snapshot=self.source_commit,
            compiler_configuration=self.compiler.fingerprint(),
            repository_root=repository_root,
            language=language,
            target=contract or "",
            contract=contract,
            function=function,
            source_file=source_file,
            deployment=self._single_deployment(contract),
            fork_reference=self.fork_reference(),
            program_context=self.identity_digest(),
        )
        chosen = campaign_id or f"cp_{base.target_digest()}"
        return CampaignIdentity(**{**base.__dict__, "campaign_id": chosen})

    def fork_reference(self) -> str:
        if self.fork is None:
            return ""
        return f"{self.fork.chain_id}:{self.fork.block}:{self.fork.state_snapshot}"

    def _single_deployment(self, contract: str) -> str:
        found = [
            d
            for d in self.deployments
            if (not contract or d.contract_name == contract) and d.address
        ]
        return f"{found[0].chain_id}:{found[0].address}" if len(found) == 1 else ""

    def request_extra(self) -> dict[str, str]:
        """Fields for ``AnalysisRequest.extra``. Scope is deliberately not included."""
        extra = {
            "program_context": self.identity_digest(),
            "source_snapshot": self.source_commit,
            "compiler_configuration": self.compiler.fingerprint(),
            "poc_requirement": self.poc_requirement.value,
            "testing_mode": self.testing_mode.value,
        }
        if self.fork is not None:
            extra.update(
                {
                    "chain_id": self.fork.chain_id,
                    "fork_block": self.fork.block,
                    "state_snapshot": self.fork.state_snapshot,
                    "fork_reference": self.fork_reference(),
                }
            )
        return {key: value for key, value in extra.items() if value}

    # ---- scope -------------------------------------------------------------------------

    def scope_of(
        self,
        *,
        contract: str = "",
        file: str = "",
        address: str = "",
        chain_id: str = "",
        repository: str | None = None,
    ) -> ScopeDecision:
        """Explicit exclusion wins, an explicit inclusion is required, nothing is assumed.

        A ``repository`` asset matches only source that comes from that repository
        (``repository`` defaults to the manifest's own repository and only applies
        to a contract or file target). It never matches a bare address: a deployed
        address is bound by address assets or by a declared deployment.
        """
        if not any((contract, file, address)):
            return ScopeDecision(
                ScopeStatus.UNKNOWN, "the target names no contract, file, or address"
            )
        repo = self.repository if repository is None else repository
        source_target = bool(contract or file)
        for asset in self.out_of_scope:
            if _matches(asset, contract, file, address, chain_id, repo, source_target):
                return ScopeDecision(
                    ScopeStatus.OUT_OF_SCOPE, f"excluded by {asset.kind} {asset.identifier}", asset
                )
        if not self.in_scope:
            return ScopeDecision(ScopeStatus.UNKNOWN, "the manifest declares no in-scope assets")
        for asset in self.in_scope:
            if _matches(asset, contract, file, address, chain_id, repo, source_target):
                return ScopeDecision(
                    ScopeStatus.IN_SCOPE,
                    f"listed as in scope: {asset.kind} {asset.identifier}",
                    asset,
                )
        for deployment in self.deployments:
            if contract and deployment.contract_name == contract:
                for asset in self.in_scope:
                    if _matches(asset, "", "", deployment.address, deployment.chain_id):
                        return ScopeDecision(
                            ScopeStatus.IN_SCOPE,
                            f"deployed at in-scope address {deployment.address}",
                            asset,
                        )
        return ScopeDecision(ScopeStatus.UNKNOWN, "the target is not listed as in scope")

    def fork_allowed(self) -> bool:
        return self.testing_mode is TestingMode.FORK_ALLOWED and self.fork is not None

    def gaps(self) -> tuple[str, ...]:
        """Fields still unknown. They are reported, never filled in."""
        missing: list[str] = []
        if self.rules_version == UNKNOWN:
            missing.append("rules_version")
        if not self.in_scope:
            missing.append("in_scope")
        if self.poc_requirement is PocRequirement.UNKNOWN:
            missing.append("poc_requirement")
        if self.testing_mode is TestingMode.UNKNOWN:
            missing.append("testing_mode")
        if not self.source_commit:
            missing.append("source_commit")
        if not self.compiler.known():
            missing.append("compiler.version")
        if self.compiler.via_ir == UNKNOWN:
            missing.append("compiler.via_ir")
        if not self.impact_categories:
            missing.append("impact_categories")
        if self.fork is None:
            missing.append("fork")
        return tuple(missing)

    # ---- serialization -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = to_jsonable(self)
        assert isinstance(data, dict)
        return data

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> BountyManifest:
        if not isinstance(data, Mapping):
            raise ManifestError("a manifest must be an object")
        _reject_secrets(data)
        try:
            manifest = from_jsonable(BountyManifest, dict(data))
        except (CodecError, TypeError, KeyError) as exc:
            raise ManifestError(str(exc)) from exc
        assert isinstance(manifest, BountyManifest)
        _validate(manifest)
        return manifest


@dataclass(frozen=True)
class ScopeDecision:
    status: ScopeStatus
    reason: str
    asset: AssetRef | None = None

    @property
    def in_scope(self) -> bool:
        return self.status is ScopeStatus.IN_SCOPE


def normalize_repository(value: str) -> str:
    """Compare repositories by host/owner/name, ignoring scheme, case, and ``.git``."""
    text = value.strip().lower()
    for prefix in ("https://", "http://", "ssh://", "git://"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if text.startswith("git@"):
        text = text[4:].replace(":", "/", 1)
    text = text.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    if text.endswith(".git"):
        text = text[:-4]
    return text


def _matches(
    asset: AssetRef,
    contract: str,
    file: str,
    address: str,
    chain_id: str,
    repository: str = "",
    source_target: bool = False,
) -> bool:
    ident = asset.identifier
    if asset.kind == "repository":
        return (
            source_target
            and bool(repository)
            and normalize_repository(repository) == normalize_repository(ident)
        )
    if asset.kind == "contract":
        return bool(contract) and contract == ident
    if asset.kind == "path":
        return bool(file) and (file == ident or file.startswith(ident.rstrip("/") + "/"))
    if asset.kind == "address":
        if not address or address.lower() != ident.lower():
            return False
        return not asset.chain_id or not chain_id or asset.chain_id == chain_id
    return False


def _reject_secrets(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _SECRET_KEYS.search(str(key)) and str(key) not in {"tokens"}:
                raise ManifestError(f"field {path}{key} looks like a credential and is refused")
            _reject_secrets(item, f"{path}{key}.")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_secrets(item, f"{path}{index}.")
    elif isinstance(value, str):
        if _HEX_KEY.match(value) and path.rstrip(".").split(".")[-1] in {
            "source_ref",
            "program_url",
            "repository",
            "notes",
        }:
            raise ManifestError(f"field {path} looks like a private key and is refused")


def _validate(manifest: BountyManifest) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        raise ManifestError(f"unsupported schema version {manifest.schema_version}")
    if not manifest.platform.strip() or not manifest.program_id.strip():
        raise ManifestError("platform and program_id are required")
    limits = (
        ("in_scope", manifest.in_scope, MAX_ASSETS),
        ("out_of_scope", manifest.out_of_scope, MAX_ASSETS),
        ("known_issues", manifest.known_issues, MAX_KNOWN_ISSUES),
        ("deployments", manifest.deployments, MAX_DEPLOYMENTS),
        ("impact_categories", manifest.impact_categories, MAX_IMPACT_CATEGORIES),
    )
    for name, items, limit in limits:
        if len(items) > limit:
            raise ManifestError(f"{name} exceeds the limit of {limit}")
    if len(manifest.notes) > MAX_NOTES:
        raise ManifestError("notes exceed the size limit")
    for asset in (*manifest.in_scope, *manifest.out_of_scope):
        if asset.kind not in {"contract", "address", "path", "repository"}:
            raise ManifestError(f"unknown asset kind {asset.kind!r}")
        if not asset.identifier or len(asset.identifier) > MAX_FIELD_CHARS:
            raise ManifestError("an asset needs a bounded identifier")
    for item in manifest.impact_categories:
        if not 0 <= item.weight <= 100:
            raise ManifestError("an impact weight must be between 0 and 100")
    for deployment in manifest.deployments:
        if not re.fullmatch(r"0x[0-9a-fA-F]{40}", deployment.address):
            raise ManifestError(f"deployment address {deployment.address!r} is not an address")
        linked = [
            ("implementation", deployment.implementation),
            ("beacon", deployment.beacon),
            ("clone_of", deployment.clone_of),
            *(("facet", facet) for facet in deployment.facets),
        ]
        for label, value in linked:
            if value and not re.fullmatch(r"0x[0-9a-fA-F]{40}", value):
                raise ManifestError(f"deployment {label} {value!r} is not an address")
        if deployment.proxy_kind not in _PROXY_KINDS:
            raise ManifestError(f"unknown proxy kind {deployment.proxy_kind!r}")
        if len(deployment.facets) > 64:
            raise ManifestError("a diamond lists too many facets")
    for name in (manifest.program_url, manifest.fork.source_ref if manifest.fork else ""):
        _check_url(name)
    for text in (
        manifest.program_name,
        manifest.repository,
        manifest.source_commit,
        manifest.rules_version,
    ):
        if len(text) > MAX_FIELD_CHARS:
            raise ManifestError("a field exceeds the size limit")
    _no_dataclass_secrets(manifest)


def _check_url(value: str) -> None:
    if "://" not in value:
        return
    parts = urlsplit(value)
    if parts.username or parts.password:
        raise ManifestError("a URL must not carry credentials")
    if parts.query and _SECRET_QUERY.search(parts.query):
        raise ManifestError("a URL must not carry a key or token in its query")


def _no_dataclass_secrets(value: Any) -> None:
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _no_dataclass_secrets(getattr(value, item.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _no_dataclass_secrets(item)
    elif isinstance(value, str) and _HEX_KEY.match(value):
        # A 32-byte hex value is only accepted where a digest belongs.
        return
