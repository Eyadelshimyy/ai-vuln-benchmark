"""Trusted constructor profiles.

A profile is a typed, campaign-supplied configuration for one constructor argument.
It can name roles, integers, booleans, empty arrays, and a fixed set of local
fixtures. It cannot carry Solidity source. A profile binds to file hashes, so a
changed snapshot cannot reuse it. With no profile, the harness still refuses to
invent a struct.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.discovery.bounty.vfcs import (
    ATTACKER,
    SequenceIdentity,
    Vfcs,
    VfcsCall,
    sequence_id_of,
    stamp_source_identity,
)
from app.parsing.solidity_research import ResearchModel, SemanticCandidate, build_research_model

_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
_TYPE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){0,4}(?:\[\])?$")
_UINT = re.compile(r"^[0-9]{1,78}$")
_PATH = re.compile(r"^[A-Za-z0-9_./+-]{1,180}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_FIELD = re.compile(r"([A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]*\])?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;")

MAX_FIELDS = 32
MAX_DEPTH = 4
MAX_ARRAY = 24
MAX_FIXTURES = 8
MAX_BOUND_FILES = 16

# Harness addresses. The attacker is the existing probe actor and cannot be a
# privileged role. These are local labels, not live accounts.
ROLE_EXPRESSION = {
    "governor": "address(0xC0FE)",
    "pause_guardian": "address(0x6A2D)",
    "attacker": "address(0xA11CE)",
    "victim": "address(0xB0B)",
}
PRIVILEGED_ROLES = frozenset({"governor", "pause_guardian"})
_KINDS = frozenset({"uint", "bool", "role", "fixture", "struct", "array"})
_FIXTURE_KINDS = frozenset(
    {"decimals_token", "decimals_feed", "asset_list_factory", "factory_holder"}
)

COMET_CONTRACT = "CometWithExtendedAssetList"
COMET_FILE = "contracts/CometWithExtendedAssetList.sol"
COMET_CONFIGURATION = "contracts/CometConfiguration.sol"
COMET_CORE = "contracts/CometCore.sol"
COMET_PIN = "f766f51583c23acc33b2a7824654ef2029a96804"
PAUSE_SIGNATURE = "pause(bool,bool,bool,bool,bool)"
PAUSE_STATEMENT = (
    "an account that is neither the configured governor nor the configured "
    "pause guardian must not change the protocol pause flags"
)


class ProfileError(ValueError):
    """A profile is malformed or does not bind to the source it names."""

    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


@dataclass(frozen=True)
class ProfileValue:
    kind: str
    value: str = ""
    type_name: str = ""
    fields: tuple[tuple[str, ProfileValue], ...] = ()
    items: tuple[ProfileValue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "struct":
            return {
                "kind": "struct",
                "type_name": self.type_name,
                "fields": [{"name": name, "value": child.to_dict()} for name, child in self.fields],
            }
        if self.kind == "array":
            return {
                "kind": "array",
                "type_name": self.type_name,
                "items": [item.to_dict() for item in self.items],
            }
        return {"kind": self.kind, "value": self.value}


@dataclass(frozen=True)
class ProfileFixture:
    fixture_id: str
    kind: str
    decimals: int | None = None
    depends_on: str = ""

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"fixture_id": self.fixture_id, "kind": self.kind}
        if self.decimals is not None:
            body["decimals"] = self.decimals
        if self.depends_on:
            body["depends_on"] = self.depends_on
        return body


@dataclass(frozen=True)
class ConstructorProfile:
    """One approved constructor argument for one source snapshot."""

    profile_id: str
    contract: str
    source_file: str
    source_sha256: str
    parameter: str
    model_type: str
    solidity_type: str
    struct_file: str
    struct_name: str
    value: ProfileValue
    fixtures: tuple[ProfileFixture, ...]
    imports: tuple[str, ...]
    bound_files: tuple[tuple[str, str], ...]
    assumptions: tuple[str, ...]
    property_function: str
    property_statement: str
    property_getter: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "contract": self.contract,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "parameter": self.parameter,
            "model_type": self.model_type,
            "solidity_type": self.solidity_type,
            "struct_file": self.struct_file,
            "struct_name": self.struct_name,
            "value": self.value.to_dict(),
            "fixtures": [item.to_dict() for item in self.fixtures],
            "imports": list(self.imports),
            "bound_files": [{"path": path, "sha256": digest} for path, digest in self.bound_files],
            "assumptions": list(self.assumptions),
            "property_function": self.property_function,
            "property_statement": self.property_statement,
            "property_getter": self.property_getter,
        }

    def public_dict(self) -> dict[str, Any]:
        """Provenance safe to persist. Local fixtures are not a deployment."""
        return {
            "profile_id": self.profile_id,
            "validation": "accepted",
            "contract": self.contract,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "parameter": self.parameter,
            "model_type": self.model_type,
            "solidity_type": self.solidity_type,
            "bound_files": [{"path": path, "sha256": digest} for path, digest in self.bound_files],
            "fixtures": [item.to_dict() for item in self.fixtures],
            "assumptions": list(self.assumptions),
            "property_function": self.property_function,
            "property_statement": self.property_statement,
            "property_getter": self.property_getter,
            "configuration": self.value.to_dict(),
            "local_environment": True,
            "establishes_deployed_state": False,
        }


@dataclass(frozen=True)
class PreparedProfile:
    expression: str
    deploy_lines: tuple[str, ...]
    contracts: str
    import_paths: tuple[str, ...]
    provenance: dict[str, Any]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_profile(raw: Mapping[str, Any] | None) -> ConstructorProfile:
    """Validate a profile document. Rejects anything that is not a typed value."""
    if not isinstance(raw, Mapping):
        raise ProfileError("profile_malformed", "a constructor profile must be an object")
    allowed = {
        "profile_id",
        "contract",
        "source_file",
        "source_sha256",
        "parameter",
        "model_type",
        "solidity_type",
        "struct_file",
        "struct_name",
        "value",
        "fixtures",
        "imports",
        "bound_files",
        "assumptions",
        "property_function",
        "property_statement",
        "property_getter",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ProfileError("profile_malformed", f"unknown profile fields: {', '.join(unknown)}")
    profile_id = _required_text(raw, "profile_id")
    if not _PROFILE_ID.fullmatch(profile_id):
        raise ProfileError("profile_malformed", "profile_id is not a stable identifier")
    contract = _ident(raw, "contract")
    parameter = _ident(raw, "parameter")
    model_type = _type_name(_required_text(raw, "model_type"), "model_type")
    solidity_type = _type_name(_required_text(raw, "solidity_type"), "solidity_type")
    if solidity_type.endswith("[]"):
        raise ProfileError("profile_malformed", "the constructor type is not an array")
    struct_name = _ident(raw, "struct_name")
    if solidity_type.split(".")[-1] != model_type.split(".")[-1]:
        raise ProfileError("profile_type_mismatch", "solidity_type does not name the model type")
    if struct_name != model_type.split(".")[-1]:
        raise ProfileError("profile_type_mismatch", "struct_name does not match the model type")
    source_file = _path(raw, "source_file")
    struct_file = _path(raw, "struct_file")
    source_sha256 = _sha(raw, "source_sha256")
    fixtures = _fixtures(raw.get("fixtures", []))
    fixture_ids = {item.fixture_id for item in fixtures}
    value = _value(raw.get("value"), fixture_ids, depth=0)
    if value.kind != "struct":
        raise ProfileError("profile_malformed", "the profile value must be a struct")
    if value.type_name != solidity_type:
        raise ProfileError("profile_type_mismatch", "the struct type does not match solidity_type")
    _reject_privileged_attacker(value)
    imports = _paths(raw.get("imports", []), "imports")
    bound = _bound_files(raw.get("bound_files", []))
    if not any(path == source_file and digest == source_sha256 for path, digest in bound):
        raise ProfileError(
            "profile_source_mismatch", "bound_files does not include the target file hash"
        )
    assumptions = _statements(raw.get("assumptions", []))
    property_function = _required_text(raw, "property_function")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\((?:[A-Za-z0-9_,\[\]]*)\)", property_function):
        raise ProfileError("profile_malformed", "property_function is not a signature")
    property_statement = _required_text(raw, "property_statement")
    if len(property_statement) > 400 or any(ch in property_statement for ch in "{};`"):
        raise ProfileError("profile_malformed", "property_statement is not plain text")
    property_getter = _required_text(raw, "property_getter")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\(\)", property_getter):
        raise ProfileError("profile_malformed", "property_getter is not a no-argument signature")
    return ConstructorProfile(
        profile_id=profile_id,
        contract=contract,
        source_file=source_file,
        source_sha256=source_sha256,
        parameter=parameter,
        model_type=model_type,
        solidity_type=solidity_type,
        struct_file=struct_file,
        struct_name=struct_name,
        value=value,
        fixtures=fixtures,
        imports=tuple(imports),
        bound_files=bound,
        assumptions=tuple(assumptions),
        property_function=property_function,
        property_statement=property_statement,
        property_getter=property_getter,
    )


def struct_field_names(text: str, struct_name: str) -> tuple[str, ...] | None:
    """Field names of ``struct Name { ... }``, or None when that struct is absent."""
    if not _IDENT.fullmatch(struct_name):
        return None
    match = re.search(rf"\bstruct\s+{re.escape(struct_name)}\s*\{{", text)
    if match is None:
        return None
    start = match.end()
    depth = 1
    index = start
    while index < len(text) and depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    if depth:
        return None
    body = _strip_comments(text[start : index - 1])
    names = tuple(item.group(2) for item in _FIELD.finditer(body))
    if not names or len(names) > MAX_FIELDS:
        return None
    return names


def _expect_struct(value: ProfileValue, text: str) -> None:
    """Nested struct fields must be the fields declared in the bound source."""
    if value.kind == "array":
        for item in value.items:
            _expect_struct(item, text)
        return
    if value.kind != "struct":
        return
    name = value.type_name.split(".")[-1]
    declared = struct_field_names(text, name)
    expected = tuple(field_name for field_name, _child in value.fields)
    if declared is None or declared != expected:
        raise ProfileError(
            "profile_struct_mismatch",
            f"{name} fields do not match the bound source",
        )
    for _field_name, child in value.fields:
        _expect_struct(child, text)


def prepare_profile(profile: ConstructorProfile, sources: Mapping[str, str]) -> PreparedProfile:
    """Solidity fragments for a profile that still matches ``sources``."""
    for path, digest in profile.bound_files:
        text = sources.get(path)
        if text is None or sha256_text(text) != digest:
            raise ProfileError(
                "profile_source_mismatch",
                f"{path} does not match the hash this profile was bound to",
            )
    declared = struct_field_names(sources.get(profile.struct_file, ""), profile.struct_name)
    expected = tuple(name for name, _value in profile.value.fields)
    if declared is None or declared != expected:
        raise ProfileError(
            "profile_struct_mismatch",
            f"{profile.struct_name} fields do not match the bound source",
        )
    _expect_struct(profile.value, sources.get(profile.struct_file, ""))
    for path in profile.imports:
        if path not in sources:
            raise ProfileError("profile_import_missing", f"{path} is not in the source set")
    variables = _deploy_fixtures(profile.fixtures)
    expression = _render(profile.value, variables)
    contracts = _fixture_contracts(profile.fixtures)
    if len(expression) > 8_000 or len(contracts) > 8_000:
        raise ProfileError("profile_bound_exceeded", "the rendered configuration is too large")
    deploy = tuple(line for _name, _variable, line in variables)
    return PreparedProfile(
        expression=expression,
        deploy_lines=deploy,
        contracts=contracts,
        import_paths=profile.imports,
        provenance=profile.public_dict(),
    )


def comet_pause_profile(sources: Mapping[str, str]) -> ConstructorProfile:
    """The local pause-authorization profile for the pinned Comet source.

    Refuses when the authorization check or the Configuration fields are not the
    ones this profile knows how to fill. It does not rewrite the target.
    """
    target = sources.get(COMET_FILE, "")
    configuration = sources.get(COMET_CONFIGURATION, "")
    core = sources.get(COMET_CORE, "")
    if not target or not configuration or not core:
        raise ProfileError("profile_source_unsupported", "the Comet closure is incomplete")
    compact = _compact(target)
    if "msg.sender!=governor&&msg.sender!=pauseGuardian" not in compact:
        raise ProfileError(
            "profile_source_unsupported",
            "pause() does not have the expected governor and pauseGuardian check",
        )
    if "function pause(" not in target or "bool supplyPaused" not in target:
        raise ProfileError("profile_source_unsupported", "pause() is not the expected function")
    if "config.baseMinForRewards == 0" not in target:
        raise ProfileError("profile_source_unsupported", "the base-minimum check is absent")
    if "config.storeFrontPriceFactor > FACTOR_SCALE" not in target:
        raise ProfileError("profile_source_unsupported", "the storefront check is absent")
    if "PRICE_FEED_DECIMALS = 8" not in core or "MAX_BASE_DECIMALS = 18" not in core:
        raise ProfileError(
            "profile_source_unsupported", "decimal limits are not the expected constants"
        )
    if "BASE_ACCRUAL_SCALE = 1e6" not in core:
        raise ProfileError("profile_source_unsupported", "the accrual scale is not 1e6")
    declared = struct_field_names(configuration, "Configuration")
    if declared is None:
        raise ProfileError("profile_source_unsupported", "Configuration is not declared")
    values = _comet_fields()
    if tuple(values) != declared:
        raise ProfileError(
            "profile_source_unsupported",
            "Configuration fields are not the set this profile can fill",
        )
    bound = (
        (COMET_FILE, sha256_text(target)),
        (COMET_CONFIGURATION, sha256_text(configuration)),
        (COMET_CORE, sha256_text(core)),
    )
    raw = {
        "profile_id": "comet-pause-authorization",
        "contract": COMET_CONTRACT,
        "source_file": COMET_FILE,
        "source_sha256": sha256_text(target),
        "parameter": "config",
        "model_type": "Configuration",
        "solidity_type": "CometConfiguration.Configuration",
        "struct_file": COMET_CONFIGURATION,
        "struct_name": "Configuration",
        "value": {
            "kind": "struct",
            "type_name": "CometConfiguration.Configuration",
            "fields": [{"name": name, "value": values[name]} for name in declared],
        },
        "fixtures": [
            {"fixture_id": "baseToken", "kind": "decimals_token", "decimals": 6},
            {"fixture_id": "baseFeed", "kind": "decimals_feed", "decimals": 8},
            {"fixture_id": "factory", "kind": "asset_list_factory"},
            {"fixture_id": "holder", "kind": "factory_holder", "depends_on": "factory"},
        ],
        "imports": [
            "contracts/CometCore.sol",
            "contracts/interfaces/IAssetListFactory.sol",
            "contracts/interfaces/IAssetListFactoryHolder.sol",
        ],
        "bound_files": [{"path": path, "sha256": digest} for path, digest in bound],
        "assumptions": [
            "governor and pauseGuardian are local harness addresses, distinct from the attacker",
            "base token decimals are 6 so baseScale equals BASE_ACCRUAL_SCALE (1e6)",
            "the base price feed reports 8 decimals, matching PRICE_FEED_DECIMALS",
            "baseMinForRewards is 1 because the constructor rejects 0",
            "storeFrontPriceFactor is 0, which passes the constructor and is not the "
            "FACTOR_SCALE configuration of public issue 1104",
            "interest-rate fields are 0 because the constructor divides those values by SECONDS_PER_YEAR",
            "assetConfigs is empty. Length 0 is within MAX_ASSETS_FOR_ASSET_LIST and pause "
            "does not read collateral",
            "the extension delegate and asset-list factory are local fixtures, not deployed "
            "Compound contracts",
            "this configuration is a local construction environment, not live market state",
        ],
        "property_function": PAUSE_SIGNATURE,
        "property_statement": PAUSE_STATEMENT,
        "property_getter": "isSupplyPaused()",
    }
    return parse_profile(raw)


def seed_sources(
    selection: Any, sources: Mapping[str, str], profile: ConstructorProfile
) -> dict[str, str]:
    """The files a profile sequence is compiled from: the target closure, not extra context."""
    if getattr(selection, "mode", "") == "explicit":
        paths = [path for path in selection.selected if path in sources]
    else:
        paths = [
            path
            for path in selection.selected
            if selection.role_of(path) in {"target", "compile_dependency"} and path in sources
        ]
    if profile.source_file not in paths:
        raise ProfileError(
            "profile_dependency_unreadable",
            f"{profile.source_file} is not in the selected closure",
        )
    for path, _digest in profile.bound_files:
        if path not in paths and path in sources:
            paths.append(path)
        if path not in sources:
            raise ProfileError("profile_dependency_unreadable", f"{path} was not read")
    return {path: sources[path] for path in paths}


def attach_profile_seed(campaign: Any, analysis: Any) -> dict[str, Any] | None:
    """Add the profile's authorization sequence to a campaign analysis, or record why not."""
    raw = campaign.spec.constructor_profile
    if not raw:
        return None
    try:
        profile = parse_profile(raw)
        chosen = seed_sources(campaign.selection, analysis.sources, profile)
        prepare_profile(profile, chosen)
    except ProfileError as exc:
        analysis.sequence_skipped[f"constructor_profile:{raw.get('profile_id', 'unknown')}"] = (
            f"{exc.code}: {exc.reason}"
        )
        record = {"validation": "rejected", "reason_code": exc.code, "reason": exc.reason}
        campaign.cache["constructor_profile"] = record
        return record
    model = build_research_model(chosen)
    seeded = _seed_sequence(model, analysis.sequence_identity, profile, chosen)
    if isinstance(seeded, str):
        analysis.sequence_skipped[f"constructor_profile:{profile.profile_id}"] = seeded
        record = {
            "validation": "rejected",
            "reason_code": "profile_property_unsupported",
            "reason": seeded,
        }
        campaign.cache["constructor_profile"] = record
        return record
    sequence, candidate = seeded
    if sequence.sequence_id not in analysis.models:
        analysis.sequences = (*analysis.sequences, sequence)
        analysis.models[sequence.sequence_id] = model
        analysis.sequence_sources[sequence.sequence_id] = dict(chosen)
        analysis.candidates = (*analysis.candidates, candidate)
    record = profile.public_dict()
    campaign.cache["constructor_profile"] = record
    return record


def _seed_sequence(
    model: ResearchModel,
    identity: SequenceIdentity,
    profile: ConstructorProfile,
    sources: Mapping[str, str],
) -> tuple[Vfcs, SemanticCandidate] | str:
    function = next(
        (
            item
            for item in model.all_functions()
            if item.contract == profile.contract and item.signature == profile.property_function
        ),
        None,
    )
    if function is None:
        return f"profile_property_unsupported: {profile.property_function} is not in the model"
    if not _authorizes(function.body):
        return (
            "profile_property_unsupported: the function does not check governor and pauseGuardian"
        )
    getter = next(
        (
            item
            for item in model.all_functions()
            if item.contract == profile.contract and item.signature == profile.property_getter
        ),
        None,
    )
    if getter is None:
        return f"profile_property_unsupported: {profile.property_getter} is not in the model"
    arguments = tuple(
        (param.name, "true" if index == 0 else "false")
        for index, param in enumerate(function.params)
    )
    if not function.params or not all(param.type_name == "bool" for param in function.params):
        return "profile_property_unsupported: the probe parameters are not booleans"
    call = VfcsCall(
        profile.contract,
        function.signature,
        "execute",
        ATTACKER,
        arguments=arguments,
        established_by=f"constructor_profile:{profile.profile_id}",
    )
    sequence = stamp_source_identity(
        Vfcs(
            sequence_id=sequence_id_of((call,), "unauthorized-call", identity),
            template="unauthorized-call",
            origin=f"constructor_profile:{profile.profile_id}",
            calls=(call,),
            property_under_test=profile.property_statement,
            derived_from=f"profile.authorization@{profile.contract}.{function.signature}",
            identity=identity,
        ),
        sources,
    )
    candidate = SemanticCandidate(
        detector="profile.authorization",
        family="access_control",
        title="unauthorized pause",
        summary=profile.property_statement,
        file=profile.source_file,
        line=function.line,
        contract=profile.contract,
        function=function.signature,
        status="profile_seed",
        confidence="source",
    )
    return sequence, candidate


def _authorizes(body: str) -> bool:
    compact = _compact(body)
    return (
        "msg.sender!=governor&&msg.sender!=pauseGuardian" in compact
        or "msg.sender==governor||msg.sender==pauseGuardian" in compact
    )


def _comet_fields() -> dict[str, dict[str, Any]]:
    zero = {"kind": "uint", "value": "0"}
    return {
        "governor": {"kind": "role", "value": "governor"},
        "pauseGuardian": {"kind": "role", "value": "pause_guardian"},
        "baseToken": {"kind": "fixture", "value": "baseToken"},
        "baseTokenPriceFeed": {"kind": "fixture", "value": "baseFeed"},
        "extensionDelegate": {"kind": "fixture", "value": "holder"},
        "supplyKink": zero,
        "supplyPerYearInterestRateSlopeLow": zero,
        "supplyPerYearInterestRateSlopeHigh": zero,
        "supplyPerYearInterestRateBase": zero,
        "borrowKink": zero,
        "borrowPerYearInterestRateSlopeLow": zero,
        "borrowPerYearInterestRateSlopeHigh": zero,
        "borrowPerYearInterestRateBase": zero,
        "storeFrontPriceFactor": zero,
        "trackingIndexScale": {"kind": "uint", "value": "1"},
        "baseTrackingSupplySpeed": zero,
        "baseTrackingBorrowSpeed": zero,
        "baseMinForRewards": {"kind": "uint", "value": "1"},
        "baseBorrowMin": zero,
        "targetReserves": zero,
        "assetConfigs": {
            "kind": "array",
            "type_name": "CometConfiguration.AssetConfig",
            "items": [],
        },
    }


def _deploy_fixtures(
    fixtures: tuple[ProfileFixture, ...],
) -> list[tuple[str, str, str]]:
    by_id = {item.fixture_id: item for item in fixtures}
    ordered: list[ProfileFixture] = []
    pending = list(fixtures)
    while pending:
        ready = [
            item
            for item in pending
            if not item.depends_on or item.depends_on in {done.fixture_id for done in ordered}
        ]
        if not ready:
            raise ProfileError("profile_fixture_cycle", "fixture dependencies do not resolve")
        ordered.extend(ready)
        pending = [item for item in pending if item not in ready]
    rendered: list[tuple[str, str, str]] = []
    names = {item.fixture_id: f"bf_{item.fixture_id}" for item in ordered}
    for item in ordered:
        contract = _contract_name(item)
        variable = names[item.fixture_id]
        if item.kind == "factory_holder":
            dependency = names[item.depends_on]
            line = f"        {contract} {variable} = new {contract}(address({dependency}));"
        else:
            line = f"        {contract} {variable} = new {contract}();"
        rendered.append((item.fixture_id, variable, line))
        by_id[item.fixture_id] = item
    return rendered


def _fixture_contracts(fixtures: tuple[ProfileFixture, ...]) -> str:
    blocks: list[str] = []
    for item in fixtures:
        contract = _contract_name(item)
        if item.kind in {"decimals_token", "decimals_feed"}:
            decimals = item.decimals if item.decimals is not None else 0
            blocks.append(
                f"contract {contract} {{\n"
                f"    uint8 public immutable d;\n"
                f"    constructor() {{ d = {decimals}; }}\n"
                f"    function decimals() external view returns (uint8) {{ return d; }}\n"
                f"}}\n"
            )
        elif item.kind == "asset_list_factory":
            blocks.append(
                f"contract {contract} is IAssetListFactory {{\n"
                "    function createAssetList(CometCore.AssetConfig[] memory)\n"
                "        external\n"
                "        returns (address)\n"
                "    {\n"
                "        return address(this);\n"
                "    }\n"
                "}\n"
            )
        elif item.kind == "factory_holder":
            blocks.append(
                f"contract {contract} is IAssetListFactoryHolder {{\n"
                f"    address public immutable factory;\n"
                f"    constructor(address factory_) {{ factory = factory_; }}\n"
                f"    function assetListFactory() external view returns (address) {{ return factory; }}\n"
                f"}}\n"
            )
    return "\n".join(blocks)


def _contract_name(item: ProfileFixture) -> str:
    prefix = {
        "decimals_token": "BugForgeTok",
        "decimals_feed": "BugForgeFeed",
        "asset_list_factory": "BugForgeFactory",
        "factory_holder": "BugForgeHolder",
    }[item.kind]
    return f"{prefix}_{item.fixture_id}"


def _render(value: ProfileValue, variables: list[tuple[str, str, str]]) -> str:
    if value.kind == "uint":
        return value.value
    if value.kind == "bool":
        return value.value
    if value.kind == "role":
        return ROLE_EXPRESSION[value.value]
    if value.kind == "fixture":
        name = next(variable for fid, variable, _line in variables if fid == value.value)
        return f"address({name})"
    if value.kind == "array":
        if value.items:
            raise ProfileError("profile_malformed", "only an empty array is rendered inline")
        return f"new {value.type_name}[](0)"
    fields = ", ".join(f"{name}: {_render(child, variables)}" for name, child in value.fields)
    return f"{value.type_name}({{ {fields} }})"


def _value(raw: Any, fixture_ids: set[str], depth: int) -> ProfileValue:
    if depth > MAX_DEPTH or not isinstance(raw, Mapping):
        raise ProfileError("profile_malformed", "a profile value is missing or too deep")
    kind = raw.get("kind")
    if kind not in _KINDS:
        raise ProfileError("profile_malformed", "a profile value kind is not supported")
    if kind == "uint":
        text = str(raw.get("value", ""))
        if not _UINT.fullmatch(text):
            raise ProfileError("profile_malformed", "a uint value is not a decimal integer")
        return ProfileValue("uint", text)
    if kind == "bool":
        text = str(raw.get("value", ""))
        if text not in {"true", "false"}:
            raise ProfileError("profile_malformed", "a bool value is not true or false")
        return ProfileValue("bool", text)
    if kind == "role":
        text = str(raw.get("value", ""))
        if text not in ROLE_EXPRESSION:
            raise ProfileError("profile_malformed", f"unknown address role {text!r}")
        return ProfileValue("role", text)
    if kind == "fixture":
        text = str(raw.get("value", ""))
        if text not in fixture_ids:
            raise ProfileError("profile_unresolved_fixture", f"fixture {text!r} is not declared")
        return ProfileValue("fixture", text)
    if kind == "array":
        type_name = _type_name(str(raw.get("type_name", "")), "array type")
        items = raw.get("items", [])
        if not isinstance(items, list) or len(items) > MAX_ARRAY:
            raise ProfileError("profile_malformed", "an array is missing or too long")
        if items:
            raise ProfileError("profile_malformed", "only an empty array is accepted in this phase")
        return ProfileValue("array", type_name=type_name)
    type_name = _type_name(str(raw.get("type_name", "")), "struct type")
    fields_raw = raw.get("fields", [])
    if not isinstance(fields_raw, list) or not fields_raw or len(fields_raw) > MAX_FIELDS:
        raise ProfileError("profile_malformed", "a struct needs its fields")
    fields: list[tuple[str, ProfileValue]] = []
    seen: set[str] = set()
    for item in fields_raw:
        if not isinstance(item, Mapping) or not _IDENT.fullmatch(str(item.get("name", ""))):
            raise ProfileError("profile_malformed", "a struct field name is not an identifier")
        name = str(item["name"])
        if name in seen:
            raise ProfileError("profile_malformed", f"duplicate struct field {name}")
        seen.add(name)
        fields.append((name, _value(item.get("value"), fixture_ids, depth + 1)))
    return ProfileValue("struct", type_name=type_name, fields=tuple(fields))


def _fixtures(raw: Any) -> tuple[ProfileFixture, ...]:
    if not isinstance(raw, list) or len(raw) > MAX_FIXTURES:
        raise ProfileError("profile_malformed", "fixtures must be a bounded list")
    found: list[ProfileFixture] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ProfileError("profile_malformed", "a fixture is not an object")
        fixture_id = str(item.get("fixture_id", ""))
        kind = str(item.get("kind", ""))
        if not _IDENT.fullmatch(fixture_id) or kind not in _FIXTURE_KINDS:
            raise ProfileError("profile_malformed", "a fixture id or kind is not allowed")
        if fixture_id in seen:
            raise ProfileError("profile_malformed", f"duplicate fixture {fixture_id}")
        seen.add(fixture_id)
        decimals = item.get("decimals")
        if kind in {"decimals_token", "decimals_feed"}:
            if (
                not isinstance(decimals, int)
                or isinstance(decimals, bool)
                or not 0 <= decimals <= 18
            ):
                raise ProfileError("profile_malformed", "fixture decimals must be an integer 0..18")
        elif decimals is not None:
            raise ProfileError("profile_malformed", "this fixture does not take decimals")
        depends = str(item.get("depends_on", ""))
        if kind == "factory_holder":
            if not depends:
                raise ProfileError("profile_unresolved_fixture", "a factory holder needs a factory")
        elif depends:
            raise ProfileError("profile_malformed", "this fixture does not take a dependency")
        found.append(
            ProfileFixture(
                fixture_id,
                kind,
                decimals if isinstance(decimals, int) else None,
                depends,
            )
        )
    ids = {item.fixture_id for item in found}
    for item in found:
        if item.depends_on and item.depends_on not in ids:
            raise ProfileError(
                "profile_unresolved_fixture", f"{item.fixture_id} depends on a missing fixture"
            )
    return tuple(found)


def _bound_files(raw: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list) or not raw or len(raw) > MAX_BOUND_FILES:
        raise ProfileError("profile_malformed", "bound_files must list the hashed sources")
    found: list[tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ProfileError("profile_malformed", "a bound file is not an object")
        path = str(item.get("path", ""))
        digest = str(item.get("sha256", ""))
        if not _PATH.fullmatch(path) or ".." in path.split("/"):
            raise ProfileError("profile_malformed", "a bound path is not allowed")
        if not _SHA.fullmatch(digest):
            raise ProfileError("profile_malformed", "a bound hash is not sha256")
        found.append((path, digest))
    return tuple(found)


def _reject_privileged_attacker(value: ProfileValue) -> None:
    if value.kind == "role" and value.value in {"attacker", "victim"}:
        raise ProfileError(
            "profile_role_conflict", "a privileged field cannot be the attacker or the victim"
        )
    if value.kind == "struct":
        for name, child in value.fields:
            if name in {"governor", "pauseGuardian"} and (
                child.kind != "role" or child.value not in PRIVILEGED_ROLES
            ):
                raise ProfileError(
                    "profile_role_conflict",
                    f"{name} must be an explicit governor or pause_guardian role",
                )
            if name in {"governor", "pauseGuardian"}:
                continue
            _reject_privileged_attacker(child)
    if value.kind == "array":
        for item in value.items:
            _reject_privileged_attacker(item)


def _required_text(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProfileError("profile_malformed", f"{key} is required")
    return value.strip()


def _ident(raw: Mapping[str, Any], key: str) -> str:
    value = _required_text(raw, key)
    if not _IDENT.fullmatch(value):
        raise ProfileError("profile_malformed", f"{key} is not an identifier")
    return value


def _type_name(value: str, label: str) -> str:
    if not _TYPE_NAME.fullmatch(value):
        raise ProfileError("profile_malformed", f"{label} is not a type name")
    return value


def _path(raw: Mapping[str, Any], key: str) -> str:
    value = _required_text(raw, key)
    if not _PATH.fullmatch(value) or ".." in value.split("/"):
        raise ProfileError("profile_malformed", f"{key} is not a repository path")
    return value


def _sha(raw: Mapping[str, Any], key: str) -> str:
    value = _required_text(raw, key)
    if not _SHA.fullmatch(value):
        raise ProfileError("profile_malformed", f"{key} is not a sha256 hex digest")
    return value


def _paths(raw: Any, label: str) -> list[str]:
    if not isinstance(raw, list) or len(raw) > MAX_BOUND_FILES:
        raise ProfileError("profile_malformed", f"{label} must be a bounded list")
    found: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not _PATH.fullmatch(item) or ".." in item.split("/"):
            raise ProfileError("profile_malformed", f"{label} contains a path that is not allowed")
        found.append(item)
    return found


def _statements(raw: Any) -> list[str]:
    if not isinstance(raw, list) or len(raw) > 24:
        raise ProfileError("profile_malformed", "assumptions must be a bounded list")
    found: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip() or len(item) > 400:
            raise ProfileError("profile_malformed", "an assumption is not plain text")
        if any(ch in item for ch in "{};`"):
            raise ProfileError("profile_malformed", "an assumption cannot carry code")
        found.append(item.strip())
    return found


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//.*?$", " ", text, flags=re.M)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", _strip_comments(text))
