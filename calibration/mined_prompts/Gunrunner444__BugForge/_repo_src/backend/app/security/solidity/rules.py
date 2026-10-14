"""Graph-backed Solidity detectors.

Rules read syntax events produced by the Solidity parser. They do not treat a
keyword as proof, and they do not mark findings verified.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache

from app.analyzers.framework_detector import FrameworkInfo
from app.domain.security import VulnerabilityClass
from app.parsing.model import SyntaxEvent, SyntaxGraph
from app.parsing.solidity_account_abstraction import account_abstraction_present
from app.parsing.solidity_cfg import (
    operation_guarded,
    placeholder_is_guarded,
    write_reachable_after,
)
from app.parsing.solidity_cross import analyze_project
from app.parsing.solidity_defi import DefiIssue, analyze_defi
from app.parsing.solidity_flow import (
    oracle_freshness_protects,
    signature_domain_gap,
    signature_replay_gap,
)
from app.parsing.solidity_guards import initializer_is_protected, reentrancy_guard_holds
from app.parsing.solidity_ir import build_semantic_program
from app.parsing.solidity_loops import loop_grows_state, loop_has_external_call, loop_is_bounded
from app.parsing.solidity_methodology import analyze_methodology
from app.parsing.solidity_modifiers import (
    ModifierResolution,
    modifier_home_graph,
    resolve_modifier,
)
from app.parsing.solidity_proxy import analyze_proxy
from app.parsing.solidity_research import SemanticCandidate, build_research_model
from app.parsing.solidity_research_suite import FAMILIES
from app.parsing.solidity_storage import analyze_storage
from app.parsing.solidity_version import checked_arithmetic
from app.security.rules.base import RuleDocumentation, SecurityObservation, SecurityRule

_DOC = RuleDocumentation(
    detects="A Solidity-specific pattern on the syntax graph.",
    evidence="The function or statement span that triggered the rule.",
    limitations=(
        "Static structure only. Cross-contract callbacks, assembly control flow, "
        "and compiler type checking are not fully modeled."
    ),
    false_positives=(
        "A custom modifier or a check expressed in assembly can be missed, "
        "which keeps the finding at potential rather than suppressing it."
    ),
)


def solidity_security_rules() -> list[SecurityRule]:
    return [
        ReentrancyRule(),
        TxOriginRule(),
        UncheckedCallRule(),
        ArbitraryDelegatecallRule(),
        MissingAuthorizationRule(),
        DowncastRule(),
        UncheckedArithmeticRule(),
        Pre08ArithmeticRule(),
        RoundingRule(),
        SignatureReplayRule(),
        InitializerRule(),
        StorageCollisionRule(),
        UnboundedLoopRule(),
        UnboundedStateLoopRule(),
        Erc20ReturnRule(),
        CallbackReentrancyRule(),
        CrossFunctionReentrancyRule(),
        UpgradeAuthRule(),
        EncodePackedRule(),
        RandomnessRule(),
        SignatureDomainRule(),
        SelfdestructRule(),
        SignedCastRule(),
        StaleOracleRule(),
        DonationInflationRule(),
        FeeOnTransferRule(),
        AssemblySensitiveRule(),
        YulStructureRule(),
        CrossContractRule(),
        AccountingDesyncRule(),
        SiblingAuthRule(),
        BoundaryRule(),
        Erc4626InflationRule(),
        FlashSpotRule(),
        *_defi_rules(),
        *_research_rules(),
    ]


class _SolidityRule(SecurityRule):
    rule_id: str
    vulnerability_class: VulnerabilityClass
    documentation = _DOC
    event_kind = ""

    def check(
        self,
        graph: SyntaxGraph,
        *,
        frameworks: Sequence[FrameworkInfo] = (),
    ) -> list[SecurityObservation]:
        del frameworks
        if graph.language != "solidity" or graph.parser_tier.value == "profile_fallback":
            return []
        return self._check(graph)

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        return []

    def _obs(
        self,
        graph: SyntaxGraph,
        event: SyntaxEvent,
        title: str,
        summary: str,
        vulnerability_class: VulnerabilityClass | None = None,
    ) -> SecurityObservation:
        fields = _fields(event.extra)
        return SecurityObservation(
            rule_id=self.rule_id,
            vulnerability_class=vulnerability_class or self.vulnerability_class,
            title=title,
            summary=summary,
            file_path=graph.file_path,
            line=event.line,
            evidence_text=event.text[:300],
            confidence="medium",
            language="solidity",
            documentation=self.documentation,
            parser_backend=graph.parser_backend,
            parser_tier=str(graph.parser_tier),
            node_id=fields.get("function", ""),
            metadata={
                "contract": fields.get("contract", ""),
                "function": fields.get("function", ""),
                "status": "potential",
                "analysis_origin": "parser",
                "semantic_status": "unknown",
            },
        )


class ReentrancyRule(_SolidityRule):
    rule_id = "sol.reentrancy"
    vulnerability_class = VulnerabilityClass.REENTRANCY

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        return _with_semantics(graph, _reentrancy(self, graph, hooks_only=False))


class CallbackReentrancyRule(_SolidityRule):
    rule_id = "sol.callback_reentrancy"
    vulnerability_class = VulnerabilityClass.REENTRANCY

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        return _reentrancy(self, graph, hooks_only=True)


class TxOriginRule(_SolidityRule):
    rule_id = "sol.tx_origin"
    vulnerability_class = VulnerabilityClass.AUTHORIZATION

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        return [
            self._obs(graph, event, "tx.origin authorization", "Authorization uses tx.origin.")
            for event in graph.events
            if event.kind == "sol_tx_origin"
        ]


class UncheckedCallRule(_SolidityRule):
    rule_id = "sol.unchecked_call"
    vulnerability_class = VulnerabilityClass.UNSAFE_EXTERNAL_CALL

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        found: list[SecurityObservation] = []
        for event in graph.events:
            fields = _fields(event.extra)
            if event.kind == "sol_external_call" and fields.get("kind") in {
                "call",
                "delegatecall",
                "staticcall",
                "send",
            }:
                if fields.get("checked") != "true":
                    kind = fields.get("kind") or "call"
                    found.append(
                        self._obs(
                            graph,
                            event,
                            f"Unchecked {kind}",
                            f"The {kind} success value is not checked on this path.",
                        )
                    )
        return _with_semantics(graph, found)


class ArbitraryDelegatecallRule(_SolidityRule):
    rule_id = "sol.arbitrary_delegatecall"
    vulnerability_class = VulnerabilityClass.UNSAFE_EXTERNAL_CALL

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        model = analyze_proxy(graph)
        functions = {
            (_fields(event.extra).get("contract", ""), _function_name(event)): event
            for event in _functions(graph)
        }
        delegate_events = [event for event in graph.events if event.kind == "sol_delegatecall"]
        found: list[SecurityObservation] = []
        used: set[int] = set()
        for site in model.delegates:
            if site.provenance in {"self", "immutable", "constant", "unknown"}:
                continue
            if site.upgrade_controlled:
                continue
            event = next(
                (
                    item
                    for item in delegate_events
                    if id(item) not in used and site.target in _fields(item.extra).get("target", "")
                ),
                None,
            )
            if event is None:
                event = functions.get((site.contract, site.function))
            if event is None:
                continue
            used.add(id(event))
            if site.caller_controlled:
                detail = "delegatecall target is taken from a caller-controlled value."
            else:
                detail = "delegatecall target is not a fixed implementation."
            found.append(
                self._obs(
                    graph,
                    event,
                    "Arbitrary delegatecall",
                    detail + " This is potential evidence, not a confirmed call.",
                )
            )
        return _with_semantics(graph, found, delegate=True)


class MissingAuthorizationRule(_SolidityRule):
    rule_id = "sol.missing_authorization"
    vulnerability_class = VulnerabilityClass.AUTHORIZATION

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        found: list[SecurityObservation] = []
        for event in _functions(graph):
            fields = _fields(event.extra)
            name = _function_name(event).lower()
            if name in {"initialize", "reinitialize", "constructor"}:
                continue
            if fields.get("visibility") not in {"public", "external"}:
                continue
            if fields.get("mutability") in {"view", "pure"}:
                continue
            operations = _sensitive_operations(graph, event)
            if not operations:
                continue
            guarded = _authorized_text(graph, event)
            if all(_operation_is_guarded(guarded, operation) for operation in operations):
                continue
            found.append(
                self._obs(
                    graph,
                    event,
                    "Missing authorization",
                    f"{name} changes privileged state or transfers value without a check that dominates that operation.",
                )
            )
        return _with_semantics(graph, found)


class DowncastRule(_SolidityRule):
    rule_id = "sol.downcast"
    vulnerability_class = VulnerabilityClass.UNSAFE_ARITHMETIC

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        found: list[SecurityObservation] = []
        for event in graph.events:
            if event.kind != "sol_downcast":
                continue
            function = _enclosing_function(graph, event)
            if function and "type(" in function.text and ".max" in function.text:
                continue
            found.append(
                self._obs(
                    graph, event, "Unsafe downcast", "Narrowing cast has no visible range check."
                )
            )
        return found


class UncheckedArithmeticRule(_SolidityRule):
    rule_id = "sol.unchecked_arithmetic"
    vulnerability_class = VulnerabilityClass.UNSAFE_ARITHMETIC

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        return [
            self._obs(
                graph,
                event,
                "Unchecked arithmetic",
                "Arithmetic sits in an explicit unchecked block.",
            )
            for event in graph.events
            if event.kind == "sol_unchecked"
        ]


class Pre08ArithmeticRule(_SolidityRule):
    rule_id = "sol.pre_0_8_arithmetic"
    vulnerability_class = VulnerabilityClass.UNSAFE_ARITHMETIC

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        if _checked(graph) is not False:
            return []
        return [
            self._obs(
                graph,
                event,
                "Pre-0.8 arithmetic",
                "State update uses arithmetic before Solidity 0.8 checks.",
            )
            for event in graph.events
            if event.kind == "sol_state_write" and re.search(r"[+\-*/]", event.text)
        ]


class RoundingRule(_SolidityRule):
    rule_id = "sol.rounding"
    vulnerability_class = VulnerabilityClass.UNSAFE_ARITHMETIC

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        return [
            self._obs(
                graph,
                event,
                "Division before multiplication",
                "This expression divides before it multiplies.",
            )
            for event in graph.events
            if event.kind == "sol_div_mul"
        ]


class SignatureReplayRule(_SolidityRule):
    rule_id = "sol.signature_replay"
    vulnerability_class = VulnerabilityClass.SIGNATURE_FLAW

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        found: list[SecurityObservation] = []
        for event in graph.events:
            if event.kind != "sol_ecrecover":
                continue
            function = _enclosing_function(graph, event)
            if function is None:
                continue
            gap = signature_replay_gap(function.text)
            if gap is None:
                continue
            found.append(
                self._obs(
                    graph,
                    event,
                    "Signature replay risk",
                    gap + " This is potential evidence, not a confirmed replay.",
                )
            )
        return _with_semantics(graph, found)


class InitializerRule(_SolidityRule):
    rule_id = "sol.initializer"
    vulnerability_class = VulnerabilityClass.UNSAFE_PROXY

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        found: list[SecurityObservation] = []
        for event in _functions(graph):
            name = _function_name(event)
            fields = _fields(event.extra)
            if name not in {"initialize", "reinitialize"}:
                continue
            if fields.get("visibility") not in {"public", "external"}:
                continue
            if _initializer_protected(
                graph, fields.get("contract", ""), fields.get("modifiers", "")
            ):
                continue
            found.append(
                self._obs(
                    graph,
                    event,
                    "Initializer without guard",
                    "initialize is externally callable without an initializer modifier.",
                )
            )
        return found


class _MethodologyRule(_SolidityRule):
    issue_id = ""

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        found: list[SecurityObservation] = []
        for issue in analyze_methodology(graph):
            if issue.rule_id != self.issue_id:
                continue
            event = next(
                (
                    item
                    for item in graph.events
                    if item.kind == "sol_function" and _same_function(issue, item)
                ),
                None,
            )
            if event is None:
                continue
            found.append(self._obs(graph, event, self.issue_id, issue.summary))
        return found


class AccountingDesyncRule(_MethodologyRule):
    rule_id = "sol.accounting_desync"
    issue_id = "sol.accounting_desync"
    vulnerability_class = VulnerabilityClass.BUSINESS_LOGIC

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        return _with_semantics(graph, super()._check(graph))


class SiblingAuthRule(_MethodologyRule):
    rule_id = "sol.sibling_auth"
    issue_id = "sol.sibling_auth"
    vulnerability_class = VulnerabilityClass.AUTHORIZATION

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        return _with_semantics(graph, super()._check(graph))


class BoundaryRule(_MethodologyRule):
    rule_id = "sol.boundary"
    issue_id = "sol.boundary"
    vulnerability_class = VulnerabilityClass.BUSINESS_LOGIC


class Erc4626InflationRule(_MethodologyRule):
    rule_id = "sol.erc4626_inflation"
    issue_id = "sol.erc4626_inflation"
    vulnerability_class = VulnerabilityClass.BUSINESS_LOGIC


class FlashSpotRule(_MethodologyRule):
    rule_id = "sol.flash_spot"
    issue_id = "sol.flash_spot"
    vulnerability_class = VulnerabilityClass.BUSINESS_LOGIC


class StorageCollisionRule(_SolidityRule):
    rule_id = "sol.storage_collision"
    vulnerability_class = VulnerabilityClass.UNSAFE_PROXY

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        model = analyze_proxy(graph)
        found: list[SecurityObservation] = []
        mutable_impl = [
            event
            for event in graph.events
            if event.kind == "sol_state"
            and _fields(event.extra).get("name") in {"implementation", "impl"}
            and _fields(event.extra).get("mutability") == "storage"
        ]
        names = {_fields(event.extra).get("name", "") for event in mutable_impl}
        calls = [
            event
            for event in graph.events
            if event.kind == "sol_delegatecall"
            and any(
                name
                and re.search(rf"\b{re.escape(name)}\b", _fields(event.extra).get("target", ""))
                for name in names
            )
        ]
        anchor = (
            calls[0]
            if calls
            else next(
                (event for event in graph.events if event.kind == "sol_delegatecall"),
                None,
            )
        )
        comparisons = model.storage.comparisons if model.storage is not None else []
        compatible_only = bool(comparisons) and all(item.compatible is True for item in comparisons)
        if calls and not compatible_only:
            found.append(
                self._obs(
                    graph,
                    calls[0],
                    "Proxy storage collision indicator",
                    "delegatecall targets a mutable implementation variable in normal storage. "
                    "An EIP-1967 constant elsewhere does not make that variable the standard slot. "
                    "A compatible shared layout is not reported as a collision. "
                    "This is potential evidence, not a confirmed collision.",
                )
            )
        if anchor is not None and model.storage is not None:
            for note in model.storage.overlaps:
                found.append(self._obs(graph, anchor, "Proxy storage collision indicator", note))
        return _with_semantics(graph, found)


class UnboundedLoopRule(_SolidityRule):
    rule_id = "sol.unbounded_loop"
    vulnerability_class = VulnerabilityClass.DENIAL_OF_SERVICE

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        found: list[SecurityObservation] = []
        for event in graph.events:
            if event.kind != "sol_loop":
                continue
            function = _enclosing_function(graph, event)
            if loop_is_bounded(event.text, function.text if function else ""):
                continue
            if not loop_has_external_call(event.text):
                continue
            found.append(
                self._obs(
                    graph,
                    event,
                    "Unbounded external loop",
                    "Loop bound follows a dynamic length and performs an external call.",
                )
            )
        return found


class UnboundedStateLoopRule(_SolidityRule):
    rule_id = "sol.unbounded_state_loop"
    vulnerability_class = VulnerabilityClass.DENIAL_OF_SERVICE

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        found: list[SecurityObservation] = []
        for event in graph.events:
            if event.kind != "sol_loop":
                continue
            function = _enclosing_function(graph, event)
            if loop_is_bounded(event.text, function.text if function else ""):
                continue
            if loop_has_external_call(event.text) or not loop_grows_state(event.text):
                continue
            found.append(
                self._obs(
                    graph,
                    event,
                    "Unbounded state-growth loop",
                    "Loop bound follows a dynamic length and writes storage.",
                )
            )
        return found


class Erc20ReturnRule(_SolidityRule):
    rule_id = "sol.erc20_unchecked_return"
    vulnerability_class = VulnerabilityClass.UNSAFE_EXTERNAL_CALL

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        found: list[SecurityObservation] = []
        for event in graph.events:
            if event.kind != "sol_erc20":
                continue
            if _fields(event.extra).get("checked") == "true" or "require(" in event.text:
                continue
            found.append(
                self._obs(
                    graph,
                    event,
                    "Ignored ERC-20 return",
                    "ERC-20 transfer return value is ignored.",
                )
            )
        return found


class CrossFunctionReentrancyRule(_SolidityRule):
    rule_id = "sol.cross_function_reentrancy"
    vulnerability_class = VulnerabilityClass.REENTRANCY

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        found: list[SecurityObservation] = []
        for function in _functions(graph):
            if _reentrancy_locked(graph, function):
                continue
            body = _inside(graph, function)
            external = [
                event
                for event in body
                if event.kind == "sol_external_call"
                and _fields(event.extra).get("kind")
                in {"call", "delegatecall", "send", "transfer", "safeTransferFrom"}
            ]
            if not external or function.span is None:
                continue
            for call in body:
                if call.kind != "sol_direct_call" or call.span is None:
                    continue
                callee = _fields(call.extra).get("name", "")
                shared = _state_names(body) & _writes_of(graph, callee)
                if not shared or callee == _function_name(function):
                    continue
                external_span = external[0].span
                if external_span is None or call.span.start_byte <= external_span.start_byte:
                    continue
                same_path = write_reachable_after(
                    function.text, external[0].text[:80], call.text[:80]
                )
                if same_path is False:
                    continue
                found.append(
                    self._obs(
                        graph,
                        function,
                        "Cross-function reentrancy",
                        f"External call happens before {callee}, which writes state.",
                    )
                )
                break
        return found


class EncodePackedRule(_SolidityRule):
    rule_id = "sol.encode_packed"
    vulnerability_class = VulnerabilityClass.SIGNATURE_FLAW

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        found: list[SecurityObservation] = []
        for event in graph.events:
            if event.kind != "sol_encode_packed":
                continue
            fields = _fields(event.extra)
            dynamic = int(fields.get("dynamic") or "0")
            unknown = int(fields.get("unknown") or "0")
            if dynamic >= 2 or (dynamic >= 1 and unknown >= 1):
                found.append(
                    self._obs(
                        graph,
                        event,
                        "Ambiguous abi.encodePacked",
                        "Packed encoding concatenates more than one dynamic value.",
                    )
                )
        return found


class RandomnessRule(_SolidityRule):
    rule_id = "sol.insecure_randomness"
    vulnerability_class = VulnerabilityClass.WEAK_CRYPTOGRAPHY

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        return [
            self._obs(
                graph,
                event,
                "Predictable randomness",
                "Block data is used as a randomness source. A deadline comparison is not this pattern.",
            )
            for event in graph.events
            if event.kind == "sol_randomness"
        ]


class SignatureDomainRule(_SolidityRule):
    rule_id = "sol.signature_domain"
    vulnerability_class = VulnerabilityClass.SIGNATURE_FLAW

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        found: list[SecurityObservation] = []
        for event in graph.events:
            if event.kind != "sol_ecrecover":
                continue
            function = _enclosing_function(graph, event)
            if function is None:
                continue
            gap = signature_domain_gap(function.text)
            if gap is None:
                continue
            found.append(
                self._obs(
                    graph,
                    event,
                    "Signature missing domain separation",
                    gap + " This is potential evidence, not a confirmed signature flaw.",
                )
            )
        return _with_semantics(graph, found)


class SelfdestructRule(_SolidityRule):
    rule_id = "sol.selfdestruct"
    vulnerability_class = VulnerabilityClass.DENIAL_OF_SERVICE

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        floor = next(
            (item for item in graph.semantic_context if item.startswith("compiler_floor=")),
            "compiler_floor=unknown",
        )
        return [
            self._obs(
                graph,
                event,
                "selfdestruct is version-sensitive",
                f"{floor}. selfdestruct does not have the same effect on every EVM version.",
            )
            for event in graph.events
            if event.kind == "sol_selfdestruct"
        ]


class SignedCastRule(_SolidityRule):
    rule_id = "sol.signed_cast"
    vulnerability_class = VulnerabilityClass.UNSAFE_ARITHMETIC

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        found: list[SecurityObservation] = []
        for event in graph.events:
            if event.kind != "sol_sign_cast":
                continue
            function = _enclosing_function(graph, event)
            if function and "type(" in function.text and ".max" in function.text:
                continue
            found.append(
                self._obs(
                    graph,
                    event,
                    "Signed to unsigned conversion",
                    "A signed value is cast to an unsigned type without a visible bound.",
                )
            )
        return found


class StaleOracleRule(_SolidityRule):
    rule_id = "sol.stale_oracle"
    vulnerability_class = VulnerabilityClass.UNSAFE_EXTERNAL_CALL

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        found: list[SecurityObservation] = []
        for function in _functions(graph):
            calls = [event for event in _inside(graph, function) if event.kind == "sol_oracle_call"]
            if not calls:
                continue
            if oracle_freshness_protects(function.text) is True:
                continue
            found.append(
                self._obs(
                    graph,
                    calls[0],
                    "Unchecked oracle freshness",
                    "An oracle answer is used without a freshness or round check. This is a potential indicator.",
                )
            )
        return found


class DonationInflationRule(_SolidityRule):
    rule_id = "sol.donation_inflation"
    vulnerability_class = VulnerabilityClass.UNSAFE_ARITHMETIC

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        return _defi_observations(
            self,
            graph,
            "sol.donation_inflation",
            "Donation or inflation indicator",
        )


class FeeOnTransferRule(_SolidityRule):
    rule_id = "sol.fee_on_transfer"
    vulnerability_class = VulnerabilityClass.UNSAFE_EXTERNAL_CALL

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        return _defi_observations(
            self,
            graph,
            "sol.fee_on_transfer",
            "Fee-on-transfer indicator",
        )


class UpgradeAuthRule(_SolidityRule):
    rule_id = "sol.upgrade_auth"
    vulnerability_class = VulnerabilityClass.UNSAFE_PROXY

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        model = analyze_proxy(graph)
        functions = {
            (_fields(event.extra).get("contract", ""), _function_name(event)): event
            for event in _functions(graph)
        }
        found: list[SecurityObservation] = []
        for site in model.upgrades:
            if site.authorized:
                continue
            event = functions.get((site.contract, site.function))
            if event is None:
                continue
            found.append(
                self._obs(
                    graph,
                    event,
                    "Upgradeable function without auth",
                    site.summary,
                )
            )
        return found


class CrossContractRule(_SolidityRule):
    rule_id = "sol.cross_contract"
    vulnerability_class = VulnerabilityClass.REENTRANCY

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        model = analyze_project()
        if not model.contracts and not project_has_context():
            model = analyze_project({graph.file_path: graph})
        names = {event.text.strip() for event in graph.events if event.kind == "sol_contract"}
        found: list[SecurityObservation] = []
        anchor = next((event for event in graph.events if event.kind == "sol_function"), None)
        if anchor is None:
            return []
        for item in model.findings:
            if item.contract not in names:
                continue
            if item.confidence == "unknown" and item.kind == "authorization":
                title = "Unresolved cross-contract authorization"
                kind = VulnerabilityClass.AUTHORIZATION
            elif item.kind == "authorization":
                title = "Unresolved cross-contract authorization"
                kind = VulnerabilityClass.AUTHORIZATION
            elif item.kind == "reentrancy":
                title = "Cross-contract state dependency"
                kind = VulnerabilityClass.REENTRANCY
            else:
                title = "Cross-contract relationship"
                kind = VulnerabilityClass.MISSING_SECURITY_CONTROL
            found.append(
                self._obs(
                    graph,
                    anchor,
                    title,
                    item.summary + " This is potential evidence, not a verified vulnerability.",
                    kind,
                )
            )
        if model.incomplete and names:
            found.append(
                self._obs(
                    graph,
                    anchor,
                    "Incomplete cross-contract model",
                    model.limit_reason
                    or "The project model hit a bound and is incomplete, not safe.",
                    VulnerabilityClass.MISSING_SECURITY_CONTROL,
                )
            )
        return found


def project_has_context() -> bool:
    from app.parsing.solidity_cross import project_context_active

    return project_context_active()


class YulStructureRule(_SolidityRule):
    rule_id = "sol.yul"
    vulnerability_class = VulnerabilityClass.DYNAMIC_EXECUTION

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        detail = analyze_storage(graph).yul_detail
        if detail is None or not detail.hostile:
            return []
        anchor = next((event for event in graph.events if event.kind == "sol_function"), None)
        if anchor is None:
            return []
        return [
            self._obs(
                graph,
                anchor,
                item,
                item + " This is potential Yul structure, not a verified vulnerability. "
                "An incomplete model is not evidence that the assembly is safe.",
                _yul_class(item),
            )
            for item in detail.hostile
        ]


class AssemblySensitiveRule(_SolidityRule):
    rule_id = "sol.assembly_sensitive"
    vulnerability_class = VulnerabilityClass.DYNAMIC_EXECUTION

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        sensitive = {"sstore", "call", "delegatecall", "staticcall", "callcode", "selfdestruct"}
        found: list[SecurityObservation] = []
        for event in graph.events:
            if event.kind != "sol_yul":
                continue
            opcode = _fields(event.extra).get("op", event.text).strip()
            if opcode not in sensitive:
                continue
            found.append(
                self._obs(
                    graph,
                    event,
                    f"Inline assembly {opcode}",
                    f"Yul {opcode} is security-relevant. Unknown assembly around it is not treated as safe. "
                    "This is potential evidence, not a confirmed vulnerability.",
                )
            )
        return found


def _reentrancy(
    rule: _SolidityRule, graph: SyntaxGraph, *, hooks_only: bool
) -> list[SecurityObservation]:
    hooks = {"onerc721received", "tokensreceived", "onerc1155received", "onerc1155batchreceived"}
    found: list[SecurityObservation] = []
    for function in _functions(graph):
        name = _function_name(function).lower()
        if hooks_only and name not in hooks:
            continue
        if not hooks_only and name in hooks:
            continue
        if _reentrancy_locked(graph, function):
            continue
        body_events = _inside(graph, function)
        calls = [
            event
            for event in body_events
            if event.kind == "sol_external_call"
            and _fields(event.extra).get("kind")
            in {"call", "delegatecall", "staticcall", "transfer", "send", "safeTransferFrom"}
            and event.span is not None
        ]
        writes = [
            event
            for event in body_events
            if event.kind == "sol_state_write" and event.span is not None
        ]
        reads = [
            event
            for event in body_events
            if event.kind == "sol_state_read" and event.span is not None
        ]
        for call in calls:
            for write in writes:
                assert call.span is not None and write.span is not None
                if write.span.start_byte <= call.span.start_byte:
                    continue
                if not _state_depends_on_call(call, write, reads, writes):
                    continue
                same_path = write_reachable_after(function.text, call.text[:80], write.text[:80])
                if same_path is False:
                    continue
                title = "Callback reentrancy" if hooks_only else "State update after external call"
                variable = _fields(write.extra).get("name", "")
                found.append(
                    rule._obs(
                        graph,
                        function,
                        title,
                        f"{variable or 'state'} is written after control may transfer to another contract.",
                    )
                )
                break
    return _with_semantics(graph, found)


def _with_semantics(
    graph: SyntaxGraph,
    observations: list[SecurityObservation],
    *,
    delegate: bool = False,
) -> list[SecurityObservation]:
    """Attach semantic/dataflow origin. Parser fallback is labeled, not treated as compiler fact."""
    from app.parsing.solidity_dataflow import analyze_dataflow

    program = build_semantic_program(graph)
    flow = analyze_dataflow(program)
    from app.parsing.solidity_state_transitions import analyze_state_transitions

    transitions = analyze_state_transitions(program)
    origin = "semantic" if program.functions else "parser"
    for item in observations:
        item.metadata["analysis_origin"] = origin
        item.metadata["semantic_status"] = program.status
        item.metadata["transition_status"] = transitions.status
        item.metadata["verification_status"] = "not_requested"
        item.metadata["dataflow_status"] = flow.status
        if program.compiler_ir_status != "available":
            item.metadata["compiler_ir"] = program.compiler_ir_status
        if program.incomplete_reason or flow.incomplete_reason:
            item.metadata["incomplete_reason"] = program.incomplete_reason or flow.incomplete_reason
        function_name = str(item.metadata.get("function") or "")
        matches = program.functions_named(function_name) if function_name else ()
        if len(matches) == 1:
            item.metadata["semantic_authorization"] = matches[0].authorization
            item.metadata["analysis_origin"] = "semantic"
        if len(matches) == 1:
            from app.parsing.solidity_cross_dataflow import analyze_reentrancy

            item.metadata["property_status"] = analyze_reentrancy(
                program, matches[0].identity, flow
            ).status
            related = [
                path
                for path in transitions.paths
                if matches[0].identity in path.function_ids and path.status == "candidate"
            ]
            if related:
                item.metadata["candidate_path_status"] = related[0].status
                item.metadata["candidate_path_id"] = related[0].path_id
            checks = [
                check for check in transitions.checks if check.transition_id == matches[0].identity
            ]
            if checks:
                item.metadata["invariant_status"] = checks[0].status
        if delegate and len(matches) == 1:
            item.metadata["target_provenance"] = flow.target_provenance(matches[0].identity)
    return observations


def _functions(graph: SyntaxGraph) -> list[SyntaxEvent]:
    """Functions the semantic IR can name. Identity is contract, name, and line."""
    program = build_semantic_program(graph)
    if program.status != "partial":
        return [event for event in graph.events if event.kind == "sol_function"]
    declared = {(item.contract, item.name, item.line) for item in program.functions}
    found: list[SyntaxEvent] = []
    for event in graph.events:
        if event.kind != "sol_function":
            continue
        identity = (_fields(event.extra).get("contract", ""), _function_name(event), event.line)
        if identity in declared:
            found.append(event)
    return found


def _function_name(event: SyntaxEvent) -> str:
    match = re.search(r"function\s+([A-Za-z_][\w]*)", event.text)
    if match:
        return match.group(1)
    stripped = event.text.lstrip()
    if stripped.startswith("constructor"):
        return "constructor"
    if stripped.startswith("receive"):
        return "receive"
    if stripped.startswith("fallback"):
        return "fallback"
    return ""


def _same_function(issue: object, event: SyntaxEvent) -> bool:
    """Match contract, exact name, and source line. Never a name prefix."""
    name = _function_name(event)
    contract = _fields(event.extra).get("contract", "")
    issue_name = str(getattr(issue, "function", "") or "")
    issue_contract = str(getattr(issue, "contract", "") or "")
    issue_line = int(getattr(issue, "line", 0) or 0)
    if name != issue_name:
        return False
    if issue_contract and contract and issue_contract != contract:
        return False
    return not issue_line or event.line == issue_line


def _inside(graph: SyntaxGraph, function: SyntaxEvent) -> list[SyntaxEvent]:
    span = function.span
    if span is None:
        return []
    return [
        event
        for event in graph.events
        if event.span is not None
        and span.start_byte <= event.span.start_byte < span.end_byte
        and event is not function
    ]


def _enclosing_function(graph: SyntaxGraph, event: SyntaxEvent) -> SyntaxEvent | None:
    if event.span is None:
        return None
    chosen: SyntaxEvent | None = None
    for function in _functions(graph):
        span = function.span
        if span is None:
            continue
        if span.start_byte <= event.span.start_byte < span.end_byte and (
            chosen is None
            or (
                chosen.span
                and span.end_byte - span.start_byte < chosen.span.end_byte - chosen.span.start_byte
            )
        ):
            chosen = function
    return chosen


def _fields(extra: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in extra.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields


_PRIVILEGED = {
    "withdraw",
    "setowner",
    "upgradeto",
    "upgradetoandcall",
    "pause",
    "unpause",
    "mint",
    "grantrole",
    "transferownership",
}
_CRITICAL_STATE = {
    "owner",
    "admin",
    "governance",
    "implementation",
    "impl",
    "pendingowner",
    "paused",
    "guardian",
}
_CONTROL_TRANSFER = {"call", "delegatecall", "staticcall", "transfer", "send", "safeTransferFrom"}


def _sensitive_operations(graph: SyntaxGraph, function: SyntaxEvent) -> list[SyntaxEvent]:
    privileged = _function_name(function).lower() in _PRIVILEGED
    found: list[SyntaxEvent] = []
    for event in _inside(graph, function):
        fields = _fields(event.extra)
        if event.kind == "sol_state_write":
            written = fields.get("name", "").lower()
            if (
                written in _CRITICAL_STATE
                or "role" in written
                or written in {"totalsupply", "supply"}
            ):
                found.append(event)
            elif privileged:
                found.append(event)
        elif event.kind == "sol_delegatecall":
            found.append(event)
        elif event.kind == "sol_external_call" and fields.get("kind") in _CONTROL_TRANSFER:
            kind = fields.get("kind", "")
            native_value = (
                kind in {"send"}
                or (kind == "call" and "value" in event.text)
                or (kind == "transfer" and "," not in event.text)
            )
            if native_value or kind == "delegatecall" or privileged:
                found.append(event)
    return found


def _operation_is_guarded(function_text: str, operation: SyntaxEvent) -> bool:
    verdict = operation_guarded(function_text, operation.text[:120])
    return verdict is True


def _authorized_text(graph: SyntaxGraph, function: SyntaxEvent) -> str:
    text = function.text
    contract = _fields(function.extra).get("contract", "")
    injections: list[str] = []
    for name in _modifier_names(_fields(function.extra).get("modifiers", "")):
        resolution = resolve_modifier(graph, contract, name)
        if resolution.status == "resolved" and _modifier_guarded(graph, resolution, name):
            injections.append("require(msg.sender == owner);")
    if injections:
        brace = text.find("{")
        if brace >= 0:
            text = text[: brace + 1] + " ".join(injections) + text[brace + 1 :]
    own = _function_name(function)
    for name in _guard_helpers(graph):
        if name == own:
            continue
        text = re.sub(
            rf"\b{re.escape(name)}\s*\([^;]*\)\s*;",
            "require(msg.sender == owner);",
            text,
        )
    return text


def _modifier_guarded(graph: SyntaxGraph, resolution: ModifierResolution, name: str) -> bool:
    """A modifier guards its placeholder directly or through one guard helper call.

    ``modifier onlyOwner() { _validateOwnership(); _; }`` is guarded when the helper,
    found in the file that defines the modifier, is itself a sender guard. Helpers
    that cannot be found keep the modifier unguarded, so nothing is assumed.
    """
    if placeholder_is_guarded(resolution.body):
        return True
    home = modifier_home_graph(graph, resolution)
    if home is None:
        return False
    body = resolution.body
    for helper in _guard_helpers(home):
        if helper == name:
            continue
        body = re.sub(
            rf"\b{re.escape(helper)}\s*\([^;]*\)\s*;",
            "require(msg.sender == owner);",
            body,
        )
    return body != resolution.body and placeholder_is_guarded(body)


def _modifier_names(modifiers: str) -> list[str]:
    names: list[str] = []
    for item in modifiers.split(","):
        token = re.split(r"[\(\s]", item.strip(), maxsplit=1)[0]
        if token:
            names.append(token)
    return names


def _guard_helpers(graph: SyntaxGraph) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for function in _functions(graph):
        name = _function_name(function)
        if not name:
            continue
        grouped.setdefault(name, []).append(function.text)
    helpers: dict[str, str] = {}
    for name, bodies in grouped.items():
        if len(bodies) != 1:
            continue
        start = bodies[0].find("{")
        end = bodies[0].rfind("}")
        if start < 0 or end <= start:
            continue
        inner = bodies[0][start + 1 : end]
        probe = "function _h() {" + inner + " owner = next; }"
        if operation_guarded(probe, "owner = next") is True:
            helpers[name] = inner
    return helpers


def _initializer_protected(graph: SyntaxGraph, contract: str, modifiers: str) -> bool:
    helpers = _init_helpers(graph, contract)
    for name in _modifier_names(modifiers):
        if "initial" not in name.lower():
            continue
        resolution = resolve_modifier(graph, contract, name)
        if resolution.status != "resolved":
            continue
        if initializer_is_protected(resolution.body, helpers):
            return True
    return False


def _init_helpers(graph: SyntaxGraph, contract: str) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for function in _functions(graph):
        if _fields(function.extra).get("contract") != contract:
            continue
        name = _function_name(function)
        if name:
            grouped.setdefault(name, []).append(function.text)
    return {name: bodies[0] for name, bodies in grouped.items() if len(bodies) == 1}


def _reentrancy_locked(graph: SyntaxGraph, function: SyntaxEvent) -> bool:
    contract = _fields(function.extra).get("contract", "")
    for name in _modifier_names(_fields(function.extra).get("modifiers", "")):
        resolution = resolve_modifier(graph, contract, name)
        if resolution.status != "resolved":
            continue
        if reentrancy_guard_holds(resolution.body):
            return True
    return False


def _state_names(events: list[SyntaxEvent]) -> set[str]:
    names: set[str] = set()
    for event in events:
        if event.kind in {"sol_state_read", "sol_state_write"}:
            name = _fields(event.extra).get("name", "")
            if name:
                names.add(name)
    return names


def _writes_of(graph: SyntaxGraph, function_name: str) -> set[str]:
    names: set[str] = set()
    for event in graph.events:
        if event.kind != "sol_state_write":
            continue
        if _fields(event.extra).get("function") != function_name:
            continue
        name = _fields(event.extra).get("name", "")
        if name:
            names.add(name)
    return names


def _state_depends_on_call(
    call: SyntaxEvent,
    write: SyntaxEvent,
    reads: list[SyntaxEvent],
    writes: list[SyntaxEvent],
) -> bool:
    variable = _fields(write.extra).get("name", "")
    assert call.span is not None and write.span is not None
    read_before = any(
        _fields(read.extra).get("name") == variable
        and read.span is not None
        and read.span.start_byte < call.span.start_byte
        for read in reads
    )
    written_before = any(
        _fields(prior.extra).get("name") == variable
        and prior.span is not None
        and prior.span.start_byte < call.span.start_byte
        for prior in writes
    )
    used_in_call = bool(variable and variable in call.text)
    shared = bool(_shared_identifiers(call.text, write.text))
    if read_before or used_in_call or shared:
        return True
    if written_before:
        return False
    kind = _fields(call.extra).get("kind", "")
    if kind == "delegatecall":
        return True
    return bool(re.search(r"balance|share|deposit|supply|debt|owed|reserve", variable, re.I))


def _shared_identifiers(left: str, right: str) -> set[str]:
    skip = {"msg", "sender", "value", "call", "require", "bool", "memory", "this", "address"}
    words = set(re.findall(r"[A-Za-z_]\w*", left)) - skip
    other = set(re.findall(r"[A-Za-z_]\w*", right))
    return {word for word in words & other if word not in skip}


def _defi_rules() -> list[SecurityRule]:
    return [
        _DefiRule("sol.erc4626", "ERC-4626 accounting", VulnerabilityClass.BUSINESS_LOGIC),
        _DefiRule(
            "sol.rounding_direction",
            "Directional rounding",
            VulnerabilityClass.UNSAFE_ARITHMETIC,
        ),
        _DefiRule("sol.slippage", "Slippage bound", VulnerabilityClass.BUSINESS_LOGIC),
        _DefiRule(
            "sol.oracle_accounting",
            "Oracle valuation",
            VulnerabilityClass.UNSAFE_EXTERNAL_CALL,
        ),
        _DefiRule("sol.lending", "Lending accounting", VulnerabilityClass.BUSINESS_LOGIC),
        _DefiRule("sol.amm", "AMM reserve accounting", VulnerabilityClass.BUSINESS_LOGIC),
        _DefiRule("sol.approval", "Approval or permit", VulnerabilityClass.SIGNATURE_FLAW),
        _DefiRule(
            "sol.defi_reentrancy",
            "Token callback reentrancy",
            VulnerabilityClass.REENTRANCY,
        ),
    ]


class _DefiRule(_SolidityRule):
    def __init__(self, rule_id: str, title: str, vulnerability_class: VulnerabilityClass) -> None:
        self.rule_id = rule_id
        self.title = title
        self.vulnerability_class = vulnerability_class

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        return _defi_observations(self, graph, self.rule_id, self.title)


def _defi_observations(
    rule: _SolidityRule,
    graph: SyntaxGraph,
    rule_id: str,
    title: str,
) -> list[SecurityObservation]:
    model = analyze_defi(graph)
    indexed: dict[tuple[str, str], SyntaxEvent] = {}
    by_name: dict[str, list[SyntaxEvent]] = {}
    for event in _functions(graph):
        contract = _fields(event.extra).get("contract", "")
        fname = _function_name(event)
        indexed[(contract, fname)] = event
        by_name.setdefault(fname, []).append(event)
    found: list[SecurityObservation] = []
    seen: set[tuple[str, str, str]] = set()
    for issue in model.issues:
        if issue.rule_id != rule_id:
            continue
        key = (issue.contract, issue.function, issue.summary)
        if key in seen:
            continue
        seen.add(key)
        matched = indexed.get((issue.contract, issue.function))
        if matched is None and not issue.contract:
            matches = by_name.get(issue.function, [])
            matched = matches[0] if len(matches) == 1 else None
        if matched is None:
            continue
        found.append(_issue_observation(rule, graph, matched, title, issue))
    return found


def _issue_observation(
    rule: _SolidityRule,
    graph: SyntaxGraph,
    event: SyntaxEvent,
    title: str,
    issue: DefiIssue,
) -> SecurityObservation:
    observation = rule._obs(graph, event, title, issue.summary)
    observation.metadata["status"] = "potential"
    return observation


def _checked(graph: SyntaxGraph) -> bool | None:
    return checked_arithmetic(graph.semantic_context)


def _yul_class(item: str) -> VulnerabilityClass:
    text = item.lower()
    if "implementation" in text or "admin slot" in text or "beacon" in text:
        return VulnerabilityClass.UNSAFE_PROXY
    if "after external" in text:
        return VulnerabilityClass.REENTRANCY
    if any(
        token in text
        for token in (
            "ignored",
            "arbitrary call",
            "return-data",
            "dynamic call",
            "dynamic delegatecall",
        )
    ):
        return VulnerabilityClass.UNSAFE_EXTERNAL_CALL
    return VulnerabilityClass.DYNAMIC_EXECUTION


_RESEARCH_CLASS = {
    "caller_context": VulnerabilityClass.AUTHORIZATION,
    "oracle_quality": VulnerabilityClass.BUSINESS_LOGIC,
    "message_binding": VulnerabilityClass.SIGNATURE_FLAW,
    "account_abstraction": VulnerabilityClass.AUTHORIZATION,
    "balance_delta": VulnerabilityClass.BUSINESS_LOGIC,
    "arithmetic": VulnerabilityClass.UNSAFE_ARITHMETIC,
    # ``high_value`` is a bundle of three distinct detectors, so the family
    # fallback is only used if a new high-value detector is added without a
    # per-detector mapping below.
    "high_value": VulnerabilityClass.BUSINESS_LOGIC,
    # evidence-triggered protocol modules (v4 hooks, bridges): unauthenticated callers
    "protocol_modules": VulnerabilityClass.AUTHORIZATION,
}

# Some research families (notably ``high_value``) emit several detectors that
# belong to different vulnerability classes. The detector prefix (the part before
# the first dot) decides the real class so a read-only reentrancy finding is not
# mislabelled as transient-storage misuse and an EIP-7702 EOA assumption lands in
# the authorization class rather than reentrancy.
_DETECTOR_CLASS = {
    "read_only_reentrancy": VulnerabilityClass.REENTRANCY,
    "transient_storage": VulnerabilityClass.BUSINESS_LOGIC,
    "eip7702": VulnerabilityClass.AUTHORIZATION,
}


def _class_for_detector(detector: str, fallback: VulnerabilityClass) -> VulnerabilityClass:
    prefix = detector.split(".", 1)[0]
    return _DETECTOR_CLASS.get(prefix, fallback)


def _research_rules() -> list[SecurityRule]:
    return [_ResearchRule(family) for family in sorted(FAMILIES)]


@lru_cache(maxsize=8)
def _research_candidates(path: str, source: str, family: str) -> tuple[SemanticCandidate, ...]:
    model = build_research_model({path: source})
    if family == "account_abstraction" and not account_abstraction_present(model):
        return ()
    return tuple(FAMILIES[family](model))


class _ResearchRule(_SolidityRule):
    """Phase 50 semantic candidates. Each is a static candidate, never verification."""

    def __init__(self, family: str) -> None:
        self.family = family
        self.rule_id = f"sol.research.{family}"
        self.vulnerability_class = _RESEARCH_CLASS[family]

    def _check(self, graph: SyntaxGraph) -> list[SecurityObservation]:
        found: list[SecurityObservation] = []
        for item in _research_candidates(graph.file_path, graph.source, self.family):
            found.append(
                SecurityObservation(
                    rule_id=self.rule_id,
                    vulnerability_class=_class_for_detector(
                        item.detector, self.vulnerability_class
                    ),
                    title=item.title,
                    summary=item.summary,
                    file_path=graph.file_path,
                    line=item.line,
                    evidence_text="; ".join(item.observed)[:300],
                    confidence=item.confidence,
                    language="solidity",
                    documentation=_DOC,
                    parser_backend=graph.parser_backend,
                    parser_tier=str(graph.parser_tier),
                    node_id=item.function,
                    metadata={
                        "contract": item.contract,
                        "function": item.function,
                        "status": "potential",
                        "analysis_origin": "research_model",
                        "semantic_status": "candidate",
                        "detector": item.detector,
                        "missing": "; ".join(item.missing)[:300],
                    },
                )
            )
        return found
