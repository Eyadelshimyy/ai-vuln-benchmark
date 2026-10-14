"""Flow-sensitive, scope-aware taint tracking over a SyntaxGraph.

Precision (documented, not advertised beyond this):

* **flow-sensitive at the use site**: a sink uses the definition reaching that
  call's source position, not the file-final state of the symbol.
  ``q = "safe"; sink(q); q = input`` is not tainted at the sink.
* **not path-sensitive**: assignments inside branches/loops merge conservatively.
  If either path may taint ``q``, later uses in the parent stay tainted.
* **not flow-insensitive**: taint is not a sticky property of ``scope_id::name``.
  Historical taint is distinct from the taint reaching a given program point.
* **bounded field paths**: constant attributes and indexes (``obj.payload``,
  ``obj["payload"]``, ``items[0]``) are their own symbols. Dynamic keys stay
  unresolved. Depth, binding, and alias limits are explicit and zero-safe.

Symbols are ``scope_id::name``. Sibling functions never share locals.
Source/sink matching uses syntax-derived callees, member accesses, and
identifiers — never string-literal or comment text.

Same-file inter-procedural propagation runs only when the callee can be
resolved uniquely. Ambiguous names do not speculate. Optional
``ExternalCallee`` entries describe uniquely resolved callees in other files;
they are applied at the call that binds their result, then later sinks use the
definition reaching that program point.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from app.analyzers.framework_detector import FrameworkInfo
from app.core.config import settings
from app.parsing.model import Binding, CallArgument, CallSite, ScopeKind, SymbolKind, SyntaxGraph
from app.security.definitions import (
    SINK_SANITIZER_KINDS,
    LanguageSecurityVocab,
    SanitizerDefinition,
    SinkDefinition,
    SourceDefinition,
)
from app.security.language_vocab import vocab_for

TAINT_PRECISION = "flow-sensitive"
TAINT_PATH_SENSITIVE = False
MAX_TAINT_ROUNDS = 8
MAX_INTERPROC_DEPTH = 4
MAX_DEFINITIONS = 10_000
_PRESERVING_CALLEES = frozenset({"format", "str", "sprintf", "Sprintf", "c_str"})

_IDENT = re.compile(r"[A-Za-z_$/][\w$]*")
_SQL_PLACEHOLDER = re.compile(r"(?:\?|\$\d+|%s|:\w+)")


def compile_patterns(patterns: Sequence[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


def matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> re.Pattern[str] | None:
    for pattern in patterns:
        if pattern.search(text):
            return pattern
    return None


def _canon_qual(text: str) -> str:
    """Treat ``::`` / ``->`` / ``.`` as equivalent for cross-language matching."""
    return text.replace("::", ".").replace("->", ".")


def text_matches_any(text: str, patterns: Sequence[str]) -> str | None:
    """Qualified/identifier match. Does not search arbitrary haystacks."""
    return fragment_matches_source(text, patterns)


def fragment_matches_source(fragment: str, patterns: Sequence[str]) -> str | None:
    if not fragment:
        return None
    canon = _canon_qual(fragment)
    for pattern in patterns:
        p = _canon_qual(pattern)
        if not p:
            continue
        if canon == p:
            return pattern
        if canon.startswith(p + ".") or canon.startswith(p + "("):
            return pattern
        if canon.endswith("." + p) and "." in p:
            return pattern
        # Explicit regex sources (e.g. shell ``$1``) keep anchors; dotted
        # API patterns are never treated as regex because ``.`` is a member.
        if pattern.startswith("^") or pattern.endswith("$"):
            try:
                if re.search(pattern, fragment):
                    return pattern
            except re.error:
                continue
    return None


def vocab_sources(
    vocab: LanguageSecurityVocab, frameworks: Sequence[FrameworkInfo]
) -> tuple[SourceDefinition, ...]:
    extra: list[SourceDefinition] = []
    names = {fw.name.lower() for fw in frameworks}
    for key, sources in vocab.extra_sources_by_framework.items():
        if key.lower() in names:
            extra.extend(sources)
    return vocab.sources + tuple(extra)


@dataclass
class _Reach:
    reason: str | None
    def_index: int = 0
    merged: bool = False


@dataclass(frozen=True)
class _Def:
    """One reaching definition of a symbol, ordered by source position."""

    start_byte: int
    line: int
    definition_index: int
    reason: str | None
    merged: bool = False


@dataclass
class TaintState:
    """Flow-sensitive taint: ordered definitions per symbol, queried at a use."""

    defs: dict[str, tuple[_Def, ...]]
    source_pats: tuple[str, ...] = ()
    incomplete: bool = False
    limit_reason: str = ""

    def final_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for sid, defs in self.defs.items():
            if defs and defs[-1].reason:
                out[sid] = defs[-1].reason
        return out

    def reason_at_name(self, name: str, scope_id: str, at_byte: int) -> str | None:
        """Taint of ``name`` reaching ``at_byte`` in ``scope_id``. Never sibling scopes."""
        current: str | None = scope_id
        while current:
            candidate = f"{current}::{name}"
            if candidate in self.defs:
                return _reason_at(self.defs[candidate], at_byte)
            if "/" not in current:
                break
            current = current.rsplit("/", 1)[0]
        module = f"module::{name}"
        if module in self.defs:
            return _reason_at(self.defs[module], at_byte)
        return None

    @staticmethod
    def from_final_map(tainted: dict[str, str], source_pats: Sequence[str] = ()) -> TaintState:
        """Treat a file-final map as definitions at byte 0 (legacy / tests)."""
        defs: dict[str, tuple[_Def, ...]] = {
            sid: (_Def(start_byte=0, line=1, definition_index=0, reason=reason),)
            for sid, reason in tainted.items()
        }
        return TaintState(defs=defs, source_pats=tuple(source_pats))


@dataclass(frozen=True)
class ExternalCallee:
    """A callee in another file, resolved uniquely from an import.

    ``return_effects`` are summary tokens (``source:…``, ``param:N``,
    ``sanitized:kind:…``). ``param_sinks`` records parameters that reach a sink
    inside that callee: ``(index, vulnerability_class, sink, line)``.
    """

    local_name: str
    callee_file: str
    callee_function: str
    return_effects: tuple[str, ...] = ()
    param_sinks: tuple[tuple[int, str, str, int], ...] = ()
    partial: bool = False
    hops: int = 0
    semantic_id: str = ""
    relationship: str = "import"


def reaching_sanitizer_kind(taint: str) -> str | None:
    """Outermost sanitizer kind carried by a use-site reason, if that is all it is."""
    marker = ":sanitized:"
    if marker not in taint or "|" in taint:
        return None
    kind = taint.split(marker, 1)[1].split(":", 1)[0]
    if not kind or kind in {"source", "param", "from", "cross_file", "callarg"}:
        return None
    return kind


def _use_byte(span: object | None, line: int) -> int:
    start = getattr(span, "start_byte", None)
    if isinstance(start, int):
        return start
    return max(0, line) * 10_000


def receiver_argument_offset(
    parameters: Sequence[str],
    decorators: Sequence[str],
    *,
    instance: bool,
) -> int:
    """Map a call argument index onto a callee parameter index.

    Instance calls skip ``self`` or ``cls``. An explicit ``Class.method`` call
    does not, except for a classmethod, whose first parameter is implicit.
    ``@staticmethod`` never shifts. This is the same rule for same-file calls
    and cross-file summaries.
    """
    if not parameters:
        return 0
    text = " ".join(decorators)
    if "staticmethod" in text:
        return 0
    first = parameters[0]
    if "classmethod" in text and first in {"cls", "self"}:
        return 1
    if first in {"self", "cls"}:
        return 1 if instance else 0
    return 0


_PYTHON_BUILTIN_SINKS = frozenset({"eval", "exec", "compile", "__import__"})


def builtin_call_shadowed(graph: SyntaxGraph, call: CallSite) -> bool:
    """True when a bare builtin name is bound to a local definition at this call.

    Python function and method scopes treat a name assigned or defined anywhere
    in that scope as local for the whole function. Module scope stays sequential:
    a later ``def eval`` does not hide an earlier module-level builtin call.
    Nested definitions hide the name only in their enclosing function. Qualified
    calls such as ``obj.eval`` are left to the normal sink rule. Imported library
    APIs such as ``Markup`` are not builtins and still match their sink rules.
    """
    qualified = call.qualified or call.name
    if not call.name or qualified != call.name or "." in qualified:
        return False
    if graph.language == "python" and call.name not in _PYTHON_BUILTIN_SINKS:
        return False
    at = _use_byte(call.span, call.line)
    current = call.scope_id or "module"
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        if _scope_defines_name(
            graph,
            current,
            call.name,
            at,
            whole_scope=_python_function_scope(graph, current),
        ):
            return True
        if "/" not in current:
            break
        current = current.rsplit("/", 1)[0]
    if "module" not in seen and _scope_defines_name(graph, "module", call.name, at):
        return True
    return False


def calls_contained_in(
    calls: Sequence[CallSite],
    *,
    scope_id: str,
    line: int,
    span: object | None,
    names: Collection[str] | None = None,
    outermost: bool = False,
) -> list[CallSite]:
    """Calls that belong to a binding or return, by source span.

    Line equality is used only when one side has no span. Nested calls stay in
    the result unless ``outermost`` drops those strictly inside another match.
    """
    host_start = getattr(span, "start_byte", None)
    host_end = getattr(span, "end_byte", None)
    chosen: list[CallSite] = []
    for call in calls:
        if call.scope_id != scope_id:
            continue
        if names is not None and not _call_name_selected(call, names):
            continue
        call_start = getattr(call.span, "start_byte", None)
        call_end = getattr(call.span, "end_byte", None)
        if (
            isinstance(host_start, int)
            and isinstance(host_end, int)
            and isinstance(call_start, int)
            and isinstance(call_end, int)
        ):
            if host_start <= call_start and call_end <= host_end:
                chosen.append(call)
            continue
        if call.line == line:
            chosen.append(call)
    if not outermost:
        return chosen
    return [
        call
        for call in chosen
        if not any(_call_strictly_contains(other, call) for other in chosen if other is not call)
    ]


def _call_name_selected(call: CallSite, names: Collection[str]) -> bool:
    if call.name in names or call.qualified in names:
        return True
    simple = (call.qualified or call.name).rsplit(".", 1)[-1]
    return simple in names


def _call_strictly_contains(outer: CallSite, inner: CallSite) -> bool:
    if outer.span is None or inner.span is None:
        return False
    inside = (
        outer.span.start_byte <= inner.span.start_byte
        and inner.span.end_byte <= outer.span.end_byte
    )
    if not inside:
        return False
    return (
        outer.span.start_byte < inner.span.start_byte or outer.span.end_byte > inner.span.end_byte
    )


def _python_function_scope(graph: SyntaxGraph, scope_id: str) -> bool:
    if graph.language != "python":
        return False
    for scope in graph.scopes:
        if scope.scope_id == scope_id:
            return scope.kind in {ScopeKind.FUNCTION, ScopeKind.METHOD}
    return False


def _scope_defines_name(
    graph: SyntaxGraph,
    scope_id: str,
    name: str,
    at_byte: int,
    *,
    whole_scope: bool = False,
) -> bool:
    for binding in graph.bindings:
        if binding.scope_id != scope_id or binding.name != name:
            continue
        if _import_binding(binding):
            continue
        if whole_scope or _use_byte(binding.span, binding.line) <= at_byte:
            return True
    for entity in graph.entities:
        if entity.name != name or entity.entity_type not in {
            "function",
            "async_function",
            "method",
            "async_method",
        }:
            continue
        if _entity_binding_scope(graph, entity) != scope_id:
            continue
        if whole_scope or entity.start_byte <= at_byte:
            return True
    return False


def _import_binding(binding: Binding) -> bool:
    """An imported name is the callee, not a local replacement for it."""
    for callee in binding.rhs_callees:
        simple = callee.rsplit(".", 1)[-1]
        if simple in {"require", "import", "__import__"}:
            return True
    return False


def _entity_binding_scope(graph: SyntaxGraph, entity: object) -> str | None:
    name = getattr(entity, "name", "")
    start = getattr(entity, "start_byte", None)
    if not isinstance(start, int):
        return None
    for scope in graph.scopes:
        if scope.name != name or scope.span is None:
            continue
        if abs(scope.span.start_byte - start) <= 2:
            return scope.parent_id or "module"
    return None


def _reason_at(defs: tuple[_Def, ...], at_byte: int) -> str | None:
    hitting: _Def | None = None
    for item in defs:
        if item.start_byte <= at_byte:
            hitting = item
        else:
            break
    return hitting.reason if hitting is not None else None


def _freeze_timeline(
    timeline: dict[str, dict[int, _Def]],
    source_pats: Sequence[str],
    *,
    incomplete: bool = False,
    limit_reason: str = "",
) -> TaintState:
    frozen: dict[str, tuple[_Def, ...]] = {}
    for sid, by_index in timeline.items():
        frozen[sid] = tuple(
            sorted(by_index.values(), key=lambda item: (item.start_byte, item.definition_index))
        )
    return TaintState(
        defs=frozen,
        source_pats=tuple(source_pats),
        incomplete=incomplete,
        limit_reason=limit_reason,
    )


def _empty_rhs(binding: Binding) -> bool:
    return (
        not binding.rhs
        and not binding.rhs_idents
        and not binding.rhs_accesses
        and not binding.rhs_callees
    )


def _killing_assignment(binding: Binding) -> bool:
    """Literal RHS that is not a parameter/placeholder — clears this definition only."""
    return (
        binding.rhs_is_literal
        and not binding.rhs_callees
        and not binding.rhs_accesses
        and not _empty_rhs(binding)
    )


def _reach_for_definition(
    binding: Binding,
    computed: str | None,
    previous_this_walk: _Reach | None,
    existing: _Def | None,
) -> _Reach:
    """Taint of this definition. Never copy a later definition backward."""
    existing_reason = existing.reason if existing is not None else None
    if _empty_rhs(binding):
        reason = existing_reason or computed
    elif _killing_assignment(binding):
        reason = None
    elif computed:
        reason = computed
    else:
        reason = existing_reason
    merged = False
    if binding.is_conditional and previous_this_walk is not None:
        reason = _merge_reason(previous_this_walk.reason, reason)
        merged = True
    return _Reach(reason=reason, def_index=binding.definition_index, merged=merged)


def _peel_sanitizer(effect: str) -> tuple[str | None, str]:
    kind: str | None = None
    core = effect
    while core.startswith("sanitized:"):
        rest = core[len("sanitized:") :]
        part, sep, inner = rest.partition(":")
        if not sep or not part:
            break
        if kind is None:
            kind = part
        core = inner
    return kind, core


def _external_for_call(
    call: CallSite, externals: Mapping[str, ExternalCallee]
) -> ExternalCallee | None:
    """Match a call to one known callee.

    A qualified call such as ``obj.method`` or ``importlib...run`` matches only
    that qualified key. The bare name is used for an unqualified call. A
    ``new.Type.method`` call uses the ``Type.method`` key.
    """
    qualified = call.qualified or call.name
    if qualified.startswith("new."):
        qualified = qualified[4:]
    if "." in qualified and qualified != call.name:
        return externals.get(qualified)
    return externals.get(qualified) or externals.get(call.name)


def _call_matching_binding(
    binding: Binding,
    calls: Sequence[CallSite],
    externals: Mapping[str, ExternalCallee],
) -> CallSite | None:
    names: list[str] = []
    for callee in binding.rhs_callees:
        simple = callee.rsplit(".", 1)[-1]
        if callee in externals or simple in externals:
            names.append(callee)
            names.append(simple)
    if not names:
        return None
    chosen = calls_contained_in(
        calls,
        scope_id=binding.scope_id,
        line=binding.line,
        span=binding.span,
        names=set(names),
        outermost=True,
    )
    if len(chosen) == 1:
        return chosen[0]
    if binding.span is None:
        return None
    for call in chosen:
        if call.span is not None and call.span.start_byte == binding.span.start_byte:
            return call
    return None


def _render_effect(
    effect: str,
    call: CallSite,
    scope_id: str,
    source_pats: Sequence[str],
    reaching: dict[str, _Reach],
    at_byte: int,
    ext: ExternalCallee,
) -> str | None:
    kind, core = _peel_sanitizer(effect)
    prefix = f"cross_file:{ext.callee_file}:{ext.callee_function}:"
    if kind:
        prefix += f"sanitized:{kind}:"
    if core.startswith("source:"):
        return prefix + core
    if not core.startswith("param:"):
        return None
    try:
        index = int(core.split(":", 1)[1])
    except ValueError:
        return None
    arg = call.argument_at(index)
    if arg is None:
        return None
    reason = _argument_taint(arg, scope_id, reaching, source_pats, at_byte)
    if not reason:
        return None
    return prefix + core + ":" + reason


def _external_binding_reason(
    binding: Binding,
    calls: Sequence[CallSite],
    externals: Mapping[str, ExternalCallee],
    source_pats: Sequence[str],
    reaching: dict[str, _Reach],
) -> str | None:
    if not binding.rhs_callees or _killing_assignment(binding):
        return None
    call = _call_matching_binding(binding, calls, externals)
    if call is None:
        return None
    ext = _external_for_call(call, externals)
    if ext is None or not ext.return_effects:
        return None
    at_byte = _use_byte(binding.span, binding.line)
    unsanitized: list[str] = []
    sanitized: list[str] = []
    for effect in ext.return_effects:
        rendered = _render_effect(
            effect, call, binding.scope_id, source_pats, reaching, at_byte, ext
        )
        if not rendered:
            continue
        if ":sanitized:" in rendered:
            sanitized.append(rendered)
        else:
            unsanitized.append(rendered)
    chosen = unsanitized or sanitized
    return chosen[0] if chosen else None


def analyze_taint(
    graph: SyntaxGraph,
    sources: Sequence[SourceDefinition],
    externals: Mapping[str, ExternalCallee] | None = None,
) -> TaintState:
    """Compute ordered reaching definitions. Use ``taint_at_use`` at sinks."""
    source_pats = tuple(p for src in sources for p in src.patterns)
    external_map = dict(externals or {})
    calls = graph.calls
    timeline: dict[str, dict[int, _Def]] = {}
    rounds = 0
    changed = True
    interproc_depth = 0
    definition_capped = False
    depth_saturated = False
    blocked_fields, field_depth_limited, field_binding_limited = field_limit_facts(graph)
    alias_info = proven_alias_mirrors(graph)
    route_param_ids = {param_id for route in graph.routes for param_id in route.parameter_ids}
    alias_limited = alias_info.truncated
    mirror_ids: dict[tuple[str, str, int], int] = {}
    next_mirror = 1_000_000
    while changed and rounds < MAX_TAINT_ROUNDS:
        changed = False
        rounds += 1
        applied = 0
        prior = _freeze_timeline(timeline, source_pats)
        return_reasons = _return_reasons(graph, prior, source_pats)
        reaching: dict[str, _Reach] = {}
        for binding in _ordered_bindings(graph):
            applied += 1
            if applied > MAX_DEFINITIONS:
                definition_capped = True
                break
            field_key = (binding.symbol_id, binding.definition_index)
            existing = timeline.get(binding.symbol_id, {}).get(binding.definition_index)
            if field_key in blocked_fields:
                # Drop the definition. Do not write a clean fact that erases an
                # earlier finding on the same path.
                continue
            computed = None
            if binding.symbol_id in route_param_ids and (
                binding.kind is SymbolKind.PARAMETER or binding.declarator == "param"
            ):
                computed = f"source:route:{binding.name}"
            if not computed:
                computed = _binding_taint(
                    binding, source_pats, reaching, return_reasons, graph=graph
                )
            if not computed:
                computed = _sanitized_call_result(graph.language, binding, source_pats, reaching)
            if external_map:
                ext_reason = _external_binding_reason(
                    binding, calls, external_map, source_pats, reaching
                )
                if ext_reason and (not computed or ":sanitized:" in ext_reason):
                    computed = ext_reason
            new = _reach_for_definition(
                binding,
                computed,
                reaching.get(binding.symbol_id),
                existing,
            )
            old_reason = existing.reason if existing is not None else None
            if old_reason != new.reason:
                changed = True
            reaching[binding.symbol_id] = new
            byte = _use_byte(binding.span, binding.line)
            timeline.setdefault(binding.symbol_id, {})[binding.definition_index] = _Def(
                start_byte=byte,
                line=binding.line,
                definition_index=binding.definition_index,
                reason=new.reason,
                merged=new.merged,
            )
            if field_key not in blocked_fields and is_field_path(binding.name):
                mirrored, hit_alias_cap, next_mirror = _mirror_field_timeline(
                    binding,
                    new.reason,
                    byte,
                    timeline,
                    reaching,
                    alias_info,
                    mirror_ids,
                    next_mirror,
                )
                if hit_alias_cap:
                    alias_limited = True
                if mirrored:
                    changed = True
        if interproc_depth < MAX_INTERPROC_DEPTH:
            state = _freeze_timeline(timeline, source_pats)
            live = {s: r.reason for s, r in reaching.items() if r.reason}
            added_edge = False
            for new_id, reason, idx in _interprocedural_edges(
                graph, live, source_pats, state=state
            ):
                existing = timeline.get(new_id, {}).get(idx)
                if existing is not None and existing.reason == reason:
                    continue
                target = _binding_at(graph, new_id, idx)
                byte = (
                    existing.start_byte
                    if existing is not None
                    else _use_byte(target.span, target.line)
                    if target is not None
                    else 0
                )
                line = (
                    existing.line
                    if existing is not None
                    else target.line
                    if target is not None
                    else 1
                )
                timeline.setdefault(new_id, {})[idx] = _Def(
                    start_byte=byte,
                    line=line,
                    definition_index=idx,
                    reason=reason,
                    merged=existing.merged if existing is not None else False,
                )
                changed = True
                added_edge = True
            interproc_depth += 1
            if interproc_depth >= MAX_INTERPROC_DEPTH and added_edge:
                depth_saturated = True
    reasons: list[str] = []
    if changed and rounds >= MAX_TAINT_ROUNDS:
        reasons.append("taint round limit")
    if definition_capped:
        reasons.append("taint definition limit")
    if depth_saturated:
        reasons.append("interprocedural depth limit")
    if field_depth_limited:
        reasons.append("field depth limit")
    if field_binding_limited:
        reasons.append("field binding limit")
    if alias_limited:
        reasons.append("field alias limit")
    return _freeze_timeline(
        timeline,
        source_pats,
        incomplete=bool(reasons),
        limit_reason="; ".join(reasons),
    )


def propagate_taint(
    graph: SyntaxGraph,
    sources: Sequence[SourceDefinition],
) -> dict[str, str]:
    """File-final map (last definition per symbol). Sinks must use ``taint_at_use``."""
    return analyze_taint(graph, sources).final_map()


def taint_at_use(
    graph: SyntaxGraph,
    call: CallSite,
    sources: Sequence[SourceDefinition],
    *,
    argument_indexes: Sequence[int] | None = None,
) -> str | None:
    """Taint reaching ``call`` at its source position."""
    state = analyze_taint(graph, sources)
    return call_taint_reason(call, state.source_pats, state, argument_indexes=argument_indexes)


def _ordered_bindings(graph: SyntaxGraph) -> list[Binding]:
    def key(binding: Binding) -> tuple[int, int, int]:
        if binding.span is not None:
            return (binding.span.start_byte, binding.line, binding.definition_index)
        return (10**12, binding.line, binding.definition_index)

    return sorted(graph.bindings, key=key)


def _merge_reason(left: str | None, right: str | None) -> str | None:
    """Conservative branch merge. One clean side stays visible in the reason."""
    if left and right:
        return left if left == right else f"merge:{left}|{right}"
    present = left or right
    if present:
        return f"merge:{present}|clean"
    return None


def _callee_is_known_sanitizer(callee: str, sanitizers: Sequence[SanitizerDefinition]) -> bool:
    for sanitizer in sanitizers:
        for name in sanitizer.api_names:
            if callee == name or callee.endswith("." + name):
                return True
    return False


def _preserves_taint(callee: str) -> bool:
    """String formatting and ``str()`` keep the taint of their inputs."""
    return callee.rsplit(".", 1)[-1] in _PRESERVING_CALLEES


def field_depth(name: str) -> int:
    """Number of ``.`` and ``[`` steps in a field path. Plain names are depth 0."""
    return sum(1 for ch in name if ch in ".[")


def is_field_path(name: str) -> bool:
    return field_depth(name) > 0


def field_root(name: str) -> tuple[str, str] | None:
    """Split ``obj.payload`` into ``("obj", ".payload")``."""
    if not is_field_path(name):
        return None
    root: list[str] = []
    for ch in name:
        if ch in ".[":
            break
        root.append(ch)
    root_name = "".join(root)
    if not root_name:
        return None
    return root_name, name[len(root_name) :]


def field_flow_allowed(name: str) -> bool:
    depth = field_depth(name)
    if depth == 0:
        return True
    limit = settings.taint_max_field_depth
    return limit > 0 and depth <= limit


def field_path_from_reason(taint: str) -> str:
    """Outermost constant field path recorded in a use-site reason."""
    marker = "field:"
    start = 0
    while True:
        index = taint.find(marker, start)
        if index < 0:
            return ""
        path, sep, _tail = taint[index + len(marker) :].partition(":")
        if sep and path and field_depth(path) > 0 and " " not in path:
            return path
        start = index + len(marker)


@dataclass(frozen=True)
class AliasMirrors:
    """Proven object aliases. ``targets[(scope, name)]`` is ``(other, byte)``."""

    targets: dict[tuple[str, str], tuple[tuple[str, int], ...]]
    truncated: bool = False


def field_limit_facts(graph: SyntaxGraph) -> tuple[set[tuple[str, int]], bool, bool]:
    """Field definitions that must not carry taint, plus which limit fired."""
    blocked: set[tuple[str, int]] = set()
    seen = 0
    depth_limited = False
    binding_capped = False
    limit = settings.taint_max_field_bindings
    for binding in _ordered_bindings(graph):
        if not is_field_path(binding.name):
            continue
        key = (binding.symbol_id, binding.definition_index)
        if not field_flow_allowed(binding.name):
            blocked.add(key)
            depth_limited = True
            continue
        seen += 1
        if limit <= 0 or seen > limit:
            blocked.add(key)
            binding_capped = True
    return blocked, depth_limited, binding_capped


def proven_alias_mirrors(graph: SyntaxGraph) -> AliasMirrors:
    """Unique, non-conditional, single-assignment aliases.

    A later reassignment drops the name entirely. Multi-hop aliases are kept
    only when each step toward the root was assigned no later than the step
    that captured it. Zero ``taint_max_alias_edges`` yields no mirrors.
    """
    limit = settings.taint_max_alias_edges
    if limit <= 0:
        return AliasMirrors(targets={})
    edges = _direct_alias_edges(graph)
    grouped: dict[tuple[str, str], dict[str, int]] = {}
    truncated = False
    for scope, name in sorted(edges):
        targets, hold, cut = _proven_alias_targets(scope, name, edges, limit)
        truncated = truncated or cut
        for target in sorted(targets):
            grouped.setdefault((scope, name), {})[target] = hold
            grouped.setdefault((scope, target), {})[name] = hold
    capped: dict[tuple[str, str], tuple[tuple[str, int], ...]] = {}
    for key, names in grouped.items():
        items = tuple(sorted(names.items()))
        if len(items) > limit:
            truncated = True
            items = items[:limit]
        capped[key] = items
    return AliasMirrors(targets=capped, truncated=truncated)


def _direct_alias_edges(graph: SyntaxGraph) -> dict[tuple[str, str], tuple[str, int]]:
    grouped: dict[tuple[str, str], list[Binding]] = {}
    for binding in graph.bindings:
        if is_field_path(binding.name):
            continue
        grouped.setdefault((binding.scope_id, binding.name), []).append(binding)
    edges: dict[tuple[str, str], tuple[str, int]] = {}
    for key, bindings in grouped.items():
        if len(bindings) != 1:
            continue
        binding = bindings[0]
        if binding.is_conditional or binding.kind is SymbolKind.PARAMETER:
            continue
        if binding.rhs_is_literal or binding.rhs_callees or binding.rhs_accesses:
            continue
        if len(binding.rhs_idents) != 1:
            continue
        target = binding.rhs_idents[0]
        if not target or target == binding.name or is_field_path(target):
            continue
        if any(ch in binding.rhs.strip() for ch in ".([{:"):
            continue
        edges[key] = (target, _use_byte(binding.span, binding.line))
    return edges


def _proven_alias_targets(
    scope: str,
    name: str,
    edges: dict[tuple[str, str], tuple[str, int]],
    limit: int,
) -> tuple[frozenset[str], int, bool]:
    edge = edges.get((scope, name))
    if edge is None:
        return frozenset(), 0, False
    target, hold = edge
    found = [target]
    seen = {name, target}
    current = target
    current_byte = hold
    hops = 1
    while hops < limit:
        nxt = edges.get((scope, current))
        if nxt is None:
            return frozenset(found), hold, False
        nxt_target, nxt_byte = nxt
        if nxt_target in seen or nxt_byte > current_byte:
            return frozenset(), 0, False
        hops += 1
        seen.add(nxt_target)
        found.append(nxt_target)
        current = nxt_target
        current_byte = nxt_byte
    further = edges.get((scope, current)) is not None
    return frozenset(found), hold, further


def _field_reaching(
    accesses: Sequence[str],
    scope_id: str,
    reaching: dict[str, _Reach] | TaintState,
    at_byte: int,
) -> str | None:
    """Taint of a constant field path. Whole-value idents are resolved by the caller first."""
    for access in accesses:
        if not is_field_path(access) or not field_flow_allowed(access):
            continue
        if isinstance(reaching, TaintState):
            reason = reaching.reason_at_name(access, scope_id, at_byte)
        else:
            sid = _resolve_symbol(access, scope_id, reaching)
            reason = reaching[sid].reason if sid is not None else None
        if reason:
            return f"field:{access}:{reason}"
    return None


def _mirror_field_timeline(
    binding: Binding,
    reason: str | None,
    byte: int,
    timeline: dict[str, dict[int, _Def]],
    reaching: dict[str, _Reach],
    alias_info: AliasMirrors,
    mirror_ids: dict[tuple[str, str, int], int],
    next_mirror: int,
) -> tuple[bool, bool, int]:
    """Copy one field definition onto proven aliases. Returns changed, capped, next id."""
    parsed = field_root(binding.name)
    if parsed is None:
        return False, False, next_mirror
    root, suffix = parsed
    mirrors = alias_info.targets.get((binding.scope_id, root), ())
    budget = settings.taint_max_alias_edges
    changed = False
    capped = False
    for offset, (other, hold) in enumerate(mirrors):
        if budget <= 0 or offset >= budget:
            capped = True
            break
        other_name = f"{other}{suffix}"
        if other_name == binding.name or not field_flow_allowed(other_name):
            continue
        sid = f"{binding.scope_id}::{other_name}"
        key = (sid, binding.symbol_id, binding.definition_index)
        index = mirror_ids.get(key)
        if index is None:
            index = next_mirror
            next_mirror += 1
            mirror_ids[key] = index
        previous = timeline.get(sid, {}).get(index)
        start = max(byte, hold)
        if previous is not None and previous.reason == reason and previous.start_byte == start:
            reaching[sid] = _Reach(reason=reason, def_index=index, merged=False)
            continue
        timeline.setdefault(sid, {})[index] = _Def(
            start_byte=start,
            line=binding.line,
            definition_index=index,
            reason=reason,
            merged=False,
        )
        reaching[sid] = _Reach(reason=reason, def_index=index, merged=False)
        changed = True
    return changed, capped, next_mirror


def lexically_shadows(graph: SyntaxGraph | None, scope_id: str, name: str, at_byte: int) -> bool:
    """True when ``name`` is rebound in the scope that contains this use.

    Import bindings are not shadows. A nested function or assignment hides the
    imported name in that function. A module-level assignment hides it only
    after the assignment. Qualified ``fastapi.Query`` is hidden when ``fastapi``
    itself is rebound. Unknown re-exports are not resolved here.
    """
    if graph is None or not name:
        return False
    root = name.split(".", 1)[0]
    return _name_rebound(graph, scope_id or "module", root, at_byte)


def _name_rebound(graph: SyntaxGraph, scope_id: str, name: str, at_byte: int) -> bool:
    """A local binding hides a name only when that name was imported.

    ``req.query`` stays a source when ``req`` is a parameter. ``Query`` stops
    being a FastAPI source when the file imported ``Query`` and a nested
    function or assignment rebinds it. Names that were never imported are not
    framework bindings, so ordinary parameters do not suppress them.
    """
    if not _imported_name(graph, name):
        return False
    scopes = {scope.scope_id: scope for scope in graph.scopes}
    current: str | None = scope_id
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        scope = scopes.get(current)
        function_scope = scope is not None and scope.kind.value in {"function", "method"}
        for binding in graph.bindings:
            if (
                binding.scope_id != current
                or binding.name != name
                or _is_import_binding(graph, binding)
            ):
                continue
            if function_scope:
                return True
            start = binding.span.start_byte if binding.span is not None else 0
            if start < at_byte:
                return True
        for child in graph.scopes:
            if child.parent_id != current or child.name != name:
                continue
            if child.kind.value not in {"function", "method", "class"}:
                continue
            if function_scope:
                return True
            start = child.span.start_byte if child.span is not None else 0
            if start < at_byte:
                return True
        current = scope.parent_id if scope is not None else None
    return False


def _is_import_binding(graph: SyntaxGraph, binding: Binding) -> bool:
    if binding.span is None:
        return False
    for item in graph.imports:
        if item.is_from_import:
            bound = item.alias or item.name or ""
        else:
            bound = item.alias or (item.module or "").split(".", 1)[0]
        if bound == binding.name and item.start_byte == binding.span.start_byte:
            return True
    return False


def _callee_is_source(callee: str, source_pats: Sequence[str]) -> bool:
    if fragment_matches_source(callee, source_pats):
        return True
    return fragment_matches_source(callee.rsplit(".", 1)[-1], source_pats) is not None


def _binding_taint(
    binding: Binding,
    source_pats: Sequence[str],
    reaching: dict[str, _Reach],
    return_reasons: dict[str, str] | None = None,
    graph: SyntaxGraph | None = None,
) -> str | None:
    if binding.rhs_is_literal and not binding.rhs_callees and not binding.rhs_accesses:
        return None
    at_byte = _use_byte(binding.span, binding.line)
    source_callees = [
        callee
        for callee in binding.rhs_callees
        if _callee_is_source(callee, source_pats)
        and not lexically_shadows(graph, binding.scope_id, callee, at_byte)
    ]
    transforming = [
        callee
        for callee in binding.rhs_callees
        if callee not in source_callees
        and not any(callee.startswith(source + ".") for source in source_callees)
        and not _preserves_taint(callee)
    ]
    if not transforming:
        for fragment in (*binding.rhs_accesses, *binding.rhs_callees, *binding.rhs_idents):
            hit = fragment_matches_source(fragment, source_pats)
            if hit and not lexically_shadows(graph, binding.scope_id, fragment, at_byte):
                return f"source:{hit}"
        for ident in binding.rhs_idents:
            sid = _resolve_symbol(ident, binding.scope_id, reaching)
            if sid is not None and reaching[sid].reason:
                return f"from:{sid}:{reaching[sid].reason}"
        field_reason = _field_reaching(
            binding.rhs_accesses,
            binding.scope_id,
            reaching,
            _use_byte(binding.span, binding.line),
        )
        if field_reason:
            return field_reason
        return None
    if graph is not None and return_reasons:
        for call in calls_contained_in(
            graph.calls,
            scope_id=binding.scope_id,
            line=binding.line,
            span=binding.span,
            outermost=True,
        ):
            resolved = _resolve_local_callee(graph, call)
            if resolved is None:
                continue
            callee, _instance = resolved
            callee_scope = _scope_for_entity(graph, callee)
            if callee_scope and callee_scope in return_reasons:
                name = getattr(callee, "name", "") or call.name
                return f"from_return:{name}:{return_reasons[callee_scope]}"
    return None


def _sanitized_call_result(
    language: str,
    binding: Binding,
    source_pats: Sequence[str],
    reaching: dict[str, _Reach],
) -> str | None:
    """A known effective sanitizer call keeps taint, marked with its kind."""
    if not binding.rhs_callees:
        return None
    vocab = vocab_for(language)
    if vocab is None:
        return None
    sanitizer: SanitizerDefinition | None = None
    for candidate in vocab.sanitizers:
        if not candidate.effective:
            continue
        for name in candidate.api_names:
            for callee in binding.rhs_callees:
                if callee == name or callee.endswith("." + name):
                    sanitizer = candidate
                    break
            if sanitizer is not None:
                break
        if sanitizer is not None:
            break
    if sanitizer is None:
        return None
    inner: str | None = None
    for fragment in (*binding.rhs_accesses, *binding.rhs_idents):
        if any(
            fragment == callee or callee.endswith("." + fragment) for callee in binding.rhs_callees
        ):
            continue
        hit = fragment_matches_source(fragment, source_pats)
        if hit:
            inner = f"source:{hit}"
            break
    if inner is None:
        for ident in binding.rhs_idents:
            if any(
                ident == callee or callee.endswith("." + ident) or callee.startswith(ident + ".")
                for callee in binding.rhs_callees
            ):
                continue
            sid = _resolve_symbol(ident, binding.scope_id, reaching)
            if sid is not None and reaching[sid].reason:
                inner = f"from:{ident}:{reaching[sid].reason}"
                break
    if inner is None:
        inner = _field_reaching(
            binding.rhs_accesses,
            binding.scope_id,
            reaching,
            _use_byte(binding.span, binding.line),
        )
    if inner is None:
        return None
    return f"sanitized:{sanitizer.kind}:{inner}"


def _return_reasons(
    graph: SyntaxGraph,
    reaching: dict[str, _Reach] | TaintState,
    source_pats: Sequence[str],
) -> dict[str, str]:
    """Map a function scope to the taint reason of its return."""
    out: dict[str, str] = {}
    for ret in graph.returns:
        reason: str | None = None
        for fragment in (*ret.accesses, *ret.idents):
            hit = fragment_matches_source(fragment, source_pats)
            if hit:
                reason = f"return:source:{hit}"
                break
        if reason is None:
            ret_byte = _use_byte(ret.span, ret.line)
            for ident in ret.idents:
                if isinstance(reaching, TaintState):
                    ident_reason = reaching.reason_at_name(ident, ret.scope_id, ret_byte)
                    if ident_reason:
                        reason = f"return:{ident}:{ident_reason}"
                        break
                    continue
                sid = _resolve_symbol(ident, ret.scope_id, reaching)
                if sid is not None:
                    reason = f"return:{sid}"
                    break
        if reason is None and isinstance(reaching, TaintState):
            field_reason = _field_reaching(
                ret.accesses, ret.scope_id, reaching, _use_byte(ret.span, ret.line)
            )
            if field_reason:
                reason = f"return:{field_reason}"
        if reason:
            out[ret.scope_id] = reason
    return out


def _resolve_symbol(
    name: str, scope_id: str, reaching: dict[str, _Reach] | dict[str, str]
) -> str | None:
    """Resolve ``name`` in ``scope_id``, then enclosing scopes, never sibling functions."""
    current: str | None = scope_id
    while current:
        candidate = f"{current}::{name}"
        if candidate in reaching:
            value = reaching[candidate]
            reason = value.reason if isinstance(value, _Reach) else value
            if reason:
                return candidate
            return None
        if "/" not in current:
            break
        current = current.rsplit("/", 1)[0]
    module = f"module::{name}"
    if module in reaching:
        value = reaching[module]
        reason = value.reason if isinstance(value, _Reach) else value
        if reason:
            return module
        return None
    return None


def _binding_at(graph: SyntaxGraph, symbol_id: str, def_index: int) -> Binding | None:
    for binding in graph.bindings:
        if binding.symbol_id == symbol_id and binding.definition_index == def_index:
            return binding
    return None


def _parameter_def_index(graph: SyntaxGraph, symbol_id: str) -> int:
    matches = [b for b in graph.bindings if b.symbol_id == symbol_id]
    params = [
        b
        for b in matches
        if b.kind is SymbolKind.PARAMETER
        or b.declarator == "param"
        or (_empty_rhs(b) and b.is_declaration)
    ]
    if params:
        return min(b.definition_index for b in params)
    if matches:
        return min(b.definition_index for b in matches)
    return 0


def _interprocedural(
    graph: SyntaxGraph,
    tainted: dict[str, str],
    source_pats: Sequence[str] = (),
    *,
    state: TaintState | None = None,
) -> list[tuple[str, str]]:
    """Parameter and return edges (symbol, reason). Tests inspect this shape."""
    return [
        (sid, reason)
        for sid, reason, _idx in _interprocedural_edges(graph, tainted, source_pats, state=state)
    ]


def _interprocedural_edges(
    graph: SyntaxGraph,
    tainted: dict[str, str],
    source_pats: Sequence[str] = (),
    *,
    state: TaintState | None = None,
) -> list[tuple[str, str, int]]:
    """Parameter and return propagation only when the callee is unique.

    Call-argument taint is attached to the *parameter* definition, not a later
    reassignment of the same name. Return taint is attached to the assignment
    whose RHS is that call, not a later definition of the same symbol.
    """
    new: list[tuple[str, str, int]] = []
    use_state = state or TaintState.from_final_map(tainted, source_pats)
    for call in graph.calls:
        resolved = _resolve_local_callee(graph, call)
        if resolved is None:
            continue
        callee, instance = resolved
        callee_scope = _scope_for_entity(graph, callee)
        if callee_scope is None:
            continue
        params = list(getattr(callee, "parameters", ()) or ())
        qualified = call.qualified or call.name
        offset = (
            receiver_argument_offset(
                [param.name for param in params],
                tuple(getattr(callee, "decorators", ()) or ()),
                instance=instance,
            )
            if "." in qualified and qualified != call.name
            else 0
        )
        args = list(call.arguments)
        if not args and (call.argument_idents or call.argument_accesses):
            args = [
                CallArgument(
                    index=0,
                    text=call.argument_text,
                    is_literal=call.argument_is_literal,
                    idents=call.argument_idents,
                    accesses=call.argument_accesses,
                    dynamic=call.dynamic,
                )
            ]
        at_byte = _use_byte(call.span, call.line)
        for arg in args:
            reason = _argument_taint(arg, call.scope_id, use_state, source_pats, at_byte)
            if not reason:
                continue
            slot = arg.index + offset
            if slot < len(params):
                pname = params[slot].name
                sid = f"{callee_scope}::{pname}"
                new.append((sid, f"callarg:{arg.index}:{reason}", _parameter_def_index(graph, sid)))
    for ret in graph.returns:
        ret_reason = None
        for fragment in (*ret.accesses, *ret.idents):
            hit = fragment_matches_source(fragment, source_pats)
            if hit:
                ret_reason = f"return:source:{hit}"
                break
        if ret_reason is None:
            ret_byte = _use_byte(ret.span, ret.line)
            for ident in ret.idents:
                reason = use_state.reason_at_name(ident, ret.scope_id, ret_byte)
                if reason:
                    ret_reason = f"return:{ident}:{reason}"
                    break
            if ret_reason is None:
                field_reason = _field_reaching(ret.accesses, ret.scope_id, use_state, ret_byte)
                if field_reason:
                    ret_reason = f"return:{field_reason}"
        if ret_reason is None:
            continue
        callee_entity = _entity_for_scope(graph, ret.scope_id)
        if callee_entity is None:
            continue
        for binding in graph.bindings:
            if binding.scope_id == ret.scope_id:
                continue
            for call in calls_contained_in(
                graph.calls,
                scope_id=binding.scope_id,
                line=binding.line,
                span=binding.span,
                outermost=True,
            ):
                resolved = _resolve_local_callee(graph, call)
                if resolved is None:
                    continue
                entity, _instance = resolved
                if _same_entity(entity, callee_entity):
                    new.append((binding.symbol_id, ret_reason, binding.definition_index))
                    break
    return new


def _known_method(graph: SyntaxGraph, call: CallSite) -> tuple[object, bool] | None:
    """Resolve a method only when the receiver's class is already established.

    A unique method name is not enough. Unknown receivers, ambiguous
    constructions, and inherited methods stay unresolved.
    """
    qualified = call.qualified or ""
    if "." not in qualified:
        return None
    receiver, method = qualified.rsplit(".", 1)
    if receiver.endswith("()"):
        class_name: str | None = receiver[:-2].rsplit(".", 1)[-1]
        instance = True
    elif any(
        entity.entity_type == "class" and entity.name == receiver for entity in graph.entities
    ):
        class_name = receiver
        instance = False
    elif receiver in {"self", "cls"}:
        class_name = _enclosing_class(graph, call.scope_id)
        instance = True
    else:
        binding = _reaching_binding(graph, receiver, call.scope_id, _use_byte(call.span, call.line))
        if binding is None:
            return None
        found = _constructed_classes(graph, binding)
        if len(found) != 1:
            return None
        class_name = next(iter(found))
        instance = True
    if not class_name:
        return None
    entity = _method_on(graph, class_name, method)
    if entity is None:
        return None
    return entity, instance


def _resolve_local_callee(graph: SyntaxGraph, call: CallSite) -> tuple[object, bool] | None:
    """Resolve a same-file callee only when the call uniquely names that entity."""
    qualified = call.qualified or call.name
    if "." in qualified and qualified != call.name:
        return _known_method(graph, call)
    functions = [
        ent
        for ent in graph.entities
        if ent.entity_type in {"function", "async_function", "method", "async_method"}
    ]
    by_name: dict[str, list[object]] = {}
    by_qualified: dict[str, list[object]] = {}
    for ent in functions:
        by_name.setdefault(ent.name, []).append(ent)
        by_qualified.setdefault(ent.qualified_name, []).append(ent)
        if ent.parent:
            by_qualified.setdefault(f"{ent.parent}.{ent.name}", []).append(ent)
    if _imported_name(graph, call.name) and by_name.get(call.name):
        return None
    candidates = by_qualified.get(qualified) or by_name.get(call.name) or []
    callee: object | None = None
    if len(candidates) == 1:
        callee = candidates[0]
    else:
        qualified_hits = [
            ent
            for ent in candidates
            if getattr(ent, "qualified_name", "") == call.qualified
            or call.qualified.endswith(f".{getattr(ent, 'name', '')}")
        ]
        if len(qualified_hits) == 1:
            callee = qualified_hits[0]
    if callee is None:
        return None
    return callee, False


def _imported_name(graph: SyntaxGraph, name: str) -> bool:
    if not name:
        return False
    for item in graph.imports:
        bound = item.alias or (
            item.name if item.is_from_import else (item.module or "").split(".", 1)[0]
        )
        if bound == name:
            return True
    return False


def _entity_for_scope(graph: SyntaxGraph, scope_id: str) -> object | None:
    hits = [
        entity
        for entity in graph.entities
        if entity.entity_type in {"function", "async_function", "method", "async_method"}
        and _scope_for_entity(graph, entity) == scope_id
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def _same_entity(left: object, right: object) -> bool:
    left_id = getattr(left, "node_id", "") or ""
    right_id = getattr(right, "node_id", "") or ""
    if left_id and right_id:
        return left_id == right_id
    return (
        getattr(left, "name", None) == getattr(right, "name", None)
        and getattr(left, "start_byte", None) == getattr(right, "start_byte", None)
        and getattr(left, "qualified_name", None) == getattr(right, "qualified_name", None)
    )


def _method_on(graph: SyntaxGraph, class_name: str, method_name: str) -> object | None:
    hits = [
        entity
        for entity in graph.entities
        if entity.entity_type in {"method", "async_method"}
        and entity.parent == class_name
        and entity.name == method_name
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def _enclosing_class(graph: SyntaxGraph, scope_id: str) -> str | None:
    scopes = {scope.scope_id: scope for scope in graph.scopes}
    current: str | None = scope_id
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        scope = scopes.get(current)
        if scope is not None and scope.kind.value == "class":
            return scope.name
        if "/" not in current:
            break
        current = current.rsplit("/", 1)[0]
    return None


def _reaching_binding(graph: SyntaxGraph, name: str, scope_id: str, at_byte: int) -> Binding | None:
    current: str | None = scope_id or "module"
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        hits = [
            binding
            for binding in graph.bindings
            if binding.scope_id == current
            and binding.name == name
            and _use_byte(binding.span, binding.line) <= at_byte
        ]
        if hits:
            chosen = max(
                hits,
                key=lambda binding: (
                    _use_byte(binding.span, binding.line),
                    binding.definition_index,
                ),
            )
            if chosen.is_conditional:
                return None
            return chosen
        if "/" not in current:
            break
        current = current.rsplit("/", 1)[0]
    if "module" not in seen:
        return _reaching_binding(graph, name, "module", at_byte)
    return None


def _constructed_classes(graph: SyntaxGraph, binding: Binding) -> set[str]:
    classes = {entity.name for entity in graph.entities if entity.entity_type == "class"}
    found: set[str] = set()
    for callee in binding.rhs_callees:
        simple = callee.rsplit(".", 1)[-1]
        if simple in classes:
            found.add(simple)
    return found


def _scope_for_entity(graph: SyntaxGraph, entity: object) -> str | None:
    name = getattr(entity, "name", "")
    parent = getattr(entity, "parent", None)
    start = getattr(entity, "start_byte", None)
    matches: list[str] = []
    for scope in graph.scopes:
        if scope.name != name or scope.kind.value not in {"function", "method"}:
            continue
        if parent and parent not in scope.scope_id and parent not in (scope.parent_id or ""):
            continue
        if start is not None and scope.span is not None:
            if abs(scope.span.start_byte - int(start)) <= 2:
                return scope.scope_id
        matches.append(scope.scope_id)
    if len(matches) == 1:
        return matches[0]
    return None


def call_taint_reason(
    call: CallSite,
    source_pats: Sequence[str],
    tainted: TaintState | dict[str, str],
    *,
    argument_indexes: Sequence[int] | None = None,
    sanitizers: Sequence[SanitizerDefinition] = (),
    transparent_callees: Sequence[str] = (),
    graph: SyntaxGraph | None = None,
) -> str | None:
    state = (
        tainted
        if isinstance(tainted, TaintState)
        else TaintState.from_final_map(tainted, source_pats)
    )
    at_byte = _use_byte(call.span, call.line)
    indexes = list(argument_indexes or [])
    if call.arguments and indexes:
        for index in indexes:
            arg = call.argument_at(index)
            if arg is None:
                continue
            reason = _argument_taint(
                arg,
                call.scope_id,
                state,
                source_pats,
                at_byte,
                sanitizers,
                transparent_callees,
                graph,
            )
            if reason:
                return reason
        return None
    if call.arguments:
        for arg in call.arguments:
            if arg.keyword:
                continue
            reason = _argument_taint(
                arg,
                call.scope_id,
                state,
                source_pats,
                at_byte,
                sanitizers,
                transparent_callees,
                graph,
            )
            if reason:
                return reason
        return None
    if call.argument_is_literal and not call.dynamic and not call.argument_idents:
        return None
    for fragment in (*call.argument_accesses, *call.argument_idents):
        hit = fragment_matches_source(fragment, source_pats)
        if hit:
            return f"source:{hit}"
    for ident in call.argument_idents:
        reason = state.reason_at_name(ident, call.scope_id, at_byte)
        if reason:
            return f"from:{ident}:{reason}"
    return _field_reaching(call.argument_accesses, call.scope_id, state, at_byte)


def _argument_taint(
    arg: CallArgument,
    scope_id: str,
    reaching: dict[str, _Reach] | TaintState,
    source_pats: Sequence[str],
    at_byte: int = 0,
    sanitizers: Sequence[SanitizerDefinition] = (),
    transparent_callees: Sequence[str] = (),
    graph: SyntaxGraph | None = None,
) -> str | None:
    if arg.is_literal and not arg.dynamic and not arg.idents and not arg.accesses:
        return None
    if arg.callees and any(
        not _callee_is_source(callee, source_pats)
        and not _preserves_taint(callee)
        and not _callee_named(callee, transparent_callees)
        and not _callee_is_known_sanitizer(callee, sanitizers)
        for callee in arg.callees
    ):
        # An unknown call may replace its inputs. Do not assume it preserves taint.
        return None
    for fragment in (*arg.accesses, *arg.callees, *arg.idents):
        hit = fragment_matches_source(fragment, source_pats)
        if hit and not lexically_shadows(graph, scope_id, fragment, at_byte):
            return f"source:{hit}"
    if isinstance(reaching, TaintState):
        for ident in arg.idents:
            reason = reaching.reason_at_name(ident, scope_id, at_byte)
            if reason:
                return f"from:{ident}:{reason}"
        return _field_reaching(arg.accesses, scope_id, reaching, at_byte)
    for ident in arg.idents:
        sid = _resolve_symbol(ident, scope_id, reaching)
        if sid is not None and reaching[sid].reason:
            return f"from:{sid}:{reaching[sid].reason}"
    return _field_reaching(arg.accesses, scope_id, reaching, at_byte)


def call_matches_sink(call: CallSite, sink: SinkDefinition) -> bool:
    haystacks = (call.qualified, call.name)
    for name in sink.api_names:
        name_n = _canon_qual(name)
        for hay in haystacks:
            hay_n = _canon_qual(hay)
            if hay == name or hay_n == name_n:
                return _receiver_ok(call, sink)
            if (
                hay.endswith(f".{name}")
                or hay.endswith(f"::{name}")
                or hay.endswith(f"->{name}")
                or hay_n.endswith(f".{name_n}")
            ):
                return _receiver_ok(call, sink)
            if ("." in name_n or "::" in name or "->" in name) and (
                hay_n == name_n or hay_n.endswith("." + name_n)
            ):
                return _receiver_ok(call, sink)
    return False


def _receiver_ok(call: CallSite, sink: SinkDefinition) -> bool:
    if not sink.receiver_names:
        return True
    left = _canon_qual(call.qualified)
    return any(_canon_qual(name) in left for name in sink.receiver_names)


def argument_is_constant(call: CallSite, *, argument_indexes: Sequence[int] | None = None) -> bool:
    indexes = list(argument_indexes or [])
    if call.arguments and indexes:
        relevant = [a for i in indexes if (a := call.argument_at(i)) is not None]
        if not relevant:
            return False
        return all(a.is_literal and not a.dynamic and not a.idents for a in relevant)
    if call.arguments:
        return all(a.is_literal and not a.dynamic and not a.idents for a in call.arguments)
    if call.argument_idents:
        return False
    if call.dynamic:
        return False
    if call.argument_is_literal:
        return True
    text = call.argument_text.strip()
    if not text:
        return True
    if text[0] in {'"', "'", "`"} and text[-1] == text[0]:
        return True
    return False


def sanitizer_intervened(
    call: CallSite,
    sanitizers: Sequence[SanitizerDefinition],
    *,
    allowed_kinds: Sequence[str] = (),
    argument_indexes: Sequence[int] | None = None,
) -> SanitizerDefinition | None:
    """Match sanitizers on the relevant argument's callees, never argument text."""
    allowed = set(allowed_kinds) if allowed_kinds else None
    indexes = list(argument_indexes or [])
    args: list[CallArgument] = []
    if call.arguments and indexes:
        args = [a for i in indexes if (a := call.argument_at(i)) is not None]
    elif call.arguments:
        args = list(call.arguments)
    callees: tuple[str, ...] = ()
    for arg in args:
        callees += arg.callees
    if not callees:
        callees = call.argument_accesses
    for sanitizer in sanitizers:
        if allowed is not None and sanitizer.kind not in allowed:
            continue
        for name in sanitizer.api_names:
            name_n = _canon_qual(name)
            for callee in callees:
                callee_n = _canon_qual(callee)
                if callee == name or callee_n == name_n:
                    return sanitizer
                # Qualified sanitizers match only their full API name.
                # ``my_module.escape`` is not ``html.escape``.
                if "." in name_n and callee_n.endswith("." + name_n):
                    return sanitizer
    return None


def _callee_named(callee: str, names: Sequence[str]) -> bool:
    simple = callee.rsplit(".", 1)[-1]
    return callee in names or simple in names


def looks_parameterized_sql(call: CallSite | str) -> bool:
    """Syntax-aware parameterization, not a substring search on the call text."""
    if isinstance(call, str):
        # Legacy helper used by older tests; refuse concatenated-looking text.
        if "+" in call or "${" in call:
            return False
        return bool(_SQL_PLACEHOLDER.search(call)) and "," in call
    if call.dynamic:
        return False
    if call.arguments and _association_argument(call.arguments[0]):
        return True
    if len(call.arguments) >= 2:
        query = call.arguments[0]
        if _static_placeholder_query(query):
            return True
    if not call.arguments and not call.dynamic and call.argument_is_literal:
        return bool(_SQL_PLACEHOLDER.search(call.argument_text))
    return False


def _association_argument(query: CallArgument) -> bool:
    """A keyword or ``name: value`` argument is not a raw SQL string.

    ``where(name: user)`` binds a column. ``where(user_sql)`` and
    ``where("name = " + user)`` do not take this path.
    """
    if query.dynamic or query.is_literal:
        return False
    if query.keyword:
        return True
    text = query.text.strip()
    if not text or text[0] in {"'", '"', "`", "["}:
        return False
    if ":" not in text:
        return False
    head = text.partition(":")[0].strip()
    return bool(head) and head.replace("_", "").isalnum() and not head[0].isdigit()


def _static_placeholder_query(query: CallArgument) -> bool:
    """A fixed SQL string with placeholders. A ``text()`` wrapper may hold that string."""
    if query.dynamic or query.idents or not _SQL_PLACEHOLDER.search(query.text):
        return False
    if query.is_literal and not query.callees:
        return True
    return bool(query.callees) and all(_callee_named(callee, ("text",)) for callee in query.callees)


def identifiers_in(text: str) -> list[str]:
    return [m.group(0) for m in _IDENT.finditer(text)]


def language_vocab(graph: SyntaxGraph) -> LanguageSecurityVocab | None:
    return vocab_for(graph.language)


def applicable_sanitizer_kinds(sink: SinkDefinition) -> tuple[str, ...]:
    if sink.sanitizer_kinds:
        return sink.sanitizer_kinds
    kinds = SINK_SANITIZER_KINDS.get(sink.vulnerability_class, frozenset())
    return tuple(kinds)
