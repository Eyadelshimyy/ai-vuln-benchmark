"""Taint-aware source→sink security rules using semantic vocabularies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from app.analyzers.framework_detector import FrameworkInfo
from app.domain.security import VulnerabilityClass
from app.parsing.model import CallSite, SyntaxGraph
from app.security.definitions import (
    LanguageSecurityVocab,
    PathIssueKind,
    SinkCertainty,
    SinkDefinition,
    SourceDefinition,
)
from app.security.rules.base import RuleDocumentation, SecurityObservation, SecurityRule
from app.security.taint import (
    ExternalCallee,
    TaintState,
    _external_for_call,
    analyze_taint,
    applicable_sanitizer_kinds,
    argument_is_constant,
    builtin_call_shadowed,
    call_matches_sink,
    call_taint_reason,
    field_path_from_reason,
    language_vocab,
    looks_parameterized_sql,
    reaching_sanitizer_kind,
    sanitizer_intervened,
    vocab_sources,
)


class TaintFlowRule(SecurityRule):
    """Flag user-controlled data reaching a dangerous sink.

    Does not prove exploitability. Requires a syntax graph, not a keyword search.
    """

    def __init__(
        self,
        *,
        rule_id: str,
        vulnerability_class: VulnerabilityClass,
        documentation: RuleDocumentation,
    ) -> None:
        self.rule_id = rule_id
        self.vulnerability_class = vulnerability_class
        self.documentation = documentation

    def check(
        self,
        graph: SyntaxGraph,
        *,
        frameworks: Sequence[FrameworkInfo] = (),
        project: object | None = None,
    ) -> list[SecurityObservation]:
        vocab = language_vocab(graph)
        if vocab is None:
            return []
        sinks = [
            sink
            for sink in vocab.sinks
            if sink.vulnerability_class is self.vulnerability_class
            or (
                self.vulnerability_class is VulnerabilityClass.UNSAFE_DESERIALIZATION
                and sink.vulnerability_class is VulnerabilityClass.POTENTIAL_UNSAFE_DESERIALIZATION
            )
            or (
                self.vulnerability_class is VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL
                and sink.vulnerability_class is VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL
            )
        ]
        if not sinks:
            return []
        sources = _sources_for_graph(vocab, frameworks, graph)
        source_pats = tuple(p for src in sources for p in src.patterns)
        externals = _externals(project, graph.file_path)
        taint_state = analyze_taint(graph, sources, externals=externals or None)
        framework_name = (
            ",".join(
                fw.name
                for fw in frameworks
                if fw.language.split("/")[0] in {graph.language, "javascript", "typescript"}
            )
            or ""
        )
        observations: list[SecurityObservation] = []
        seen: set[tuple[str, int, str, str]] = set()
        for call in graph.calls:
            matched = next((sink for sink in sinks if call_matches_sink(call, sink)), None)
            if matched is None:
                continue
            if builtin_call_shadowed(graph, call):
                continue
            if (
                matched.vulnerability_class is VulnerabilityClass.COMMAND_INJECTION
                and _safe_command_argv(call)
            ):
                continue
            if matched.required_context and matched.required_context not in graph.semantic_context:
                if matched.required_context != graph.file_context:
                    continue
            indexes = matched.argument_indexes
            if argument_is_constant(call, argument_indexes=indexes or None):
                continue
            if (
                self.vulnerability_class is VulnerabilityClass.SQL_INJECTION
                and looks_parameterized_sql(call)
            ):
                continue
            taint = call_taint_reason(
                call,
                source_pats,
                taint_state,
                argument_indexes=indexes or None,
                sanitizers=vocab.sanitizers,
                transparent_callees=_sql_text_wrappers(self.vulnerability_class),
                graph=graph,
            )
            if not taint and not call.dynamic:
                continue
            if indexes and call.arguments:
                # Taint must touch a dangerous argument, not an unrelated slot.
                if not taint:
                    continue
            kinds = applicable_sanitizer_kinds(matched)
            sanitizer = sanitizer_intervened(
                call,
                vocab.sanitizers,
                allowed_kinds=kinds,
                argument_indexes=indexes or None,
            )
            if sanitizer is not None and sanitizer.effective:
                continue
            if _sanitizer_blocks_reason(taint or "", kinds):
                continue
            confidence = "high" if taint else "medium"
            if sanitizer is not None:
                confidence = "low"
            vuln = matched.vulnerability_class
            if vuln is VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL:
                path_kind = _path_kind(call, taint)
            else:
                path_kind = None
            key = (graph.file_path, call.line, call.qualified, str(indexes))
            if key in seen:
                continue
            seen.add(key)
            local_extra: dict[str, str] = {}
            if taint_state.incomplete:
                local_extra["analysis_incomplete"] = taint_state.limit_reason or "bounded"
            if taint and "merge:" in taint:
                local_extra["branch_merge"] = "true"
            observations.append(
                _observation(
                    self,
                    graph,
                    call,
                    matched,
                    confidence,
                    taint=taint or "dynamic_construction",
                    framework=framework_name or graph.framework,
                    path_kind=path_kind,
                    sanitizer=sanitizer.sanitizer_id if sanitizer else "",
                    argument_index=indexes[0] if indexes else matched.argument_index,
                    extra=local_extra or None,
                )
            )
        if externals:
            observations.extend(
                _cross_file_observations(
                    self,
                    graph,
                    externals,
                    sinks,
                    taint_state,
                    source_pats,
                    vocab,
                    framework_name,
                    seen,
                )
            )
        return observations


_IMPORT_GATED_SOURCES = {
    "Query": ("fastapi", "nestjs"),
    "Path": ("fastapi",),
    "Body": ("fastapi", "nestjs"),
    "Param": ("nestjs",),
    "Headers": ("nestjs",),
}


# Bare names that are sources only for these catalog entries. Rust ``Query``
# and other language sources stay ungated.
_GATED_SOURCE_IDS = frozenset({"fastapi.http", "nest.http", "js.nest"})


def _sources_for_graph(
    vocab: LanguageSecurityVocab,
    frameworks: Sequence[FrameworkInfo],
    graph: SyntaxGraph,
) -> tuple[SourceDefinition, ...]:
    """Project frameworks plus imports this file actually binds.

    A FastAPI or NestJS manifest is not required when the file imports the
    name. A manifest does not make an unbound ``Query`` a source.
    """
    selected = list(vocab_sources(vocab, frameworks))
    seen = {source.source_id for source in selected}
    for extras in vocab.extra_sources_by_framework.values():
        for source in extras:
            if source.source_id in seen:
                continue
            if _bound_patterns(graph, source):
                selected.append(source)
                seen.add(source.source_id)
    return _visible_sources(graph, tuple(selected))


def _visible_sources(
    graph: SyntaxGraph, sources: tuple[SourceDefinition, ...]
) -> tuple[SourceDefinition, ...]:
    """Rewrite gated framework sources to the names this file binds."""
    visible: list[SourceDefinition] = []
    for source in sources:
        if source.source_id not in _GATED_SOURCE_IDS:
            visible.append(source)
            continue
        unique = _bound_patterns(graph, source)
        if not unique:
            continue
        if unique != source.patterns:
            source = replace(source, patterns=unique)
        visible.append(source)
    return tuple(visible)


def _bound_patterns(graph: SyntaxGraph, source: SourceDefinition) -> tuple[str, ...]:
    names: list[str] = []
    for pattern in source.patterns:
        required = _IMPORT_GATED_SOURCES.get(pattern)
        if required is None or source.source_id not in _GATED_SOURCE_IDS:
            names.append(pattern)
            continue
        names.extend(_bound_framework_names(graph, pattern, required))
    return tuple(dict.fromkeys(names))


def _bound_framework_names(
    graph: SyntaxGraph, pattern: str, frameworks: tuple[str, ...]
) -> tuple[str, ...]:
    """Local names bound to a framework import, excluding a shadowing function.

    ``from fastapi import Query as Q`` binds ``Q``. ``import fastapi as fa``
    binds ``fa.Query``. A module-level ``def Query`` hides the from-import.
    Re-exports through an unknown local module are not inferred.
    """
    names: list[str] = []
    for item in graph.imports:
        module = item.module or ""
        if not _module_matches(module, frameworks):
            continue
        if item.is_from_import:
            imported = item.name or ""
            if imported != pattern:
                continue
            bound = item.alias or imported
            if not bound or bound == "*":
                continue
            names.append(bound)
            continue
        root = item.alias or module.split(".", 1)[0]
        if not root or root == "*":
            continue
        names.append(f"{root}.{pattern}")
    return tuple(dict.fromkeys(names))


def _module_matches(module: str, tokens: tuple[str, ...]) -> bool:
    parts = [part for part in module.replace("\\", "/").replace(".", "/").split("/") if part]
    for token in tokens:
        if token in parts or f"@{token}" in parts:
            return True
    return False


_SHELL_PROGRAMS = frozenset(
    {"sh", "bash", "dash", "zsh", "ksh", "cmd", "powershell", "pwsh", "cmd.exe"}
)


def _safe_command_argv(call: CallSite) -> bool:
    """True only when the program is a fixed non-shell literal and shell is off.

    ``subprocess.run(["git", user])`` and ``exec.Command("git", user)`` are not
    shell injection. ``subprocess.run([user])``, ``shell=True``, and
    ``["/bin/sh", "-c", user]`` stay dangerous. Keyword ``shell`` is read from
    the argument node, not from a substring of the file.
    """
    if _shell_keyword(call):
        return False
    positional = [argument for argument in call.arguments if not argument.keyword]
    if not positional:
        return False
    elements = _list_items(positional[0].text)
    if elements is not None:
        # subprocess.run(["git", user]) — the list's first item is the program.
        program = _literal_string(elements[0]) if elements else None
    elif len(positional) >= 2:
        # exec.Command("git", user) — a later argument is argv, not the program.
        program = _literal_string(positional[0].text)
    else:
        # eval "$q" and system(user) pass one command string. That string is
        # not a fixed executable name.
        return False
    if program is None:
        return False
    base = program.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return base not in _SHELL_PROGRAMS


def _shell_keyword(call: CallSite) -> bool:
    for argument in call.arguments:
        compact = "".join(argument.text.split()).lower()
        if argument.keyword == "shell" and compact in {"true", "1"}:
            return True
        if not argument.keyword and compact in {"shell=true", "shell=1"}:
            return True
    return False


def _list_items(text: str) -> list[str] | None:
    raw = text.strip()
    if len(raw) < 2 or raw[0] != "[" or raw[-1] != "]":
        return None
    return _split_commas(raw[1:-1])


def _split_commas(body: str) -> list[str]:
    items: list[str] = []
    buf: list[str] = []
    quote = ""
    depth = 0
    for char in body:
        if quote:
            buf.append(char)
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
            buf.append(char)
            continue
        if char in "([{":
            depth += 1
            buf.append(char)
            continue
        if char in ")]}":
            depth = max(0, depth - 1)
            buf.append(char)
            continue
        if char == "," and depth == 0:
            items.append("".join(buf).strip())
            buf = []
            continue
        buf.append(char)
    tail = "".join(buf).strip()
    if tail:
        items.append(tail)
    return items


def _literal_string(token: str) -> str | None:
    text = token.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return None


def _externals(project: object | None, file_path: str) -> dict[str, ExternalCallee]:
    if project is None:
        return {}
    getter = getattr(project, "externals_for", None)
    if not callable(getter):
        return {}
    table = getter(file_path)
    return dict(table or {})


def _sanitizer_blocks_reason(taint: str, kinds: tuple[str, ...] | list[str]) -> bool:
    if not taint or not kinds:
        return False
    kind = reaching_sanitizer_kind(taint)
    return kind is not None and kind in set(kinds)


def _cross_file_observations(
    rule: TaintFlowRule,
    graph: SyntaxGraph,
    externals: dict[str, ExternalCallee],
    sinks: list[SinkDefinition],
    taint_state: TaintState,
    source_pats: tuple[str, ...],
    vocab: object,
    framework_name: str,
    seen: set[tuple[str, int, str, str]],
) -> list[SecurityObservation]:
    if not isinstance(vocab, LanguageSecurityVocab):
        return []
    observations: list[SecurityObservation] = []
    for call in graph.calls:
        ext = _external_for_call(call, externals)
        if ext is None:
            continue
        if ext.callee_file == graph.file_path and ext.relationship != "alias":
            continue
        for index, vuln_value, sink_name, sink_line in ext.param_sinks:
            if vuln_value != rule.vulnerability_class.value:
                continue
            matched = next(
                (sink for sink in sinks if sink.vulnerability_class.value == vuln_value),
                None,
            )
            if matched is None:
                continue
            if argument_is_constant(call, argument_indexes=(index,)):
                continue
            taint = call_taint_reason(
                call,
                source_pats,
                taint_state,
                argument_indexes=(index,),
                sanitizers=vocab.sanitizers,
                transparent_callees=_sql_text_wrappers(rule.vulnerability_class),
                graph=graph,
            )
            if not taint:
                continue
            kinds = applicable_sanitizer_kinds(matched)
            sanitizer = sanitizer_intervened(
                call,
                vocab.sanitizers,
                allowed_kinds=kinds,
                argument_indexes=(index,),
            )
            if sanitizer is not None and sanitizer.effective:
                continue
            if _sanitizer_blocks_reason(taint, kinds):
                continue
            if any(item[:3] == (graph.file_path, call.line, call.qualified) for item in seen):
                continue
            key = (graph.file_path, call.line, call.qualified, f"cross:{index}:{vuln_value}")
            if key in seen:
                continue
            seen.add(key)
            observations.append(
                _observation(
                    rule,
                    graph,
                    call,
                    matched,
                    "high",
                    taint=taint,
                    framework=framework_name or graph.framework,
                    path_kind=_path_kind(call, taint)
                    if matched.vulnerability_class is VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL
                    else None,
                    sanitizer="",
                    argument_index=index,
                    extra={
                        "taint_scope": (
                            "alias" if ext.callee_file == graph.file_path else "cross_file"
                        ),
                        "defining_file": ext.callee_file,
                        "defining_symbol": ext.callee_function,
                        "caller_file": graph.file_path,
                        "caller_symbol": _caller_symbol(call),
                        "callee_file": ext.callee_file,
                        "callee_function": ext.callee_function,
                        "callee_sink": sink_name,
                        "callee_line": str(sink_line),
                        "parameter_index": str(index),
                        "cross_file_depth": str(ext.hops),
                        "hop": str(ext.hops),
                        "cross_file_partial": "true" if ext.partial else "false",
                        "parser_complete": "false" if ext.partial else "true",
                        "relationship": ext.relationship,
                        "re_export": "true" if ext.relationship == "re_export" else "false",
                        "class_method": "true" if ext.relationship == "method" else "false",
                        "semantic_id": ext.semantic_id
                        or f"{ext.callee_file}::{ext.callee_function}",
                        "flow": (
                            f"{graph.file_path} -> {ext.callee_file}::{ext.callee_function}"
                            f" param:{index} sink:{sink_name}"
                        ),
                        **(
                            {
                                "analysis_incomplete": (
                                    taint_state.limit_reason or "bounded"
                                    if taint_state.incomplete
                                    else "partial_importer"
                                )
                            }
                            if taint_state.incomplete or ext.partial
                            else {}
                        ),
                        **({"branch_merge": "true"} if "merge:" in taint else {}),
                    },
                )
            )
    return observations


def _sql_text_wrappers(vuln: VulnerabilityClass) -> tuple[str, ...]:
    """SQL text constructors keep their input. They are not sinks by themselves."""
    if vuln is VulnerabilityClass.SQL_INJECTION:
        return ("text",)
    return ()


def _caller_symbol(call: CallSite) -> str:
    """Enclosing function or method name when the call is not at module scope."""
    scope = call.scope_id or ""
    if not scope or scope == "module":
        return ""
    return scope.rsplit(":", 1)[-1]


def _sink_occurrence(graph: SyntaxGraph, call: CallSite) -> str:
    """Ordinal among same-scope calls with the same callee and argument shape.

    Identity does not use the source line or a parser byte offset. Unrelated
    same-named calls with different argument text do not shift this index.
    Inserting another identical snippet before this call does shift it; removing
    that snippet restores the previous ordinal.
    """
    structure = _call_structure(graph, call)
    call_byte = call.span.start_byte if call.span is not None else call.line * 10_000
    earlier = 0
    for other in graph.calls:
        if other is call:
            continue
        if other.scope_id != call.scope_id:
            continue
        if _call_structure(graph, other) != structure:
            continue
        other_byte = other.span.start_byte if other.span is not None else other.line * 10_000
        if (other_byte, other.line) < (call_byte, call.line):
            earlier += 1
    return str(earlier)


def _call_structure(graph: SyntaxGraph, call: CallSite) -> str:
    """Callee plus whitespace-insensitive argument shape.

    Line numbers and parser byte offsets are not part of the shape. Inserting
    another call with the same shape earlier in the scope still changes the
    occurrence ordinal. Renaming a callee or changing an argument token does too.
    """
    pieces = [call.qualified or call.name]
    if call.arguments:
        pieces.extend(_argument_shape(argument) for argument in call.arguments)
    elif call.argument_text.strip():
        pieces.append(_compact(call.argument_text))
    elif 0 < call.line <= len(graph.lines):
        pieces.append(_compact(graph.lines[call.line - 1]))
    return "\x1f".join(pieces)


def _argument_shape(argument: object) -> str:
    text = _compact(str(getattr(argument, "text", "") or ""))
    callees = ",".join(getattr(argument, "callees", ()) or ())
    accesses = ",".join(getattr(argument, "accesses", ()) or ())
    idents = ",".join(getattr(argument, "idents", ()) or ())
    kind = "lit" if getattr(argument, "is_literal", False) else "expr"
    return "\x1e".join((kind, callees, accesses, idents, text))


def _compact(text: str) -> str:
    return "".join(text.split())


def _path_kind(call: CallSite, taint: str | None) -> PathIssueKind:
    text = call.argument_text
    if call.arguments:
        text = " ".join(a.text for a in call.arguments)
    if ".." in text:
        return PathIssueKind.UNSAFE_PATH_CONSTRUCTION
    if taint:
        return PathIssueKind.POSSIBLE_PATH_TRAVERSAL
    return PathIssueKind.USER_CONTROLLED_PATH


def _observation(
    rule: TaintFlowRule,
    graph: SyntaxGraph,
    call: CallSite,
    sink: SinkDefinition,
    confidence: str,
    *,
    taint: str,
    framework: str,
    path_kind: PathIssueKind | None,
    sanitizer: str,
    argument_index: int = 0,
    extra: dict[str, str] | None = None,
) -> SecurityObservation:
    line = call.line
    snippet = graph.lines[line - 1].strip() if 0 < line <= len(graph.lines) else call.argument_text
    title_class = sink.vulnerability_class.value.replace("_", " ")
    if sink.certainty is SinkCertainty.INDICATOR:
        title = f"Potential {title_class}"
    else:
        title = f"Potential {title_class}"
    span = call.span
    limitations = rule.documentation.limitations
    if sink.notes:
        limitations = f"{limitations} {sink.notes}"
    metadata = {
        "sink": call.qualified,
        "taint": taint,
        "parser_backend": graph.parser_backend,
        "parser_tier": str(graph.parser_tier),
        "node_id": call.node_id,
        "scope_id": call.scope_id,
        "sink_id": sink.sink_id,
        "sink_certainty": sink.certainty.value,
        "dangerous_condition": sink.dangerous_condition,
        "safe_alternatives": ",".join(sink.safe_alternatives),
        "framework": framework,
        "file_context": graph.file_context,
        "sanitizer": sanitizer,
        "taint_path": taint,
        "argument_index": str(argument_index),
        "taint_source": taint.split(":", 1)[-1] if taint else "",
        "taint_precision": "flow-sensitive",
        "path_sensitive": "false",
        "taint_use_site": "true",
        "taint_scope": "cross_file" if "cross_file:" in taint else "local",
        "branch_merge": "true" if "merge:" in taint else "false",
    }
    field_path = field_path_from_reason(taint)
    if field_path:
        metadata["field_path"] = field_path
    for route in graph.routes:
        if route.scope_id == call.scope_id and route.function:
            metadata["route"] = route.path
            metadata["route_method"] = route.method
            metadata["endpoint"] = route.function
            break
    if extra:
        metadata.update(extra)
    if str(graph.parser_tier) in {"profile_fallback", "detection_only"}:
        metadata.setdefault("analysis_incomplete", "parser_fallback")
    metadata.setdefault("sink_occurrence", _sink_occurrence(graph, call))
    metadata.setdefault("call_identity", _call_structure(graph, call))
    if span is not None:
        metadata.update(
            {
                "start_byte": str(span.start_byte),
                "end_byte": str(span.end_byte),
                "start_column": str(span.start_column),
                "end_column": str(span.end_column),
            }
        )
    if path_kind is not None:
        metadata["path_issue_kind"] = path_kind.value
        if path_kind is not PathIssueKind.DEMONSTRATED_DIRECTORY_ESCAPE:
            metadata["verification_required"] = "directory_escape_not_demonstrated"
    return SecurityObservation(
        rule_id=rule.rule_id,
        vulnerability_class=sink.vulnerability_class,
        title=title,
        summary=f"{call.qualified} may receive user-controlled data ({taint})",
        file_path=graph.file_path,
        line=line,
        evidence_text=snippet[:500],
        confidence=confidence,
        language=graph.language,
        documentation=rule.documentation,
        metadata=metadata,
        end_line=span.end_line if span is not None else line,
        start_column=span.start_column if span is not None else None,
        end_column=span.end_column if span is not None else None,
        start_byte=span.start_byte if span is not None else None,
        end_byte=span.end_byte if span is not None else None,
        node_id=call.node_id,
        parser_backend=graph.parser_backend,
        framework_context=framework,
        taint_path=taint,
        possible_false_positives=rule.documentation.false_positives,
        limitations=limitations,
        argument_index=argument_index,
        parser_tier=str(graph.parser_tier),
        scope_id=call.scope_id,
        sink_id=sink.sink_id,
        source_id=taint,
    )


SQL_DOC = RuleDocumentation(
    detects="User-controlled data or dynamically constructed strings reaching a query API.",
    evidence="Call site, taint reason, surrounding source line, parser backend.",
    limitations="Intra-procedural plus uniquely resolved same-file and same-repository calls. Flow-sensitive at the use site, not path-sensitive. Bounded interprocedural and bounded cross-file. Does not prove the query is exploitable.",
    false_positives="ORMs, query builders, and sanitized helpers can still match.",
)
CMD_DOC = RuleDocumentation(
    detects="User-controlled data reaching process/shell execution APIs.",
    evidence="Call site and taint/dynamic construction.",
    limitations="Does not model shell escaping or allow-lists unless a known sanitizer is present.",
    false_positives="Fixed command arrays with a tainted argument that is not the executable.",
)
PATH_DOC = RuleDocumentation(
    detects="User-controlled paths used in file APIs (potential path traversal).",
    evidence="Call site and taint reason. Static analysis never claims a demonstrated directory escape.",
    limitations="Does not prove the path can leave the intended root.",
    false_positives="Path joins that later enforce a root prefix.",
)
SSRF_DOC = RuleDocumentation(
    detects="User-controlled URLs passed to HTTP clients.",
    evidence="Call site and taint reason.",
    limitations="Does not check allow-lists or parser confusion.",
    false_positives="Internal URLs built from trusted config mixed with request data.",
)
XSS_DOC = RuleDocumentation(
    detects="User-controlled data written to HTML/DOM sinks.",
    evidence="Call site and taint reason.",
    limitations="Does not evaluate auto-escaping templates.",
    false_positives="Frameworks that escape by default (e.g. React text nodes).",
)
DESER_DOC = RuleDocumentation(
    detects="Deserialization of untrusted data. Indicator APIs such as JSON.parse are not treated as confirmed gadget execution.",
    evidence="Call site and taint reason.",
    limitations="JSON.parse / encoding/json are often safe; they emit potential_unsafe_deserialization.",
    false_positives="Trusted cache payloads.",
)
EVAL_DOC = RuleDocumentation(
    detects="Dynamic evaluation/execution of code.",
    evidence="Call site; constant literals are ignored.",
    limitations="Flow-sensitive at the use site, not path-sensitive. Bounded same-file and cross-file interprocedural analysis when the callee is unique. Cannot see runtime-built strings assembled outside the repository.",
    false_positives="Test helpers and debug REPL hooks.",
)
REDIRECT_DOC = RuleDocumentation(
    detects="Redirects parameterized by user input.",
    evidence="Call site and taint reason.",
    limitations="Does not validate open-redirect allow-lists.",
    false_positives="Redirects to relative paths computed from enums.",
)


def default_taint_rules() -> list[SecurityRule]:
    return [
        TaintFlowRule(
            rule_id="sec.taint.sql",
            vulnerability_class=VulnerabilityClass.SQL_INJECTION,
            documentation=SQL_DOC,
        ),
        TaintFlowRule(
            rule_id="sec.taint.command",
            vulnerability_class=VulnerabilityClass.COMMAND_INJECTION,
            documentation=CMD_DOC,
        ),
        TaintFlowRule(
            rule_id="sec.taint.path",
            vulnerability_class=VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL,
            documentation=PATH_DOC,
        ),
        TaintFlowRule(
            rule_id="sec.taint.ssrf",
            vulnerability_class=VulnerabilityClass.SSRF,
            documentation=SSRF_DOC,
        ),
        TaintFlowRule(
            rule_id="sec.taint.xss",
            vulnerability_class=VulnerabilityClass.XSS,
            documentation=XSS_DOC,
        ),
        TaintFlowRule(
            rule_id="sec.taint.deser",
            vulnerability_class=VulnerabilityClass.UNSAFE_DESERIALIZATION,
            documentation=DESER_DOC,
        ),
        TaintFlowRule(
            rule_id="sec.taint.eval",
            vulnerability_class=VulnerabilityClass.DYNAMIC_EXECUTION,
            documentation=EVAL_DOC,
        ),
        TaintFlowRule(
            rule_id="sec.taint.redirect",
            vulnerability_class=VulnerabilityClass.UNSAFE_REDIRECT,
            documentation=REDIRECT_DOC,
        ),
    ]
