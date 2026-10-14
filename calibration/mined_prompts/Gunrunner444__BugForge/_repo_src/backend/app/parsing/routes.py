"""Syntactic HTTP route facts.

Only a member call or decorator with a string-literal path is a route, and
only when the receiver is bound to a known framework constructor whose origin
is proven from import or require bindings. The decorator is not executed and
the framework is not imported at analysis time.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.domain.source import ParsedEntity, ParsedImport
from app.parsing.model import Binding

_ROUTE_METHODS = frozenset(
    {"get", "post", "put", "delete", "patch", "head", "options", "all", "route", "api_route"}
)
_PATH_PARAM = re.compile(r"<(?:[A-Za-z_]\w*:)?([A-Za-z_]\w*)>|\{([A-Za-z_]\w*)\}|:([A-Za-z_]\w*)")
_JS_LANGS = frozenset({"javascript", "typescript"})
_REQUIRE_CALLEES = frozenset({"require", "import"})


def is_route_method(name: str) -> bool:
    return name.lower() in _ROUTE_METHODS


def route_constructors_for_language(language: str) -> frozenset[str]:
    """Constructor names that create a route object for this language."""
    return frozenset(route_constructor_origins(language))


def route_constructor_origins(language: str) -> dict[str, frozenset[str]]:
    """Constructor simple-name → modules that may provide it for this language."""
    from app.analyzers.framework_registry import get_framework_registry

    origins: dict[str, set[str]] = {}
    for spec in get_framework_registry().all_specs():
        if not _language_matches(spec.language, language):
            continue
        modules = frozenset(module.strip().lower() for module in spec.constructor_modules if module)
        if not modules:
            continue
        for name in spec.constructors:
            origins.setdefault(name, set()).update(modules)
    return {name: frozenset(modules) for name, modules in origins.items()}


def constructed_route_receiver(
    bindings: Sequence[Binding],
    *,
    language: str,
    name: str,
    at_byte: int,
    scope_id: str,
    imports: Sequence[ParsedImport] = (),
    entities: Sequence[ParsedEntity] = (),
) -> bool:
    """True when ``name`` is bound to a framework constructor at this use.

    ``app = Flask(__name__)`` after ``from flask import Flask`` counts.
    ``from flask import Flask as App`` then ``app = App(__name__)`` counts.
    An unrelated object named ``app``, a local ``def FastAPI``, or an
    unimported constructor name does not.
    """
    origins = route_constructor_origins(language)
    if not origins or not name or "." in name:
        return False
    current: str | None = scope_id or "module"
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        hits = [
            binding
            for binding in bindings
            if binding.scope_id == current
            and binding.name == name
            and _binding_byte(binding) <= at_byte
        ]
        if hits:
            chosen = max(
                hits, key=lambda binding: (_binding_byte(binding), binding.definition_index)
            )
            return _callee_is_framework_constructor(
                chosen,
                origins,
                bindings=bindings,
                imports=imports,
                entities=entities,
            )
        if "/" not in current:
            break
        current = current.rsplit("/", 1)[0]
    if "module" not in seen:
        return constructed_route_receiver(
            bindings,
            language=language,
            name=name,
            at_byte=at_byte,
            scope_id="module",
            imports=imports,
            entities=entities,
        )
    return False


def looks_like_route_path(path: str) -> bool:
    """A literal path, not an arbitrary string such as a cache key."""
    return path.startswith("/")


def path_parameters(path: str) -> tuple[str, ...]:
    found: list[str] = []
    for match in _PATH_PARAM.finditer(path):
        name = next(group for group in match.groups() if group)
        if name in {"self", "cls"} or name in found:
            continue
        found.append(name)
    return tuple(found)


def _language_matches(spec_language: str, graph_language: str) -> bool:
    spec = spec_language.strip().lower()
    lang = graph_language.strip().lower()
    if not spec or not lang:
        return False
    parts = {part for part in spec.replace(",", "/").split("/") if part}
    if lang in parts:
        return True
    if lang in _JS_LANGS and parts & _JS_LANGS:
        return True
    return spec == lang


def _scope_chain(scope_id: str) -> set[str]:
    current = scope_id or "module"
    found = {current, "module"}
    while "/" in current:
        current = current.rsplit("/", 1)[0]
        found.add(current)
    return found


def _binding_byte(binding: Binding) -> int:
    start = getattr(binding.span, "start_byte", None)
    if isinstance(start, int):
        return start
    return max(0, binding.line) * 10_000


def _entity_byte(entity: ParsedEntity) -> int:
    if entity.end_byte > 0 or entity.start_byte > 0:
        return entity.start_byte
    return max(0, entity.start_line) * 10_000


def _import_byte(imp: ParsedImport) -> int:
    if imp.end_byte > 0 or imp.start_byte > 0:
        return imp.start_byte
    if imp.line_number <= 1:
        return 0
    return max(0, imp.line_number) * 10_000


def _callee_is_framework_constructor(
    binding: Binding,
    origins: dict[str, frozenset[str]],
    *,
    bindings: Sequence[Binding],
    imports: Sequence[ParsedImport],
    entities: Sequence[ParsedEntity],
) -> bool:
    at_byte = _binding_byte(binding)
    for callee in binding.rhs_callees:
        if _constructor_origin_proven(
            callee,
            origins,
            bindings=bindings,
            imports=imports,
            entities=entities,
            at_byte=at_byte,
            scope_id=binding.scope_id,
        ):
            return True
    return False


def _constructor_origin_proven(
    callee: str,
    origins: dict[str, frozenset[str]],
    *,
    bindings: Sequence[Binding],
    imports: Sequence[ParsedImport],
    entities: Sequence[ParsedEntity],
    at_byte: int,
    scope_id: str,
) -> bool:
    simple = callee.rsplit(".", 1)[-1]
    if "." in callee:
        module = callee.rsplit(".", 1)[0].split("(", 1)[0].strip()
        if simple not in origins:
            return False
        if not _module_matches(module, origins[simple]):
            return False
        return not _name_is_shadowed(
            simple,
            bindings=bindings,
            imports=imports,
            entities=entities,
            at_byte=at_byte,
            scope_id=scope_id,
        )
    origin = _latest_name_origin(
        simple,
        bindings=bindings,
        imports=imports,
        entities=entities,
        at_byte=at_byte,
        scope_id=scope_id,
    )
    if origin is None:
        return False
    kind, module, exported = origin
    if kind in {"function", "class", "assignment"}:
        return False
    constructor_name = exported if exported in origins else simple
    if constructor_name not in origins:
        return False
    return _module_matches(module, origins[constructor_name])


def _module_matches(module: str, allowed: frozenset[str]) -> bool:
    root = module.strip().strip("\"'").lower().replace("\\", "/")
    if not root:
        return False
    if "/" in root:
        root = root.rsplit("/", 1)[-1]
    root = root.split(".", 1)[0]
    return root in allowed


def _name_is_shadowed(
    name: str,
    *,
    bindings: Sequence[Binding],
    imports: Sequence[ParsedImport],
    entities: Sequence[ParsedEntity],
    at_byte: int,
    scope_id: str,
) -> bool:
    origin = _latest_name_origin(
        name,
        bindings=bindings,
        imports=imports,
        entities=entities,
        at_byte=at_byte,
        scope_id=scope_id,
    )
    return origin is not None and origin[0] in {"function", "class", "assignment"}


def _latest_name_origin(
    name: str,
    *,
    bindings: Sequence[Binding],
    imports: Sequence[ParsedImport],
    entities: Sequence[ParsedEntity],
    at_byte: int,
    scope_id: str,
) -> tuple[str, str, str] | None:
    candidates: list[tuple[int, int, tuple[str, str, str]]] = []
    for imp in imports:
        bound = _imported_local_name(imp)
        if bound != name:
            continue
        byte = _import_byte(imp)
        if byte > at_byte:
            continue
        module = imp.module or ""
        exported = imp.name or bound
        kind = (
            "require"
            if imp.syntax_kind in {"require"} or imp.import_type == "require"
            else "import"
        )
        candidates.append((byte, 0, (kind, module, exported)))
    scope_chain = _scope_chain(scope_id)
    for binding in bindings:
        if binding.name != name:
            continue
        if binding.scope_id not in scope_chain:
            continue
        byte = _binding_byte(binding)
        if byte > at_byte:
            continue
        origin = _origin_from_binding(binding, imports)
        if origin is not None:
            candidates.append((byte, binding.definition_index, origin))
    for entity in entities:
        if entity.name != name:
            continue
        if entity.parent:
            continue
        byte = _entity_byte(entity)
        if byte > at_byte:
            continue
        kind = "class" if "class" in entity.entity_type else "function"
        candidates.append((byte, 0, (kind, "", name)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def _imported_local_name(imp: ParsedImport) -> str:
    if imp.alias:
        return imp.alias
    if imp.name and imp.name != "*":
        return imp.name
    module = (imp.module or "").strip().strip("\"'")
    if not module:
        return ""
    return module.replace("\\", "/").rsplit("/", 1)[-1].split(".", 1)[0]


def _origin_from_binding(
    binding: Binding, imports: Sequence[ParsedImport]
) -> tuple[str, str, str] | None:
    rhs = binding.rhs.strip()
    callees = {callee.rsplit(".", 1)[-1] for callee in binding.rhs_callees}
    if callees & _REQUIRE_CALLEES:
        module = _require_module(rhs, binding, imports)
        if module:
            return (
                "require",
                module,
                _imported_local_name_for_module(module, imports) or binding.name,
            )
        return ("assignment", "", binding.name)
    if _looks_like_import_rhs(rhs) and not binding.rhs_callees:
        module, exported = _split_import_rhs(rhs)
        return ("import", module, exported)
    if binding.rhs_callees or binding.rhs_accesses or binding.rhs_is_literal:
        return ("assignment", "", binding.name)
    if rhs and _looks_like_import_rhs(rhs):
        module, exported = _split_import_rhs(rhs)
        return ("import", module, exported)
    if rhs:
        return ("assignment", "", binding.name)
    return None


def _looks_like_import_rhs(rhs: str) -> bool:
    token = rhs.strip()
    if not token or any(ch in token for ch in "()[]{}\"'"):
        return False
    return bool(re.fullmatch(r"[A-Za-z_][\w.]*", token))


def _split_import_rhs(rhs: str) -> tuple[str, str]:
    token = rhs.strip()
    if "." not in token:
        return token, token.rsplit(".", 1)[-1]
    module, exported = token.rsplit(".", 1)
    return module, exported


def _require_module(rhs: str, binding: Binding, imports: Sequence[ParsedImport]) -> str:
    match = re.search(r"""(?:require|import)\(\s*['"]([^'"]+)['"]""", rhs)
    if match:
        return match.group(1)
    for imp in imports:
        if imp.syntax_kind not in {"require", "import"} and imp.import_type not in {
            "require",
            "import",
        }:
            continue
        if abs(_import_byte(imp) - _binding_byte(binding)) <= 2 or imp.line_number == binding.line:
            return imp.module
    return ""


def _imported_local_name_for_module(module: str, imports: Sequence[ParsedImport]) -> str:
    for imp in imports:
        if (imp.module or "").strip().strip("\"'") == module:
            return _imported_local_name(imp)
    return ""
