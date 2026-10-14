"""JSON Schema generation from Python callable signatures.

A callable's parameters are introspected via :mod:`inspect` and type
hints. Pydantic ``create_model`` then produces the canonical JSON Schema
that every dialect re-emits in its own wire format.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, create_model


def resolve_callable_signature(func: Callable[..., Any]) -> inspect.Signature:
    """Resolve postponed annotations in *func*'s signature when possible."""
    signature = inspect.signature(func)
    resolved_hints = resolve_type_hints(func)
    if not resolved_hints:
        return signature
    parameters = [
        parameter.replace(annotation=resolved_hints.get(parameter.name, parameter.annotation))
        for parameter in signature.parameters.values()
    ]
    return signature.replace(
        parameters=parameters,
        return_annotation=resolved_hints.get("return", signature.return_annotation),
    )


def resolve_type_hints(func: Callable[..., Any]) -> dict[str, Any]:
    """Best-effort ``get_type_hints`` that never raises."""
    try:
        return get_type_hints(func)
    except Exception:
        return {}


def json_schema_for_callable(func: Callable[..., Any], *, tool_name: str) -> dict[str, Any]:
    """Build a JSON Schema for *func*'s keyword-bound parameters.

    Args:
        func: The callable whose signature is converted to a schema.
        tool_name: Stable tool name used to derive the Pydantic model
            name (purely cosmetic in the schema).

    Returns:
        A JSON Schema dictionary describing the parameters object.
    """
    fields: dict[str, tuple[Any, Any]] = {}
    signature = resolve_callable_signature(func)
    resolved_hints = resolve_type_hints(func)

    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        annotation = resolved_hints.get(parameter.name, parameter.annotation)
        if annotation is inspect.Signature.empty:
            annotation = Any
        default = parameter.default
        if default is inspect.Signature.empty:
            default = ...
        fields[parameter.name] = (annotation, default)

    model_name = f"{tool_name.title().replace('_', '')}Params"
    model: type[BaseModel] = create_model(model_name, **fields)  # type: ignore[call-overload]
    schema: dict[str, Any] = model.model_json_schema()
    return schema


__all__ = [
    "json_schema_for_callable",
    "resolve_callable_signature",
    "resolve_type_hints",
]
