"""Load finite-state linguistic machine definitions."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

from .definition import MachineDefinition, RuntimeBindings, coerce_definition
from .models import MachineSpec

_DEFAULT_EXPORTS = (
    "definition",
    "machine_definition",
    "spec",
    "machine_spec",
    "build",
    "machine",
)


def load_machine_definition(
    reference: str | Path,
    *,
    bindings: RuntimeBindings | None = None,
    binding_modules: list[str | Path] | None = None,
) -> MachineDefinition:
    """Load a machine definition from Python, YAML, module refs, or JSON."""
    text = str(reference)
    path_text, attr = _split_reference(text)
    path = Path(path_text)
    extra = bindings or RuntimeBindings()
    for module_ref in binding_modules or []:
        module_path = Path(str(module_ref))
        module = (
            _module_from_path(module_path)
            if module_path.suffix == ".py" or module_path.is_file()
            else importlib.import_module(str(module_ref))
        )
        extra.modules[module_path.stem if module_path.is_file() else module.__name__] = module
    if path.suffix == ".json" and path.is_file() and attr is None:
        return coerce_definition(MachineSpec.from_json(path), bindings=extra)
    if path.suffix in {".yaml", ".yml"} and path.is_file() and attr is None:
        definition = _load_from_yaml(path)
        return MachineDefinition(definition.spec, definition.bindings.merge(extra))
    if path.suffix == ".py" or path.is_file():
        return _load_from_module(_module_from_path(path), attr, bindings=extra)
    return _load_from_module(importlib.import_module(path_text), attr, bindings=extra)


def load_machine_spec(reference: str | Path) -> MachineSpec:
    """Load only the serializable machine spec."""
    return load_machine_definition(reference).spec


def _split_reference(reference: str) -> tuple[str, str | None]:
    path = Path(reference)
    if path.exists():
        return reference, None
    if ":" not in reference:
        return reference, None
    path_text, attr = reference.rsplit(":", 1)
    return path_text, attr


def _module_from_path(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(path)
    module_name = f"_ellements_fsm_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load machine module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_from_module(
    module: ModuleType,
    attr: str | None,
    *,
    bindings: RuntimeBindings | None = None,
) -> MachineDefinition:
    if attr is not None:
        if not hasattr(module, attr):
            raise AttributeError(f"{module.__name__!r} has no export {attr!r}")
        return coerce_definition(getattr(module, attr), bindings=bindings)
    for candidate in _DEFAULT_EXPORTS:
        if hasattr(module, candidate):
            return coerce_definition(getattr(module, candidate), bindings=bindings)
    names = ", ".join(_DEFAULT_EXPORTS)
    raise AttributeError(
        f"{module.__name__!r} must export one of {names}, or use module:export"
    )


def _load_from_yaml(path: Path) -> MachineDefinition:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load FSLM YAML specs.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML machine spec must be a mapping: {path}")
    bindings_data = data.pop("bindings", {}) or {}
    bindings = RuntimeBindings()
    imports = bindings_data.get("imports", {}) if isinstance(bindings_data, dict) else {}
    if not isinstance(imports, dict):
        raise ValueError("bindings.imports must be a mapping")
    for alias, module_ref in imports.items():
        ref_text = str(module_ref)
        module_path = (path.parent / ref_text).resolve()
        module = (
            _module_from_path(module_path)
            if module_path.suffix == ".py" and module_path.is_file()
            else importlib.import_module(ref_text)
        )
        bindings.modules[str(alias)] = module
    return MachineDefinition(MachineSpec.model_validate(data), bindings)


__all__ = ["load_machine_definition", "load_machine_spec"]
