"""Lightweight safe YAML parsing without loading model-provider integrations."""

from __future__ import annotations

import re
from typing import Any

import yaml

__all__ = ["load_yaml"]

try:
    from yaml import CSafeLoader as _SafeLoader
except ImportError:
    from yaml import SafeLoader as _SafeLoader

_PYTHON_SCANNER = re.compile(r"[\t!]|[|>][0-9+-]*#")
# Keep non-UTF-8 byte documents on Python's encoding-aware scanner.
_PYTHON_SCANNER_BYTES = re.compile(_PYTHON_SCANNER.pattern.encode() + rb"|[\x00\xfe\xff]")


def load_yaml(content: str | bytes) -> Any:
    """Parse one document without retaining its contents or constructed objects."""
    # Preserve Python handling of tabs, non-specific tags and block-header comments.
    python_scanner = (
        _PYTHON_SCANNER_BYTES.search(content)
        if isinstance(content, bytes)
        else _PYTHON_SCANNER.search(content)
    )
    if _SafeLoader is not yaml.SafeLoader and python_scanner is None:
        try:
            return yaml.load(content, Loader=_SafeLoader)
        except (yaml.YAMLError, UnicodeError):
            # LibYAML diagnostics differ; retain the existing Python diagnostics.
            pass
    return yaml.safe_load(content)
