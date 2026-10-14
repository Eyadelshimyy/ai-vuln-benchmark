"""Load OLS/HC regression bindings from the installed Rust shared library.

The Rust extension exports ``PyInit__regression_core`` from the same binary as
``fast_mlsirm._core``. This loader initializes that secondary module without
source rewriting, dynamic compilation, network access, or filesystem mutation.
"""

from __future__ import annotations

from importlib.machinery import ExtensionFileLoader
from importlib.util import module_from_spec, spec_from_loader
import sys
import threading
from types import ModuleType


_MODULE_NAME = "fast_mlsirm._regression_core"
_LOAD_LOCK = threading.Lock()


def _initialize_module() -> ModuleType:
    """Initialize and cache the secondary regression extension exactly once."""
    from . import _core

    binary_path = getattr(_core, "__file__", None)
    if not binary_path:
        raise ImportError("fast_mlsirm._core does not expose an extension path")
    loader = ExtensionFileLoader(_MODULE_NAME, binary_path)
    spec = spec_from_loader(_MODULE_NAME, loader)
    if spec is None:
        raise ImportError("could not create the fast_mlsirm regression extension spec")
    module = module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_MODULE_NAME, None)
        raise
    return module


def regression_core() -> ModuleType:
    """Return the cached secondary Rust regression extension module.

    Raises
    ------
    ImportError
        If the installed wheel predates ``PyInit__regression_core`` or the
        extension loader cannot initialize the secondary module.
    """

    with _LOAD_LOCK:
        cached = sys.modules.get(_MODULE_NAME)
        if cached is not None:
            return cached
        return _initialize_module()
