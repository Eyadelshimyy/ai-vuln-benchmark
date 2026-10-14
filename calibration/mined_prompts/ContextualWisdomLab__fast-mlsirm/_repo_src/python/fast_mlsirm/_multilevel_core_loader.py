"""Load multilevel contextual-effects bindings from the installed Rust shared library.

The Rust extension exports both ``PyInit__core`` and
``PyInit__multilevel_core``. Python normally discovers one module per
filename, so this loader asks the standard extension loader to initialize the
secondary symbol from the already installed binary. It performs no source
rewriting, dynamic compilation, network access, or filesystem mutation.
"""

from __future__ import annotations

from importlib.machinery import ExtensionFileLoader
from importlib.util import module_from_spec, spec_from_loader
import sys
import threading
from types import ModuleType


_MODULE_NAME = "fast_mlsirm._multilevel_core"
_LOAD_LOCK = threading.Lock()


def _initialize_module() -> ModuleType:
    """Initialize and cache the secondary extension module exactly once."""
    from . import _core

    binary_path = getattr(_core, "__file__", None)
    if not binary_path:
        raise ImportError("fast_mlsirm._core does not expose an extension path")
    loader = ExtensionFileLoader(_MODULE_NAME, binary_path)
    spec = spec_from_loader(_MODULE_NAME, loader)
    if spec is None:
        raise ImportError("could not create the fast_mlsirm multilevel extension spec")
    module = module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_MODULE_NAME, None)
        raise
    return module


def multilevel_core() -> ModuleType:
    """Return the cached secondary Rust multilevel extension module.

    The cache lookup occurs under the initialization lock because
    ``_initialize_module`` must publish a temporary ``sys.modules`` entry
    before native ``exec_module`` completes.

    Raises
    ------
    ImportError
        If the installed wheel predates the dual-module multilevel entrypoint
        or the extension loader cannot initialize ``PyInit__multilevel_core``.
    """

    with _LOAD_LOCK:
        cached = sys.modules.get(_MODULE_NAME)
        if cached is not None:
            return cached
        return _initialize_module()
