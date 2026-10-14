"""Load the rotation PyO3 module from the package's existing shared library.

The Rust extension exports both ``PyInit__core`` and ``PyInit__rotation_core``.
Python normally discovers one module per filename, so this loader asks the
standard extension loader to initialize the second symbol from the already
installed binary. No source rewriting, dynamic compilation, or filesystem
mutation occurs at runtime.
"""

from __future__ import annotations

from importlib.machinery import ExtensionFileLoader
from importlib.util import module_from_spec, spec_from_loader
from types import ModuleType
import sys
import threading


_MODULE_NAME = "fast_mlsirm._rotation_core"
_LOAD_LOCK = threading.Lock()


def rotation_core() -> ModuleType:
    """Return the cached secondary Rust extension module.

    Concurrent callers are serialized through initialization so no caller can
    observe the temporary ``sys.modules`` entry before ``exec_module`` has
    completed.

    Raises
    ------
    ImportError
        If the installed wheel predates the dual-module rotation entrypoint or
        the extension loader cannot initialize ``PyInit__rotation_core``.
    """

    with _LOAD_LOCK:
        cached = sys.modules.get(_MODULE_NAME)
        if cached is not None:
            return cached

        from . import _core

        binary_path = getattr(_core, "__file__", None)
        if not binary_path:
            raise ImportError("fast_mlsirm._core does not expose an extension path")
        loader = ExtensionFileLoader(_MODULE_NAME, binary_path)
        spec = spec_from_loader(_MODULE_NAME, loader)
        if spec is None:
            raise ImportError("could not create the fast_mlsirm rotation extension spec")
        module = module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        try:
            loader.exec_module(module)
        except BaseException:
            sys.modules.pop(_MODULE_NAME, None)
            raise
        return module
