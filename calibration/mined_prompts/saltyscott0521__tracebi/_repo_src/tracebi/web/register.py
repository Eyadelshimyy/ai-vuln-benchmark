"""
``tracebi.register`` — the friendly facade over the project registry.

Lets notebook and script authors register resources without learning the
registry API. Lookups check the registry first and fall back to the
project-root ``models/`` and ``pipelines/`` directories, so the same code
works with or without a running server.

Also importable as ``tracebi.web.register`` — the original spelling, kept
because it appears in every scaffold and doc. The registry is no longer
part of the web layer, so ``tracebi.register`` is the accurate name.
"""

from __future__ import annotations

from typing import Callable, Optional


def _registry():
    from tracebi.registry import registry as _r
    return _r


class _Register:
    """
    Callable namespace exposed as ``tracebi.web.register``.

    Methods mirror the registry singleton so notebook authors can register
    resources without learning the FastAPI layout.
    """

    def connector(self, connector) -> "_Register":
        _registry().add_connector(connector)
        return self

    def model(self, model, default: bool = False) -> "_Register":
        _registry().add_model(model, default=default)
        return self

    def set_default_model(self, name: str) -> "_Register":
        _registry().set_default_model(name)
        return self

    def get_default_model(self):
        """Return the default DataModel — registry first, then models/ on disk."""
        found = _registry().get_default_model()
        if found is not None:
            return found
        from tracebi.model_registry import get_default_model as _get
        return _get()

    def get_model(self, name: str):
        """Return a model by name — registry first, then models/ on disk."""
        found = _registry().get_model(name)
        if found is not None:
            return found
        from tracebi.model_registry import get_model as _get
        return _get(name)

    def pipeline(self, name: str, runner) -> "_Register":
        _registry().add_pipeline(name, runner)
        return self

    def get_runner(self, name: str):
        """Return a runner by name — registry first, then pipelines/ on disk."""
        found = _registry().get_pipeline(name)
        if found is not None:
            return found
        from tracebi.pipeline_registry import get_runner as _get
        return _get(name)

    def report(self, name: str, description: str = "") -> Callable:
        """Decorator: register a zero-arg report factory."""
        return _registry().report(name, description=description)

    def scheduled(
        self,
        name: str,
        cron: str,
        description: str = "",
    ) -> Callable:
        """Decorator: register a report factory tagged with a cron schedule.

        Deprecated. The cron string is never read. Put a ``"schedule"``
        block in the package's ``report.json`` and use ``tracebi schedule``.
        The report is still registered.
        """
        import warnings
        warnings.warn(
            "register.scheduled never ran anything. Put a \"schedule\" "
            "block in the package's report.json and use `tracebi schedule`. "
            "See `tracebi schedule --help`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _registry()._scheduled(name, cron, description)

    def add_report(
        self,
        name: str,
        factory: Callable,
        description: str = "",
    ) -> "_Register":
        _registry().add_report(name, factory, description=description)
        return self

    def auto_discover(self, path: str, package: Optional[str] = None) -> list[str]:
        """Discover and import request modules in *path*."""
        from tracebi.web.discovery import auto_discover
        return auto_discover(path, package=package)


register = _Register()
