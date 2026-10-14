"""
TraceBi Registry — the process-global store of a project's resources.

Holds connectors, data models, report factories, and pipeline
runners. Populated once at startup (by an app module, by the project-root
``models/`` and ``pipelines/`` directories, or from a notebook), then read
at request time.

This lives in the library, not the web layer: the FastAPI app is one
consumer, but so are the CLI, request scripts, notebooks, and agent
tooling. ``tracebi.web.api.registry`` re-exports it for backward compatibility.

Usage in your app module::

    from tracebi.registry import registry
    from tracebi import DataModel, CSVConnector
    from tracebi.reports import Report, TableSection

    connector = CSVConnector("sales", directory="data/")
    model = DataModel("SalesModel")
    model.add_connector(connector)
    model.add_table("orders", connector="sales", source="orders.csv")
    model.connect()

    registry.add_connector(connector)
    registry.add_model(model)

    @registry.report("orders_summary", description="All orders by region")
    def orders_summary():
        ds = model.load("orders")
        return Report("Orders Summary").table(ds, title="Orders")
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional


class Registry:
    """
    Central registry for TraceBi web resources.

    Holds connectors, data models, report factories, and pipeline runners.
    Consumers (the web API, the CLI, notebooks, agent tooling) only talk to
    this — they never construct resources themselves, which keeps the seam
    between the framework and an application sharp.

    Stored objects are duck-typed: this module deliberately imports nothing
    from TraceBi, so it stays usable on the base install.

    All mutators and compound reads are guarded by an RLock so the
    registry is safe under threaded servers and dev-mode reloads.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._connectors: dict[str, Any] = {}
        self._models: dict[str, Any] = {}
        self._report_factories: dict[str, dict] = {}
        self._scheduled_factories: dict[str, dict] = {}
        self._pipelines: dict[str, Any] = {}
        self._default_model_name: Optional[str] = None

    # ── Connectors ─────────────────────────────────────────────

    def add_connector(self, connector) -> "Registry":
        """Register a BaseConnector by its name."""
        with self._lock:
            self._connectors[connector.name] = connector
        return self

    def _all_connectors(self) -> tuple[dict, dict]:
        """Every connector the project uses, by name, and which models use each.

        A connector is usually built inside a model file, not registered, so
        listing only the registry left the Sources page empty for a project
        whose data is in plain sight in its models.
        """
        with self._lock:
            registered = dict(self._connectors)
            models = list(self._models.values())
        found, used_by = dict(registered), {}
        for m in models:
            for c in m.connectors():
                found.setdefault(c.name, c)
                used_by.setdefault(c.name, [])
                if m.name not in used_by[c.name]:
                    used_by[c.name].append(m.name)
        return found, used_by

    def get_connector(self, name: str):
        return self._all_connectors()[0].get(name)

    def list_connectors(self) -> list[dict]:
        found, used_by = self._all_connectors()
        return [
            {**c.describe(), "storage": c.storage(), "used_by": used_by.get(name, [])}
            for name, c in found.items()
        ]

    # ── Models ─────────────────────────────────────────────────

    def add_model(self, model, default: bool = False) -> "Registry":
        """
        Register a DataModel by its name.

        Pass ``default=True`` to mark this model as the project default —
        request scripts can grab it via ``registry.get_default_model()``
        instead of building their own.
        """
        with self._lock:
            self._models[model.name] = model
            if default or self._default_model_name is None:
                self._default_model_name = model.name
        return self

    def set_default_model(self, name: str) -> "Registry":
        """Mark a previously-registered model as the project default."""
        with self._lock:
            if name not in self._models:
                raise KeyError(f"Model '{name}' is not registered.")
            self._default_model_name = name
        return self

    def get_default_model(self):
        """Return the project default DataModel, or None if none is set."""
        with self._lock:
            if self._default_model_name is None:
                return None
            return self._models.get(self._default_model_name)

    def get_model(self, name: str):
        with self._lock:
            return self._models.get(name)

    def release_all(self) -> None:
        """Release every registered model's open handles (a warehouse file held
        open read-only blocks a transform in another process from writing it);
        the next query reconnects."""
        with self._lock:
            models = list(self._models.values())
        for model in models:
            model.disconnect()

    def list_models(self) -> list[dict]:
        with self._lock:
            models = list(self._models.values())
        out = []
        for m in models:
            info = m.info()
            out.append({
                "name": info["name"],
                "connectors": info["connectors"],
                "tables": [t["name"] for t in info["tables"]],
                "relationships": [r["name"] for r in info["relationships"]],
                # The star schema's shape, so a list can say "1 fact · 2
                # dimensions · 9 measures" instead of a relationship count a
                # facts-and-dimensions model never has.
                "facts": [f["name"] for f in info["facts"]],
                "dimensions": [d["name"] for d in info["dimensions"]],
                "measures": [m["name"] for m in info["measures"]],
            })
        return out

    def describe_model(self, name: str) -> Optional[dict]:
        with self._lock:
            m = self._models.get(name)
        if not m:
            return None
        return m.info()

    # ── Reports ────────────────────────────────────────────────

    def add_report(
        self,
        name: str,
        factory: Callable,
        description: str = "",
    ) -> "Registry":
        """Register a report factory (a zero-arg callable that returns a Report)."""
        with self._lock:
            self._report_factories[name] = {
                "description": description,
                "factory": factory,
            }
        return self

    def remove_report(self, name: str) -> bool:
        """Forget a report. True when it was registered.

        Used by live discovery when a report's package or spec is deleted
        from disk, so the Reports page stops listing a report that can no
        longer be built.
        """
        with self._lock:
            return self._report_factories.pop(name, None) is not None

    def report(self, name: str, description: str = ""):
        """Decorator for report factory functions."""
        def decorator(fn: Callable) -> Callable:
            self.add_report(name, fn, description)
            return fn
        return decorator

    def list_reports(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "name": k,
                    "description": v["description"],
                    # "artifact": discovery backed this report with a template
                    # package (an authored package, or a spec the compiler
                    # turned into one), so it renders a self-contained schema-2
                    # artifact whose figures are fingerprinted and re-checkable
                    # with ``verify --file``. "carrier": a plain code factory
                    # with no package — rendered, but not a verifiable artifact.
                    "kind": "artifact"
                    if getattr(v["factory"], "_tracebi_package_dir", None)
                    else "carrier",
                    # How it is authored: "spec" (a reports/<name>.json in the
                    # default style), "package" (reports/<name>/ with its own
                    # template and style), or "code" (a registered factory).
                    "form": (getattr(v["factory"], "_tracebi_source", None)
                             or {}).get("form", "code"),
                }
                for k, v in self._report_factories.items()
            ]

    def run_report(self, name: str):
        """Call the report factory and return the Report object."""
        with self._lock:
            entry = self._report_factories.get(name)
        if not entry:
            return None
        return entry["factory"]()

    def report_factory(self, name: str) -> Optional[Callable]:
        """The registered factory itself, un-called.

        The public accessor consumers use to *inspect* a report's backing —
        e.g. the web layer detecting an artifact-backed report (discovery
        tags the factory with its package directory) so it can serve the
        real artifact render instead of the carrier — without reaching into
        ``_report_factories``.
        """
        with self._lock:
            entry = self._report_factories.get(name)
        return entry["factory"] if entry else None

    def report_package_dir(self, name: str) -> Optional[str]:
        """The template-package directory backing *name*, or ``None``.

        Discovery tags an artifact-backed report's factory with its package
        directory; this exposes it so the web layer can serve the real
        artifact render without reaching into the tagged attribute itself.
        """
        factory = self.report_factory(name)
        return getattr(factory, "_tracebi_package_dir", None) if factory else None

    def report_source(self, name: str) -> Optional[dict]:
        """Where *name* is defined: ``{"form", "path", "extras"}``, or ``None``.

        Discovery tags a spec's or package's factory with its source location
        so the web layer can show the files behind a report.
        """
        factory = self.report_factory(name)
        return getattr(factory, "_tracebi_source", None) if factory else None

    # ── Scheduled reports ──────────────────────────────────────

    def scheduled(
        self,
        name: str,
        cron: str,
        description: str = "",
    ):
        """
        Decorator: register a report factory and record a cron string.

        Deprecated. Nothing reads the cron string, so the decorator never
        ran a report. Put a ``"schedule"`` block in the package's
        ``report.json`` and use ``tracebi schedule``. The report is still
        registered, exactly as before.
        """
        import warnings
        warnings.warn(
            "registry.scheduled never ran anything. Put a \"schedule\" "
            "block in the package's report.json and use `tracebi schedule`. "
            "See `tracebi schedule --help`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._scheduled(name, cron, description)

    def _scheduled(self, name: str, cron: str, description: str = ""):
        """Register the factory and cron string without warning.

        ``scheduled`` warns, then calls this. The notebook facade warns
        once and calls this too, so ``@register.scheduled`` does not warn
        a second time.
        """

        def decorator(fn: Callable) -> Callable:
            self.add_report(name, fn, description)
            with self._lock:
                self._scheduled_factories[name] = {
                    "cron":        cron,
                    "description": description,
                    "factory":     fn,
                }
            return fn
        return decorator

    def list_scheduled(self) -> list[dict]:
        with self._lock:
            return [
                {"name": k, "cron": v["cron"], "description": v["description"]}
                for k, v in self._scheduled_factories.items()
            ]

    # ── Pipelines ──────────────────────────────────────────────

    def add_pipeline(self, name: str, runner) -> "Registry":
        """Register a PipelineRunner under a logical name."""
        with self._lock:
            self._pipelines[name] = runner
        return self

    def get_pipeline(self, name: str):
        with self._lock:
            return self._pipelines.get(name)

    def list_pipeline_names(self) -> list[str]:
        with self._lock:
            return list(self._pipelines.keys())


registry = Registry()
