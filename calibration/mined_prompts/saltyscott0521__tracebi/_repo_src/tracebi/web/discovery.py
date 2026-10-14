"""
Folder-based auto-discovery for request and scheduled scripts.

Two entry points:

* :func:`auto_discover` — register every report in a directory: ``*.json``
  specs and ``<name>/`` packages, in that directory and in any folder below
  it, plus import every top-level ``*.py`` / ``*.ipynb`` file (skips
  ``_*``). A report in a folder is addressed by its path, e.g.
  ``finance/weekly_summary``, so two folders can each hold one of the same
  name. Decorators inside imported modules (``@registry.report``,
  ``@registry.scheduled``) fire as a side effect of import. Notebook code
  cells are concatenated into a script first; line magics and shell escapes
  are silently dropped.

* :func:`reload_modules` — re-import the modules previously discovered
  via :func:`auto_discover`. Used by the optional dev-mode reload endpoint
  to pick up edits without restarting the server.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Optional

from tracebi.model.model_spec import (
    SPEC_SUFFIXES,
    ModelSpecError,
    read_model_doc,
    validate_model_spec,
)


# Track everything we've imported so we can reload it later.
_discovered: dict[str, str] = {}  # module_name -> file path

# Reports discovery registered from a package directory or a spec file:
# report name -> that source path. Live discovery (:func:`rescan`) compares
# this with what is on disk now.
_live_reports: dict[str, str] = {}
# A source that failed to register, with the file times it failed at, so a
# broken package is retried when its files change, not on every scan.
_failed_sources: dict[str, tuple] = {}
# Model and pipeline files live discovery has already put in the web registry.
_live_models: set[str] = set()
_live_pipelines: set[str] = set()
# Connection files already put in the web registry: stem -> (path, mtime).
_live_connections: dict[str, tuple] = {}


# Per-file outcome of every discovery attempt, in order. Discovery is
# convention-based and quiet by nature: a file in the wrong place, or one
# that raises on import, simply never appears. Recording what happened is
# the difference between "why isn't my report showing up?" and an answer.
_outcomes: list[dict] = []


def discovery_report() -> list[dict]:
    """
    What happened to every file discovery looked at.

    Each entry is ``{directory, file, module, status, reason}`` where status
    is ``registered``, ``skipped``, or ``failed``. Consumed by
    ``tracebi validate``, ``GET /api/discovery``, and agent tooling.
    """
    return [dict(o) for o in _outcomes]


def clear_discovery_report() -> None:
    """Forget recorded outcomes (used between test runs and dev reloads)."""
    _outcomes.clear()


def _register_spec_file(full_path: str, stem: str) -> dict:
    """
    Register a ``reports/<name>.json`` report spec.

    The factory resolves models and builds the report at *call* time, not at
    discovery time. Two reasons. Discovery runs at server startup, where doing
    real query work would be the same mistake the demo pipeline made — and a
    model the spec names may be registered after this file is scanned, so
    binding eagerly would make discovery order significant.

    Validation is deliberately structural only: the schema, the section types,
    the shape of each data reference. Checking a spec *against* its models
    needs the models, so that is left to ``tracebi spec validate`` and to the
    first run, which reports a real error rather than a startup failure.
    """
    from tracebi.registry import registry
    from tracebi.spec import ReportSpec

    try:
        spec = ReportSpec.from_json(open(full_path, encoding="utf-8").read())
    except Exception as exc:  # noqa: BLE001 — one bad file must not stop startup
        return {"status": "failed", "module": stem,
                "reason": f"{type(exc).__name__}: {exc}"}

    problems = [p for p in spec.validate().get("errors", [])]
    if problems:
        return {"status": "failed", "module": stem,
                "reason": "; ".join(str(p) for p in problems[:3])}

    # Compile the spec to the artifact package ONCE, here at discovery — this is
    # structural (no models, no queries), the same as the validation above — and
    # serve it through the artifact path (via the _tracebi_package_dir tag). So a
    # reports/<name>.json spec on the Reports page gets figures, the receipt
    # drawer, badges, and a schema-2 manifest, exactly like /api/spec/render and
    # a hand-authored package — not the legacy carrier render. Model resolution
    # still happens at CALL time, inside TemplatePackage.render.
    import os
    import tempfile

    from tracebi.reports.compile_spec import compile_spec

    def _sibling(name: str) -> str:
        if not name:
            return ""
        p = os.path.join(os.path.dirname(full_path), name)
        return open(p, encoding="utf-8").read() if os.path.isfile(p) else ""

    try:
        compiled = compile_spec(
            spec,
            theme_css=_sibling(getattr(spec, "theme", "") or ""),
            script_js=_sibling(getattr(spec, "script", "") or ""),
        )
        # A spec in a folder is named "finance/weekly"; a "/" in a temp
        # directory's prefix would point inside a folder that doesn't exist.
        pkg_dir = tempfile.mkdtemp(prefix=f"tracebi-spec-{stem.replace('/', '--')}-")
        for fname, content in compiled.files.items():
            with open(os.path.join(pkg_dir, fname), "w", encoding="utf-8") as fh:
                fh.write(content)
    except Exception as exc:  # noqa: BLE001 — one bad file must not stop startup
        return {"status": "failed", "module": stem,
                "reason": f"spec compile failed: {type(exc).__name__}: {exc}"}

    def factory(_spec=spec):
        from tracebi.model_registry import get_model, list_models
        models = {}
        for name in list_models():
            try:
                m = get_model(name)
            except Exception:  # noqa: BLE001 — a broken model is that model's problem
                continue
            # A DataRef may name either the file stem (sales_model) or the
            # model's own name (SalesModel); accept both so an author is not
            # made to care which one they saw.
            models[name] = m
            models[getattr(m, "name", name)] = m
        return _spec.build(models=models)

    # Tag the factory with the compiled package dir so the web layer serves the
    # real artifact render (embedded data, figure claims, schema-2 manifest)
    # instead of the carrier — the same mechanism a hand-authored package uses.
    factory._tracebi_package_dir = pkg_dir
    # Where the report is defined, for the Reports page's Source view: the
    # spec file itself (the compiled package above is a temp dir, not source).
    factory._tracebi_source = {
        "form": "spec",
        "path": full_path,
        "extras": [n for n in (getattr(spec, "theme", ""), getattr(spec, "script", "")) if n],
    }

    registry.add_report(stem, factory, spec.description or "")
    return {"status": "registered", "module": stem}


def _register_template_package(dir_path: str, stem: str) -> dict:
    """
    Register a freeform report package — ``reports/<name>/`` holding
    ``report.json`` + ``template.html`` (architecture §7, lane B).

    The same contract as :func:`_register_spec_file`: a zero-arg factory that
    resolves models at *call* time, not at discovery time, and flows through
    ``registry.add_report`` unchanged. A package is *a name and a zero-arg
    callable*, exactly like every other report — so it lists on the Reports
    page and in ``registry.list_reports`` beside specs and code factories.

    Loading validates only the *structure* of the package and its data
    bindings (``TemplatePackage.__init__`` reuses ``DataRef`` parsing and
    touches no model). Resolving a binding against its model — and the freeform
    render of ``template.html`` — happens in ``tracebi report build`` and at
    factory-call time, so a model registered after this scan still resolves.
    The factory returns the package's carrier :class:`Report` (the synthetic
    sections whose fingerprints back the receipt); the self-contained freeform
    ``.html`` is produced by the build step, not the web report-run path.
    """
    from tracebi.registry import registry
    from tracebi.reports.template_package import TemplatePackage

    try:
        pkg = TemplatePackage(dir_path)
    except Exception as exc:  # noqa: BLE001 — one bad package must not stop startup
        return {"status": "failed", "module": stem,
                "reason": f"{type(exc).__name__}: {exc}"}

    def factory(_pkg=pkg):
        from tracebi.model_registry import get_model, list_models
        models = {}
        for name in list_models():
            try:
                m = get_model(name)
            except Exception:  # noqa: BLE001 — a broken model is that model's problem
                continue
            # Accept either the file stem or the model's own name, as the spec
            # factory does — an author should not have to care which they saw.
            models[name] = m
            models[getattr(m, "name", name)] = m
        report, _ = _pkg.build(models)
        return report

    # Tag the factory with its package directory so the web layer can detect
    # an artifact-backed report (via registry.report_factory) and serve the
    # REAL artifact render — embedded data, figure claims, schema-2 manifest
    # — instead of the carrier (architecture v2 §2.3, web parity). The
    # factory contract itself is unchanged: a name and a zero-arg callable.
    factory._tracebi_package_dir = dir_path
    factory._tracebi_source = {"form": "package", "path": dir_path, "extras": []}

    registry.add_report(stem, factory, pkg.description or "")
    return {"status": "registered", "module": stem}


def auto_discover(
    path: str,
    package: Optional[str] = None,
    strict: bool = False,
    prefix: str = "",
) -> list[str]:
    """
    Register every report under *path* and import its top-level ``*.py`` /
    ``*.ipynb`` files (skips ``_*`` and hidden entries).

    A subdirectory holding ``report.json`` + ``template.html`` is a report
    package. Any other subdirectory is a folder: it is scanned the same way,
    and what it holds is registered under ``<folder>/<name>``. Code modules
    load only from the top level, where the app module convention expects
    them.

    Args:
        path:    Directory to scan. Relative paths are resolved against the
                 current working directory.
        package: Optional package name to register the modules under. When
                 omitted, modules are imported under synthetic names
                 ``tracebi_request_<filename>`` so they do not collide with
                 anything on ``sys.path``.
        strict:  Re-raise the first import error instead of recording it.
                 Off by default so one broken file cannot stop a server
                 from starting — the failure is recorded in
                 :func:`discovery_report` and surfaced by
                 ``tracebi validate``.
        prefix:  Front of every report name registered here (``""`` at a
                 single reports root; ``"finance/"`` for a library mount
                 labeled ``finance``).

    Returns:
        List of successfully imported module names.
    """
    if not os.path.isdir(path):
        _outcomes.append({
            "directory": path, "file": None, "module": None,
            "status": "skipped", "reason": "directory does not exist",
        })
        return []

    return _scan(path, prefix, package, strict)


def discover_library(
    package: Optional[str] = None,
    strict: bool = False,
) -> list[str]:
    """Register reports from every configured library root.

    Uses :func:`tracebi.report_paths.library_roots` so
    ``TRACEBI_LIBRARY_MOUNTS`` and ``TRACEBI_REPORTS_DIR`` share one path.
    """
    from tracebi.report_paths import library_roots

    found: list[str] = []
    for label, root in library_roots():
        pfx = f"{label}/" if label else ""
        found.extend(auto_discover(str(root), package=package, strict=strict,
                                   prefix=pfx))
    return found


def _scan(path: str, prefix: str, package: Optional[str], strict: bool) -> list[str]:
    """One directory of :func:`auto_discover`. *prefix* is the folder path
    below the reports root (``""`` at the top, ``"finance/"`` one level down),
    and becomes the front of every report name registered here."""
    discovered: list[str] = []
    entries = sorted(os.listdir(path))
    # The shadowing rule (architecture v2 §7): an artifact package directory
    # shadows a same-named .json spec. `tracebi migrate spec` emits the
    # package alongside the spec on purpose — the moment the directory
    # exists it is the report; deleting it rolls back to the spec.
    package_stems = {
        e for e in entries
        if not e.startswith("_")
        and os.path.isdir(os.path.join(path, e))
        and os.path.isfile(os.path.join(path, e, "report.json"))
        and os.path.isfile(os.path.join(path, e, "template.html"))
    }
    for entry in entries:
        full = os.path.join(path, entry)
        record = {"directory": path, "file": entry, "module": None}

        if entry.startswith("_"):
            _outcomes.append({**record, "status": "skipped",
                              "reason": "name starts with '_'"})
            continue
        if entry.startswith("."):
            continue                      # .git, .ipynb_checkpoints, .DS_Store
        is_py = entry.endswith(".py")
        is_nb = entry.endswith(".ipynb")
        is_spec = entry.endswith(".json")

        if is_spec and entry[: -len(".json")] in package_stems:
            stem = entry[: -len(".json")]
            print(
                f"[tracebi] {path}: artifact package '{stem}/' shadows spec "
                f"'{entry}' — the artifact is served; delete the directory "
                f"to roll back to the spec.",
                file=sys.stderr,
            )
            _outcomes.append({**record, "status": "skipped",
                              "reason": f"shadowed by artifact package '{stem}/'"})
            continue

        if is_spec:
            # A report as data rather than as code. Registered without
            # importing anything, which is the point: a spec is a bounded
            # document that can be checked before it runs, where a .py file
            # is arbitrary code that has already run by the time you see it.
            outcome = _register_spec_file(full, stem=prefix + entry[: -len(".json")])
            _outcomes.append({**record, **outcome})
            if outcome["status"] == "registered":
                discovered.append(outcome["module"])
                _live_reports[outcome["module"]] = full
            continue

        if not (is_py or is_nb):
            if not os.path.isdir(full):
                _outcomes.append({**record, "status": "skipped",
                                  "reason": "not a .py, .ipynb or .json file"})
            elif (os.path.isfile(os.path.join(full, "report.json"))
                  and os.path.isfile(os.path.join(full, "template.html"))):
                # A freeform report package: report.json + template.html in a
                # subdirectory. Registered like a spec file; other
                # subdirectories are still skipped below.
                outcome = _register_template_package(full, stem=prefix + entry)
                _outcomes.append({**record, **outcome})
                if outcome["status"] == "registered":
                    discovered.append(outcome["module"])
                    _live_reports[outcome["module"]] = full
            else:
                # A folder of reports. It registers nothing itself; its
                # contents are registered under "<folder>/<name>".
                _outcomes.append({**record, "status": "skipped",
                                  "reason": f"a folder: reports inside are "
                                            f"named '{prefix}{entry}/<name>'"})
                discovered.extend(_scan(full, f"{prefix}{entry}/", package, strict))
            continue

        if prefix:
            _outcomes.append({**record, "status": "skipped",
                              "reason": "code modules load only from the top "
                                        "of the reports folder"})
            continue

        stem = entry[: -len(".ipynb") if is_nb else -3]
        mod_name = f"{package}.{stem}" if package else f"tracebi_request_{stem}"
        record["module"] = mod_name
        from tracebi.registry import registry as _registry
        before = {r["name"] for r in _registry.list_reports()}

        try:
            if is_nb:
                from tracebi._notebook import notebook_to_source
                source = notebook_to_source(full)
                code = compile(source, full, "exec")
                module = type(sys)("tracebi_request_nb_" + stem)
                module.__file__ = full
                sys.modules[mod_name] = module
                exec(code, module.__dict__)  # noqa: S102
            else:
                spec = importlib.util.spec_from_file_location(mod_name, full)
                if spec is None or spec.loader is None:
                    _outcomes.append({**record, "status": "failed",
                                      "reason": "could not build an import spec"})
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module
                spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 — recorded, and re-raised if strict
            sys.modules.pop(mod_name, None)
            _outcomes.append({
                **record, "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            if strict:
                raise
            continue

        discovered.append(mod_name)
        _discovered[mod_name] = full
        # A code module still loads (its side effects are its own), but a
        # report it registers with no package behind it cannot render: the
        # web layer refuses it. Say so here, where the author looks, rather
        # than on the first click.
        orphans = sorted(
            r["name"] for r in _registry.list_reports()
            if r["name"] not in before and not _registry.report_package_dir(r["name"]))
        warning = None
        if orphans:
            warning = (
                f"registers {', '.join(repr(n) for n in orphans)} with no report "
                f"package, so {'it' if len(orphans) == 1 else 'they'} will not "
                f"render. Make each a package: tracebi new-report \"<name>\". "
                f"Code modules also load only at startup; packages and specs "
                f"are picked up while the server runs.")
            print(f"[tracebi] {full}: {warning}", file=sys.stderr)
        _outcomes.append({**record, "status": "registered", "reason": None,
                          **({"warning": warning} if warning else {})})
    return discovered


# ── Live discovery ──────────────────────────────────────────────────────────
#
# A running server picks up report packages, specs, model files and
# pipeline files added to (or reports removed from) the project without a
# restart. A background thread calls
# rescan() every TRACEBI_DISCOVERY_INTERVAL seconds; route handlers never
# touch the registry. Python report modules stay startup-only: re-importing
# code on a timer would re-run its side effects.


def _is_package(path: str) -> bool:
    return (os.path.isfile(os.path.join(path, "report.json"))
            and os.path.isfile(os.path.join(path, "template.html")))


def _report_sources(path: str, prefix: str = "") -> dict[str, str]:
    """Every package and spec under *path*, by report name, with the same
    naming, folder and shadowing rules as :func:`auto_discover`."""
    found: dict[str, str] = {}
    try:
        entries = sorted(os.listdir(path))
    except OSError:
        return found
    for entry in entries:
        if entry.startswith(("_", ".")):
            continue
        full = os.path.join(path, entry)
        if os.path.isdir(full):
            if _is_package(full):
                found[prefix + entry] = full
            else:
                found.update(_report_sources(full, f"{prefix}{entry}/"))
        elif entry.endswith(".json"):
            stem = entry[: -len(".json")]
            if not _is_package(os.path.join(path, stem)):
                found[prefix + stem] = full
    return found


def _file_times(source: str) -> tuple:
    names = (["report.json", "template.html"] if os.path.isdir(source) else [""])
    out = []
    for name in names:
        try:
            out.append(os.stat(os.path.join(source, name) if name else source)
                       .st_mtime_ns)
        except OSError:
            out.append(None)
    return tuple(out)


def _record(name: str, source: str, outcome: dict) -> None:
    """Replace this report's earlier outcome, so the report stays one line
    per file however many scans have run."""
    _outcomes[:] = [o for o in _outcomes if o.get("module") != name]
    _outcomes.append({"directory": os.path.dirname(source),
                      "file": os.path.basename(source), **outcome})


def register_models(models_dir: str) -> list[str]:
    """Record new ``models/*.py`` files and put each in the web registry.

    Returns the stems added this call. A model that fails to load is left
    out and retried on the next call; the model registry reloads a file
    whose contents change on its next use.
    """
    from tracebi import model_registry
    from tracebi.registry import registry

    added: list[str] = []
    if not os.path.isdir(models_dir):
        return added
    stems = model_registry.auto_discover(models_dir)
    # A refusal reported earlier is rebuilt from the current files, so a
    # deleted twin stops being reported.
    _outcomes[:] = [o for o in _outcomes
                    if not (str(o.get("module", "")).startswith("models/")
                            and " refused: " in str(o.get("reason", "")))]
    for path, reason in model_registry.clashes().items():
        if os.path.abspath(os.path.dirname(path)) == os.path.abspath(models_dir):
            name = f"models/{os.path.basename(path)}"
            _record(name, path, {"module": name, "status": "failed",
                                 "reason": reason})
    for stem in stems:
        if stem in _live_models:
            continue
        path = model_registry.model_path(stem)
        if path and path.endswith(SPEC_SUFFIXES):
            # Structural check only: the file is read, never the warehouse.
            try:
                errors = validate_model_spec(read_model_doc(path))
            except ModelSpecError as exc:
                errors = exc.errors
            except (OSError, ValueError) as exc:
                errors = [f"cannot be read ({exc})"]
            if errors:
                _record(f"models/{stem}", path, {
                    "module": f"models/{stem}", "status": "failed",
                    "reason": "; ".join(errors)})
                print(f"[tracebi] model '{stem}' failed to load: "
                      f"{'; '.join(errors)}", file=sys.stderr)
                continue
        try:
            model = model_registry.get_model(stem)
        except Exception as exc:  # noqa: BLE001 — one broken model must not stop the rest
            if path and path.endswith(SPEC_SUFFIXES):
                _record(f"models/{stem}", path, {
                    "module": f"models/{stem}", "status": "failed", "reason": str(exc)})
            print(f"[tracebi] model '{stem}' failed to load: {exc}", file=sys.stderr)
            continue
        if path and path.endswith(SPEC_SUFFIXES):
            _outcomes[:] = [o for o in _outcomes
                            if o.get("module") != f"models/{stem}"]
        if getattr(model, "name", stem) not in [m["name"] for m in registry.list_models()]:
            registry.add_model(model)
        _live_models.add(stem)
        added.append(stem)
    return added


def register_connections(connections_dir: str) -> list[str]:
    """Put each ``connections/<name>.yaml`` in the web registry, so the Sources
    page lists it. Nothing is resolved or opened: a missing environment
    variable surfaces when the connection is used, not here. A file that is
    new or changed since the last call is (re)registered; one that does not
    validate is reported and left out. Returns the names added this call."""
    from tracebi.connections import SUFFIXES, load_connection
    from tracebi.registry import registry

    added: list[str] = []
    if not os.path.isdir(connections_dir):
        return added
    entries = sorted(e for e in os.listdir(connections_dir)
                     if not e.startswith(("_", ".")) and e.endswith(SUFFIXES))
    stems = [os.path.splitext(e)[0] for e in entries]
    for entry, stem in zip(entries, stems):
        full = os.path.join(connections_dir, entry)
        name = f"connections/{entry}"
        if stems.count(stem) > 1:
            _record(name, full, {"module": name, "status": "failed",
                                 "reason": f"{stem}.yaml and {stem}.yml both exist; "
                                           f"a connection is one file, so delete one"})
            continue
        try:
            stamp = (full, os.stat(full).st_mtime_ns)
        except OSError:
            continue
        if _live_connections.get(stem) == stamp:
            continue
        try:
            connector = load_connection(full, root=os.getcwd())
        except Exception as exc:  # noqa: BLE001 — one broken file must not stop the rest
            reason = "; ".join(getattr(exc, "errors", None) or [str(exc)])
            _record(name, full, {"module": name, "status": "failed", "reason": reason})
            print(f"[tracebi] connection '{stem}' failed to load: {reason}", file=sys.stderr)
            _live_connections[stem] = stamp      # retried when the file changes
            continue
        registry.add_connector(connector)
        _outcomes[:] = [o for o in _outcomes if o.get("module") != name]
        _live_connections[stem] = stamp
        added.append(stem)
    return added


def register_pipelines(pipelines_dir: str) -> list[str]:
    """Record new ``pipelines/*.py`` and ``pipelines/*.yaml`` files and put
    each runner in the web registry. Returns the names added this call; a file
    that fails to load is left out and retried on the next call."""
    from tracebi import pipeline_registry
    from tracebi.registry import registry

    added: list[str] = []
    if not os.path.isdir(pipelines_dir):
        return added
    stems = pipeline_registry.auto_discover(pipelines_dir)
    # A refusal reported earlier is rebuilt from the current files, so a
    # deleted twin stops being reported.
    _outcomes[:] = [o for o in _outcomes
                    if not (str(o.get("module", "")).startswith("pipelines/")
                            and " refused: " in str(o.get("reason", "")))]
    for path, reason in pipeline_registry.clashes().items():
        if os.path.abspath(os.path.dirname(path)) == os.path.abspath(pipelines_dir):
            name = f"pipelines/{os.path.basename(path)}"
            _record(name, path, {"module": name, "status": "failed", "reason": reason})
    for stem in stems:
        if stem in _live_pipelines:
            continue
        try:
            runner = pipeline_registry.get_runner(stem)
        except Exception as exc:  # noqa: BLE001 — one broken pipeline must not stop the rest
            path = pipeline_registry.pipeline_path(stem)
            if path and path.endswith((".yaml", ".yml")):
                reason = "; ".join(getattr(exc, "errors", None) or [str(exc)])
                _record(f"pipelines/{stem}", path, {
                    "module": f"pipelines/{stem}", "status": "failed", "reason": reason})
            print(f"[tracebi] pipeline '{stem}' failed to load: {exc}", file=sys.stderr)
            continue
        _outcomes[:] = [o for o in _outcomes if o.get("module") != f"pipelines/{stem}"]
        if stem not in registry.list_pipeline_names():
            registry.add_pipeline(stem, runner)
        _live_pipelines.add(stem)
        added.append(stem)
    return added


def rescan(reports_dir: str, models_dir: Optional[str] = None,
           pipelines_dir: Optional[str] = None,
           connections_dir: Optional[str] = None) -> dict:
    """Bring the registry in line with the project on disk.

    Registers report packages and specs that appeared since the last scan,
    forgets ones whose source is gone, and adds new model files. Returns
    ``{"added", "removed", "failed", "models", "pipelines"}`` lists.

    *reports_dir* is one tree (tests, an explicit root). Prefer
    :func:`rescan_library` when the server should honour
    ``TRACEBI_LIBRARY_MOUNTS``.
    """
    found = _report_sources(reports_dir) if os.path.isdir(reports_dir) else {}
    return _apply_report_sources(found, models_dir, pipelines_dir, [reports_dir],
                                 connections_dir)


def rescan_library(models_dir: Optional[str] = None,
                   pipelines_dir: Optional[str] = None,
                   connections_dir: Optional[str] = None) -> dict:
    """:func:`rescan` across every configured library root."""
    from tracebi.report_paths import library_roots

    found: dict[str, str] = {}
    roots = []
    for label, root in library_roots():
        roots.append(str(root))
        if not root.is_dir():
            continue
        pfx = f"{label}/" if label else ""
        found.update(_report_sources(str(root), pfx))
    return _apply_report_sources(found, models_dir, pipelines_dir, roots,
                                 connections_dir)


def _within(path: str, roots: list[str]) -> bool:
    path = os.path.abspath(path)
    for root in roots:
        root = os.path.abspath(root)
        if path == root or path.startswith(root + os.sep):
            return True
    return False


def _apply_report_sources(found: dict[str, str], models_dir: Optional[str],
                          pipelines_dir: Optional[str], roots: list[str],
                          connections_dir: Optional[str] = None) -> dict:
    from tracebi.registry import registry

    added, removed, failed = [], [], []
    for name, source in found.items():
        if _live_reports.get(name) == source:
            continue
        times = _file_times(source)
        if _failed_sources.get(name) == (source, times):
            continue                      # still broken the same way
        if os.path.isdir(source):
            outcome = _register_template_package(source, stem=name)
        else:
            outcome = _register_spec_file(source, stem=name)
        _record(name, source, outcome)
        if outcome["status"] == "registered":
            _live_reports[name] = source
            _failed_sources.pop(name, None)
            added.append(name)
        else:
            _failed_sources[name] = (source, times)
            failed.append(name)
    # Forget only what this scan could have seen. A report an app module
    # discovered from its own folder is not under these roots; its absence
    # here is not a deletion (the demo app's reports vanished 5s after start).
    for name in [n for n in _live_reports
                 if n not in found and _within(_live_reports[n], roots)]:
        registry.remove_report(name)
        del _live_reports[name]
        _outcomes[:] = [o for o in _outcomes if o.get("module") != name]
        removed.append(name)
    connections = register_connections(connections_dir) if connections_dir else []
    models = register_models(models_dir) if models_dir else []
    pipelines = register_pipelines(pipelines_dir) if pipelines_dir else []
    return {"added": added, "removed": removed, "failed": failed,
            "models": models, "pipelines": pipelines, "connections": connections}


def start_watcher(reports_dir: str, models_dir: Optional[str],
                  interval: float, pipelines_dir: Optional[str] = None,
                  connections_dir: Optional[str] = None):
    """Run a live-discovery scan every *interval* seconds on a daemon thread.

    When ``TRACEBI_LIBRARY_MOUNTS`` is set, walks every mount
    (:func:`rescan_library`). Otherwise rescans *reports_dir* (the single
    root the server was started with). Returns a ``threading.Event``; set
    it to stop the thread.
    """
    import threading

    from tracebi.report_paths import parse_library_mounts

    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(interval):
            try:
                if parse_library_mounts() is not None:
                    changes = rescan_library(models_dir, pipelines_dir, connections_dir)
                else:
                    changes = rescan(reports_dir, models_dir, pipelines_dir,
                                     connections_dir)
            except Exception as exc:  # noqa: BLE001 — the watcher must outlive one bad scan
                print(f"[tracebi] live discovery scan failed: {exc}", file=sys.stderr)
                continue
            for key, verb in (("added", "found"), ("removed", "removed"),
                              ("models", "found model"),
                              ("pipelines", "found pipeline"),
                              ("connections", "found connection")):
                for name in changes[key]:
                    print(f"[tracebi] live discovery: {verb} {name}", file=sys.stderr)

    threading.Thread(target=loop, name="tracebi-discovery", daemon=True).start()
    return stop


def reload_modules() -> list[str]:
    """
    Re-import every module previously imported through :func:`auto_discover`.

    Useful for dev-mode: edit a request script, hit the reload endpoint,
    re-evaluate registrations without bouncing the server.
    """
    import time

    importlib.invalidate_caches()
    reloaded: list[str] = []
    for mod_name, path in list(_discovered.items()):
        if not os.path.isfile(path):
            continue
        # Bump mtime forward so Python's pyc cache (1s-resolution) cannot
        # shadow rapid back-to-back edits.
        future = time.time() + 2
        os.utime(path, (future, future))
        # Drop any stale .pyc that might already point at the old content.
        cache = importlib.util.cache_from_source(path)
        if cache and os.path.isfile(cache):
            try:
                os.remove(cache)
            except OSError:
                pass

        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        reloaded.append(mod_name)
    return reloaded


def discovered_modules() -> dict[str, str]:
    """Read-only view of currently-discovered modules → file path."""
    return dict(_discovered)
