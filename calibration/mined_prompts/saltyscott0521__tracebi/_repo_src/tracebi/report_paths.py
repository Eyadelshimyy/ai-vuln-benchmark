"""Report names are paths below the library root(s).

A report at the top of the reports folder is named ``weekly_summary``; one in a
folder is named by its path, ``finance/weekly_summary``, so two folders can
each hold a report of the same name. Every surface that turns a name into a
file (the CLI, the web API, the agent gateway, schedules) goes through here, so
a name can never climb out of a library root or ``output/``.

With :envvar:`TRACEBI_LIBRARY_MOUNTS` set (``label:/abs/path,...``), each
mount is a top-level library folder and identity is ``label/relative_path``.
When that variable is unset, :envvar:`TRACEBI_REPORTS_DIR` is the single root
and identity stays the path relative to it — unchanged from before mounts.

:func:`open_report` is the one read of a named report. The permission check
will go there.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

_MOUNTS_IGNORED_REPORTS_DIR = False


def report_name_error(name: str) -> Optional[str]:
    """``None`` if *name* is a safe report name, else a message saying why not.

    Allowed: ``weekly`` and ``finance/month_end/weekly``. Refused: an empty
    name or segment, an absolute path, ``..``, backslashes, and segments that
    start with ``.`` or ``_`` (discovery skips those, so no report has one).
    """
    if not isinstance(name, str) or not name:
        return "a report name is required"
    if "\\" in name or name.startswith("/") or os.path.isabs(name):
        return (f"invalid report name {name!r}: use the report's name, or its "
                f"path below reports/ with forward slashes (finance/weekly)")
    for part in name.split("/"):
        if not part or part in (".", "..") or part.startswith((".", "_")):
            return (f"invalid report name {name!r}: each part of a report path "
                    f"must be a folder or report name, not {part!r}")
    return None


def _mount_label_error(label: str) -> Optional[str]:
    """``None`` if *label* is a safe mount label (one path segment)."""
    if "/" in label or "\\" in label:
        return (f"invalid mount label {label!r}: must be a single path segment "
                f"(no slashes)")
    return report_name_error(label)


def parse_library_mounts(raw: Optional[str] = None) -> Optional[list[tuple[str, Path]]]:
    """Parse ``TRACEBI_LIBRARY_MOUNTS`` into ``[(label, abs_path), ...]``.

    Returns ``None`` when the variable is unset or empty. Raises
    ``ValueError`` when it is set but malformed (bad label, relative path,
    duplicate label, missing ``:``).
    """
    if raw is None:
        raw = os.environ.get("TRACEBI_LIBRARY_MOUNTS")
    if raw is None or not str(raw).strip():
        return None
    mounts: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for chunk in str(raw).split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(
                f"TRACEBI_LIBRARY_MOUNTS entry {piece!r} must be "
                f"label:/absolute/path")
        label, path_s = piece.split(":", 1)
        label = label.strip()
        path_s = path_s.strip()
        err = _mount_label_error(label)
        if err:
            raise ValueError(f"TRACEBI_LIBRARY_MOUNTS: {err}")
        if label in seen:
            raise ValueError(
                f"TRACEBI_LIBRARY_MOUNTS: duplicate mount label {label!r}")
        if not path_s or not os.path.isabs(path_s):
            raise ValueError(
                f"TRACEBI_LIBRARY_MOUNTS: path for {label!r} must be absolute "
                f"(got {path_s!r})")
        seen.add(label)
        mounts.append((label, Path(path_s)))
    if not mounts:
        return None
    return mounts


def library_roots(
    *,
    reports_dir: Optional[Union[str, os.PathLike]] = None,
) -> list[tuple[Optional[str], Path]]:
    """Library roots as ``(label_or_None, path)``.

    When :envvar:`TRACEBI_LIBRARY_MOUNTS` is set it wins: each entry is
    ``(label, abs_path)``. Otherwise the single root is
    ``(None, TRACEBI_REPORTS_DIR or reports)`` — or *reports_dir* when the
    caller passes an explicit directory (CLI ``--reports-dir``, tests).
    """
    global _MOUNTS_IGNORED_REPORTS_DIR
    mounts = parse_library_mounts()
    if mounts is not None:
        if (not _MOUNTS_IGNORED_REPORTS_DIR
                and os.environ.get("TRACEBI_REPORTS_DIR")
                and reports_dir is None):
            _MOUNTS_IGNORED_REPORTS_DIR = True
            print(
                "[tracebi] TRACEBI_LIBRARY_MOUNTS is set; "
                "TRACEBI_REPORTS_DIR is ignored.",
                file=sys.stderr,
            )
        if reports_dir is not None:
            # Explicit single-root escape hatch (CLI --reports-dir).
            return [(None, Path(reports_dir))]
        return [(label, path) for label, path in mounts]
    root = Path(reports_dir if reports_dir is not None
                else os.environ.get("TRACEBI_REPORTS_DIR", "reports"))
    return [(None, root)]


def output_html(output_dir: str, name: str) -> str:
    """``<output_dir>/<name>.html``, keeping the report's folders.

    ``finance/weekly`` builds to ``output/finance/weekly.html``, so two
    same-named reports in different folders never overwrite each other.
    """
    err = report_name_error(name)
    if err:
        raise ValueError(err)
    *folders, leaf = name.split("/")
    return os.path.join(output_dir, *folders, f"{leaf}.html")


def report_name_for_dir(directory: str, project_root: Optional[str] = None) -> str:
    """The report name of a package directory: its path in the library.

    Under a single ``TRACEBI_REPORTS_DIR``, ``reports/finance/weekly`` is
    ``finance/weekly``. Under mounts, a package in the ``finance`` mount is
    ``finance/weekly``. A package outside every library root (a test fixture,
    an explicit path) is named by its own directory basename.
    """
    full = os.path.realpath(directory)
    # Resolve roots relative to project_root when they are not absolute
    # (the default ``reports`` case).
    root_base = project_root or os.getcwd()
    for label, root in library_roots():
        root_path = root if root.is_absolute() else Path(root_base) / root
        reports = os.path.realpath(str(root_path))
        if full == reports or full.startswith(reports + os.sep):
            if full == reports:
                rel = ""
            else:
                rel = os.path.relpath(full, reports).replace(os.sep, "/")
            if label:
                return f"{label}/{rel}" if rel else label
            return rel or os.path.basename(os.path.normpath(directory))
    return os.path.basename(os.path.normpath(directory))


@dataclass(frozen=True)
class OpenedReport:
    """Where a named report lives, for one read.

    ``path`` and ``spec_path`` are the package directory and the
    ``<name>.json`` spec under the reports folder. They are set only when
    *name* passes :func:`report_name_error` — an unsafe name is refused
    before any filesystem access. ``package_dir`` is ``path`` when that
    directory holds ``report.json``; ``has_template`` / ``has_spec`` say the
    other two files are present.

    The registry fields are filled only when the caller passes a registry
    (the web app). A spec's compiled package directory lives on the factory,
    not at ``path``, so ``registry_package_dir`` is that directory.
    """

    name: str
    purpose: str
    reports_dir: Path
    name_error: Optional[str]
    path: Optional[Path]
    spec_path: Optional[Path]
    package_dir: Optional[Path]
    has_template: bool
    has_spec: bool
    registered: Optional[bool]
    registry_package_dir: Optional[str]
    source: Optional[dict]


def _resolve_under_root(root: Path, relative: str) -> tuple[Path, Path, Optional[Path], bool, bool]:
    """Locate package/spec for *relative* under *root*."""
    path = root / relative
    spec_path = root / f"{relative}.json"
    package_dir = None
    has_template = has_spec = False
    if (path / "report.json").is_file():
        package_dir = path
        has_template = (path / "template.html").is_file()
    has_spec = spec_path.is_file()
    return path, spec_path, package_dir, has_template, has_spec


def open_report(
    name: str,
    *,
    purpose: str,
    reports_dir: Optional[Union[str, os.PathLike]] = None,
    registry: Optional[object] = None,
) -> OpenedReport:
    """Resolve *name* for one read. Every open, download, Source view, MCP
    tool and schedule calls this.

    Validates *name* with :func:`report_name_error` and, when it is safe,
    locates the package directory and spec file under the library root(s).
    When *registry* is given, also locates the registered factory. An unsafe
    name sets ``name_error`` and does not touch the filesystem. A missing
    report is not an error here: the location fields stay empty so each
    caller raises the not-found error it already raises (HTTP 404, the
    gateway's error dict, the CLI's ``FileNotFoundError``).

    This is where the permission check will go. *purpose* names the read —
    ``view``, ``build``, ``source``, ``schedule``, ``manage`` — so that check
    can map it to View, Build, Publish or Manage without each call site
    changing again. The check raises before the report is handed back.
    """
    name_error = report_name_error(name)
    path = spec_path = package_dir = None
    has_template = has_spec = False
    resolved_root = Path(
        reports_dir if reports_dir is not None
        else os.environ.get("TRACEBI_REPORTS_DIR", "reports"))

    if name_error is None:
        if reports_dir is not None:
            resolved_root = Path(reports_dir)
            path, spec_path, package_dir, has_template, has_spec = (
                _resolve_under_root(resolved_root, name))
        else:
            mounts = parse_library_mounts()
            if mounts is None:
                resolved_root = Path(
                    os.environ.get("TRACEBI_REPORTS_DIR", "reports"))
                path, spec_path, package_dir, has_template, has_spec = (
                    _resolve_under_root(resolved_root, name))
            else:
                # Mounts: first segment is the label; the rest is under that root.
                parts = name.split("/", 1)
                if len(parts) == 2:
                    label, relative = parts
                    mount_map = {lbl: p for lbl, p in mounts}
                    if label in mount_map:
                        resolved_root = mount_map[label]
                        path, spec_path, package_dir, has_template, has_spec = (
                            _resolve_under_root(resolved_root, relative))
                # else: name has no label → not under any mount (fields stay empty)

    registered = registry_package_dir = source = None
    if registry is not None:
        registered = name in {r["name"] for r in registry.list_reports()}
        if registered:
            registry_package_dir = registry.report_package_dir(name)
            source = registry.report_source(name)

    return OpenedReport(
        name=name if isinstance(name, str) else "",
        purpose=purpose,
        reports_dir=resolved_root,
        name_error=name_error,
        path=path,
        spec_path=spec_path,
        package_dir=package_dir,
        has_template=has_template,
        has_spec=has_spec,
        registered=registered,
        registry_package_dir=registry_package_dir,
        source=source,
    )


def reset_mounts_warning_for_tests() -> None:
    """Clear the once-only stderr warning flag (test helper)."""
    global _MOUNTS_IGNORED_REPORTS_DIR
    _MOUNTS_IGNORED_REPORTS_DIR = False
