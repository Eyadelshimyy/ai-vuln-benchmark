"""Abstract base renderer shared by ExcelRenderer, HTMLRenderer, etc."""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from typing import Optional

from tracebi.reports.report import Report, ReportManifest


_GIT_SHA_WARNED = False


def _warn_if_unknown_git_sha(manifest: ReportManifest) -> None:
    """One loud stderr line when code provenance is missing from the manifest.

    Emitted once per process — a report rendered to two formats back to back
    would otherwise print the identical warning twice, which teaches people
    to ignore it. Not an error: the report still renders, but the receipt
    cannot say which code produced it.
    """
    global _GIT_SHA_WARNED
    if _GIT_SHA_WARNED:
        return
    if manifest.git_sha == "unknown":
        _GIT_SHA_WARNED = True
        print(
            "tracebi: warning: git_sha is 'unknown' — code provenance is "
            "missing from this render's audit trail. Run the report from a "
            "git repository with at least one commit (`git init && git add . && git commit -m init` in your project) so the "
            "manifest records the commit.",
            file=sys.stderr,
        )


class BaseRenderer(ABC):
    """
    Abstract renderer. Subclass and implement ``_render()`` to add a new format.

    The public ``render()`` method handles directory creation, calls ``_render()``,
    builds the manifest, and optionally saves it alongside the output file.

    Usage (subclass):
        class MyRenderer(BaseRenderer):
            FORMAT = "my_format"

            def _render(self, report: Report, output_path: str) -> None:
                # write report to output_path
                ...

        manifest = MyRenderer().render(report, "output/report.myformat")
    """

    FORMAT: str = "base"

    @abstractmethod
    def _render(self, report: Report, output_path: str) -> None:
        """Write the rendered output to *output_path*."""
        ...

    def _augment_manifest(self, report: Report, manifest: ReportManifest) -> None:
        """Hook: add renderer-specific fields to the receipt before it is saved.

        Runs after ``build_manifest`` and before ``_render`` — manifest first,
        artifact second — so a subclass can record what it is about to embed.
        The HTML renderer uses it to populate ``embedded_data`` (so a rendered
        report with a chart is checkable by ``tracebi verify --file``). Default:
        nothing, so Excel and other formats keep their existing receipt shape.
        """
        return None

    def render(
        self,
        report: Report,
        output_path: str,
        save_manifest: bool = True,
        manifest_path: Optional[str] = None,
    ) -> ReportManifest:
        """
        Render *report* to *output_path* and return the manifest.

        Args:
            report:        The Report to render.
            output_path:   Destination file path.
            save_manifest: When True (default), write a ``.manifest.json``
                           alongside the output file.
            manifest_path: Override the manifest file path.

        Returns:
            ReportManifest: The manifest for this render run.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # Manifest first, artifact second. If building the receipt fails, the
        # caller gets the error and no file — rather than a rendered report on
        # disk that nothing can audit, which is the one state this whole
        # mechanism exists to prevent.
        manifest = report.build_manifest(format=self.FORMAT, output_path=output_path)
        self._augment_manifest(report, manifest)
        self._render(report, output_path)
        _warn_if_unknown_git_sha(manifest)
        if save_manifest:
            mp = manifest_path or output_path + ".manifest.json"
            manifest.save(mp)
        return manifest
