"""Template loading and rendering helpers.

Templates can be loaded either from a filesystem directory or from
package resources via ``importlib.resources``. Mustache-style
substitution is delegated to :mod:`chevron`.

The renderer validates its source at construction so misconfiguration
fails fast (rather than at the first call).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import chevron

_logger = logging.getLogger(__name__)


class TemplateRenderer:
    """Load and render mustache templates from a filesystem dir or a package.

    Exactly one of ``template_dir`` or ``package`` must be provided.

    Args:
        template_dir: Directory containing template files.
        package: Importable package name whose resources hold templates.
        resource_root: Sub-path within *package* (or *template_dir*) to
            resolve template paths against. Default ``"prompts"``.
    """

    def __init__(
        self,
        template_dir: Path | str | None = None,
        *,
        package: str | None = None,
        resource_root: str = "prompts",
    ) -> None:
        if (template_dir is None) == (package is None):
            raise ValueError(
                "TemplateRenderer requires exactly one of template_dir or package."
            )

        self.resource_root = resource_root.strip("/")

        if template_dir is not None:
            self.template_dir: Path | None = Path(template_dir)
            if not self.template_dir.exists():
                raise FileNotFoundError(
                    f"Template directory not found: {self.template_dir}"
                )
            self.package: str | None = None
        else:
            assert package is not None
            self.template_dir = None
            self.package = package
            # Probe the package resource root to fail-fast on bad config.
            try:
                files(package)
            except ModuleNotFoundError as exc:
                raise ValueError(
                    f"TemplateRenderer package {package!r} cannot be imported: {exc}"
                ) from exc

    def _resource_traversable(self, template_path: str) -> Traversable:
        assert self.package is not None
        resource = files(self.package)
        if self.resource_root:
            resource = resource.joinpath(*self.resource_root.split("/"))
        return resource.joinpath(*template_path.split("/"))

    def load_template(self, template_path: str) -> str:
        """Read the raw template text at *template_path*."""
        if self.template_dir is not None:
            return (self.template_dir / template_path).read_text(encoding="utf-8")
        return self._resource_traversable(template_path).read_text(encoding="utf-8")

    @staticmethod
    def render_template(template_content: str, context: Mapping[str, Any]) -> str:
        """Render *template_content* against *context* using mustache syntax."""
        rendered: str = chevron.render(template_content, dict(context))
        return rendered

    def load_and_render(self, template_path: str, context: Mapping[str, Any]) -> str:
        """Convenience: load *template_path* and render it against *context*."""
        return self.render_template(self.load_template(template_path), context)

    def render_with_fallback(
        self,
        template_path: str,
        context: Mapping[str, Any],
        fallback: str,
        *,
        silent: bool = False,
    ) -> str:
        """Render a template, returning *fallback* on any error."""
        try:
            return self.load_and_render(template_path, context)
        except Exception as exc:
            if not silent:
                _logger.warning(
                    "Error loading template %r: %s. Using fallback content.",
                    template_path,
                    exc,
                )
            return fallback


def render_prompt_template(
    template_path: str,
    *,
    package: str,
    resource_root: str = "prompts",
    **kwargs: Any,
) -> str:
    """One-shot helper to load+render a packaged template with kwargs as context."""
    renderer = TemplateRenderer(package=package, resource_root=resource_root)
    return renderer.load_and_render(template_path, kwargs)


__all__ = ["TemplateRenderer", "render_prompt_template"]
