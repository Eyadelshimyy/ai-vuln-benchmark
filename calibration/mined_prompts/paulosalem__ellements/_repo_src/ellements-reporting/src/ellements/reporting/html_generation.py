"""HTML report generation utilities.

This module provides utilities for generating HTML reports from templates
and exporting data in multiple formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel


class HTMLReportGenerator:
    """Generate HTML reports from Jinja2 templates.

    Example:
        ```python
        from pathlib import Path
        from ellements.reporting import HTMLReportGenerator

        generator = HTMLReportGenerator(template_dir=Path("templates"))
        html = generator.generate(
            template_name="report.html",
            context={"title": "My Report", "data": [1, 2, 3]},
            output_path=Path("report.html"),
        )

        generator = HTMLReportGenerator(template_string="<h1>{{title}}</h1>")
        html = generator.generate(context={"title": "Hello"})
        ```
    """

    def __init__(
        self,
        template_dir: Path | None = None,
        template_string: str | None = None,
    ) -> None:
        """Initialize the HTML report generator.

        Args:
            template_dir: Directory containing Jinja2 templates.
            template_string: Template string (alternative to *template_dir*).

        Raises:
            ValueError: If neither *template_dir* nor *template_string* is
                provided.
        """
        if template_dir:
            self.env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=False,
            )
            self.template_dir: Path | None = template_dir
            self.template_string: str | None = None
        elif template_string:
            self.env = Environment(autoescape=False)
            self.template_string = template_string
            self.template_dir = None
        else:
            raise ValueError(
                "Either template_dir or template_string must be provided"
            )

    def generate(
        self,
        template_name: str | None = None,
        context: dict[str, Any] | None = None,
        output_path: Path | None = None,
    ) -> str:
        """Generate an HTML report.

        Args:
            template_name: Name of the template file (required when
                ``template_dir`` is used).
            context: Dictionary of template variables.
            output_path: Optional path to write the rendered HTML.

        Returns:
            The rendered HTML as a string.
        """
        context = context or {}
        if self.template_dir and template_name:
            template = self.env.get_template(template_name)
        elif self.template_string:
            template = self.env.from_string(self.template_string)
        else:
            raise ValueError(
                "template_name required when using template_dir, "
                "or use template_string in __init__"
            )

        html = str(template.render(**context))

        if output_path:
            output_path.write_text(html, encoding="utf-8")

        return html


class MultiFormatReportExporter:
    """Export reports in multiple formats (HTML, JSON, CSV)."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize the multi-format exporter.

        Args:
            output_dir: Directory to write output files (created if needed).
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_html(self, content: str, filename: str = "report.html") -> Path:
        """Write *content* as an HTML file under :attr:`output_dir`."""
        path = self.output_dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def export_json(self, data: BaseModel, filename: str = "report.json") -> Path:
        """Export a Pydantic model as JSON."""
        path = self.output_dir / filename
        path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
        return path

    def export_csv(self, df: pd.DataFrame, filename: str = "data.csv") -> Path:
        """Export a DataFrame as CSV."""
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        return path
