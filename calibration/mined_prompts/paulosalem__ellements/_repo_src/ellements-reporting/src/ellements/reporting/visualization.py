"""Visualization utilities for embedding charts in reports.

This module provides utilities for converting matplotlib figures to embeddable
formats (base64 PNG/SVG) for use in HTML reports.
"""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib.pyplot as plt


class ChartSerializer:
    """Convert matplotlib charts to embeddable formats."""

    @staticmethod
    def fig_to_base64_png(
        fig: plt.Figure, dpi: int = 100, close_fig: bool = True
    ) -> str:
        """Convert a matplotlib figure to a base64-encoded PNG."""
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=dpi)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        if close_fig:
            plt.close(fig)

        return img_base64

    @staticmethod
    def fig_to_base64_svg(fig: plt.Figure, close_fig: bool = True) -> str:
        """Convert a matplotlib figure to a base64-encoded SVG."""
        buffer = BytesIO()
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        buffer.seek(0)
        svg_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        if close_fig:
            plt.close(fig)

        return svg_base64

    @staticmethod
    def fig_to_html_img_tag(
        fig: plt.Figure,
        image_format: str = "png",
        alt: str = "Chart",
        dpi: int = 100,
        close_fig: bool = True,
    ) -> str:
        """Convert a figure directly to an HTML ``<img>`` tag.

        Args:
            fig: Matplotlib figure.
            image_format: Either ``"png"`` or ``"svg"``.
            alt: Alt text for the image.
            dpi: DPI for PNG output (ignored for SVG).
            close_fig: Whether to close the figure after conversion.
        """
        if image_format == "png":
            data = ChartSerializer.fig_to_base64_png(
                fig, dpi=dpi, close_fig=close_fig
            )
            return f'<img src="data:image/png;base64,{data}" alt="{alt}">'
        if image_format == "svg":
            data = ChartSerializer.fig_to_base64_svg(fig, close_fig=close_fig)
            return f'<img src="data:image/svg+xml;base64,{data}" alt="{alt}">'
        raise ValueError(
            f"Unsupported image_format: {image_format}. Use 'png' or 'svg'."
        )
