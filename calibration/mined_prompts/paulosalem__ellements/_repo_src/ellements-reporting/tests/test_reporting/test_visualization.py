"""Tests for visualization utilities."""

import base64

import matplotlib.pyplot as plt
import pytest
from ellements.reporting import ChartSerializer


class TestChartSerializer:
    """Test chart serialization to base64 formats."""

    @pytest.fixture
    def sample_figure(self):
        """Create a sample matplotlib figure for testing."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title("Test Chart")
        return fig

    def test_fig_to_base64_png(self, sample_figure):
        """Test PNG serialization."""
        base64_str = ChartSerializer.fig_to_base64_png(sample_figure, close_fig=True)

        # Verify it's a valid base64 string
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

        # Verify it can be decoded
        decoded = base64.b64decode(base64_str)
        assert len(decoded) > 0

        # Verify PNG signature
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_fig_to_base64_svg(self, sample_figure):
        """Test SVG serialization."""
        base64_str = ChartSerializer.fig_to_base64_svg(sample_figure, close_fig=True)

        # Verify it's a valid base64 string
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

        # Verify it can be decoded
        decoded = base64.b64decode(base64_str)
        assert len(decoded) > 0

        # Verify SVG content
        decoded_str = decoded.decode("utf-8")
        assert "<svg" in decoded_str
        assert "</svg>" in decoded_str

    def test_fig_to_html_img_tag_png(self, sample_figure):
        """Test HTML img tag generation with PNG."""
        img_tag = ChartSerializer.fig_to_html_img_tag(
            sample_figure, image_format="png", alt="Test Chart", close_fig=True
        )

        assert img_tag.startswith('<img src="data:image/png;base64,')
        assert 'alt="Test Chart"' in img_tag
        assert img_tag.endswith('">')

    def test_fig_to_html_img_tag_svg(self, sample_figure):
        """Test HTML img tag generation with SVG."""
        img_tag = ChartSerializer.fig_to_html_img_tag(
            sample_figure, image_format="svg", alt="Test Chart", close_fig=True
        )

        assert img_tag.startswith('<img src="data:image/svg+xml;base64,')
        assert 'alt="Test Chart"' in img_tag
        assert img_tag.endswith('">')

    def test_fig_to_html_img_tag_invalid_format(self, sample_figure):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported image_format"):
            ChartSerializer.fig_to_html_img_tag(
                sample_figure, image_format="invalid", close_fig=True
            )

    def test_dpi_parameter(self):
        """Test that DPI parameter affects output size."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        low_dpi = ChartSerializer.fig_to_base64_png(fig, dpi=50, close_fig=False)
        high_dpi = ChartSerializer.fig_to_base64_png(fig, dpi=300, close_fig=True)

        # Higher DPI should produce larger file
        assert len(high_dpi) > len(low_dpi)
