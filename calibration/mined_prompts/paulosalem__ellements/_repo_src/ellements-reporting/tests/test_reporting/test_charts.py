"""Tests for chart artifact generation."""

import base64

import pytest
from ellements.reporting.charts import create_chart_artifacts


class TestCreateChartArtifacts:
    def test_line_chart_artifacts_include_canvas_payloads(self):
        artifacts = create_chart_artifacts(
            {"type": "line", "x": [1, 2, 3], "y": [10, 15, 12], "title": "Trend"},
            title="Trend",
        )

        assert artifacts["chart_type"] == "line"
        assert artifacts["mime_type"] == "image/png"
        assert artifacts["data_uri"].startswith("data:image/png;base64,")
        assert artifacts["canvas_chart_payload"]["type"] == "line"
        assert artifacts["canvas_chart_payload"]["title"] == "Trend"
        assert artifacts["canvas_card_item"]["type"] == "image"
        assert "canvas-chart" in artifacts["canvas_chart_block"]

    def test_svg_output_decodes(self):
        artifacts = create_chart_artifacts(
            {
                "type": "bar",
                "labels": ["A", "B", "C"],
                "values": [3, 5, 2],
                "title": "Allocation",
            },
            image_format="svg",
        )

        assert artifacts["mime_type"] == "image/svg+xml"
        encoded = artifacts["image_base64"]
        decoded = base64.b64decode(encoded).decode("utf-8")
        assert "<svg" in decoded
        assert "</svg>" in decoded

    def test_histogram_chart_support(self):
        artifacts = create_chart_artifacts(
            {"type": "histogram", "values": [1, 2, 2, 3, 3, 3], "bins": 3, "title": "Dist"}
        )

        assert artifacts["chart_type"] == "histogram"
        assert "data:image/png;base64," in artifacts["html_img_tag"]

    def test_invalid_chart_spec_raises(self):
        with pytest.raises(ValueError, match="Unsupported chart type"):
            create_chart_artifacts({"type": "invalid", "values": [1, 2, 3]})
