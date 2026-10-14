"""Tests for HTML report generation."""

import pandas as pd
import pytest
from ellements.reporting import HTMLReportGenerator, MultiFormatReportExporter
from jinja2 import TemplateNotFound
from pydantic import BaseModel


class SampleReport(BaseModel):
    """Sample report model for testing."""

    title: str
    total: float
    items: list[str]


@pytest.fixture
def sample_template_dir(tmp_path):
    """Create a temporary directory with sample templates."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir()

    # Create a simple template
    template_file = template_dir / "test_template.html"
    template_file.write_text(
        """
        <html>
        <head><title>{{ title }}</title></head>
        <body>
            <h1>{{ title }}</h1>
            <p>Total: {{ total }}</p>
            <ul>
            {% for item in items %}
                <li>{{ item }}</li>
            {% endfor %}
            </ul>
        </body>
        </html>
        """
    )

    return template_dir


@pytest.fixture
def sample_template_string():
    """Create a simple template string."""
    return """
    <html>
    <body>
        <h1>{{ title }}</h1>
        <p>Value: {{ value }}</p>
    </body>
    </html>
    """


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    return out_dir


class TestHTMLReportGenerator:
    """Test HTML report generator."""

    def test_initialization_with_dir(self, sample_template_dir):
        """Test initialization with template directory."""
        generator = HTMLReportGenerator(template_dir=sample_template_dir)

        assert generator.template_dir == sample_template_dir
        assert generator.env is not None

    def test_initialization_with_string(self, sample_template_string):
        """Test initialization with template string."""
        generator = HTMLReportGenerator(template_string=sample_template_string)

        assert generator.template_string == sample_template_string
        assert generator.env is not None

    def test_generate_from_file(self, sample_template_dir, output_dir):
        """Test generating report from template file."""
        generator = HTMLReportGenerator(template_dir=sample_template_dir)

        context = {
            "title": "Test Report",
            "total": 123.45,
            "items": ["Item 1", "Item 2", "Item 3"],
        }

        html = generator.generate(
            template_name="test_template.html",
            context=context,
            output_path=output_dir / "report.html",
        )

        # Verify HTML content
        assert "Test Report" in html
        assert "123.45" in html
        assert "Item 1" in html
        assert "Item 2" in html

        # Verify file was written
        output_file = output_dir / "report.html"
        assert output_file.exists()
        assert "Test Report" in output_file.read_text()

    def test_generate_from_string(self, sample_template_string):
        """Test generating report from template string."""
        generator = HTMLReportGenerator(template_string=sample_template_string)

        context = {"title": "Dynamic Report", "value": 42}

        html = generator.generate(context=context)

        assert "Dynamic Report" in html
        assert "42" in html

    def test_generate_without_output_path(self, sample_template_dir):
        """Test generating without saving to file."""
        generator = HTMLReportGenerator(template_dir=sample_template_dir)

        context = {"title": "In-Memory Report", "total": 99.99, "items": []}

        html = generator.generate(template_name="test_template.html", context=context)

        # Should return HTML string without writing file
        assert isinstance(html, str)
        assert "In-Memory Report" in html

    def test_generate_with_complex_context(self, sample_template_dir, output_dir):
        """Test generating with complex nested context."""
        complex_template = sample_template_dir / "complex.html"
        complex_template.write_text(
            """
            <html>
            <body>
                <h1>{{ report.title }}</h1>
                {% for section in report.sections %}
                <section>
                    <h2>{{ section.name }}</h2>
                    <p>{{ section.description }}</p>
                </section>
                {% endfor %}
            </body>
            </html>
            """
        )

        generator = HTMLReportGenerator(template_dir=sample_template_dir)

        context = {
            "report": {
                "title": "Complex Report",
                "sections": [
                    {"name": "Section 1", "description": "First section"},
                    {"name": "Section 2", "description": "Second section"},
                ],
            }
        }

        html = generator.generate(template_name="complex.html", context=context)

        assert "Complex Report" in html
        assert "Section 1" in html
        assert "Section 2" in html

    def test_generate_missing_template(self, sample_template_dir):
        """Test error when template file doesn't exist."""
        generator = HTMLReportGenerator(template_dir=sample_template_dir)

        with pytest.raises(TemplateNotFound):
            generator.generate(template_name="nonexistent.html", context={})


class TestMultiFormatReportExporter:
    """Test multi-format report exporter."""

    def test_initialization(self, output_dir):
        """Test exporter initialization."""
        exporter = MultiFormatReportExporter(output_dir)

        assert exporter.output_dir == output_dir
        assert output_dir.exists()

    def test_initialization_creates_directory(self, tmp_path):
        """Test that initialization creates output directory."""
        new_dir = tmp_path / "new_output"
        assert not new_dir.exists()

        exporter = MultiFormatReportExporter(new_dir)

        assert new_dir.exists()
        assert exporter.output_dir == new_dir

    def test_export_html(self, output_dir):
        """Test HTML export."""
        exporter = MultiFormatReportExporter(output_dir)

        html_content = "<html><body><h1>Test Report</h1></body></html>"
        output_path = exporter.export_html(html_content, filename="test.html")

        # Verify file was created
        assert output_path.exists()
        assert output_path.name == "test.html"

        # Verify content
        content = output_path.read_text(encoding="utf-8")
        assert "Test Report" in content

    def test_export_html_custom_filename(self, output_dir):
        """Test HTML export with custom filename."""
        exporter = MultiFormatReportExporter(output_dir)

        html_content = "<html><body>Custom</body></html>"
        output_path = exporter.export_html(html_content, filename="custom_name.html")

        assert output_path.name == "custom_name.html"
        assert output_path.exists()

    def test_export_json(self, output_dir):
        """Test JSON export with Pydantic model."""
        exporter = MultiFormatReportExporter(output_dir)

        report = SampleReport(
            title="JSON Test", total=456.78, items=["Alpha", "Beta", "Gamma"]
        )

        output_path = exporter.export_json(report, filename="report.json")

        # Verify file was created
        assert output_path.exists()
        assert output_path.name == "report.json"

        # Verify JSON content
        import json

        with open(output_path) as f:
            data = json.load(f)

        assert data["title"] == "JSON Test"
        assert data["total"] == 456.78
        assert len(data["items"]) == 3

    def test_export_json_formatting(self, output_dir):
        """Test JSON export uses pretty formatting."""
        exporter = MultiFormatReportExporter(output_dir)

        report = SampleReport(title="Formatted", total=100.00, items=["One", "Two"])

        output_path = exporter.export_json(report)

        # Verify indentation (pretty printing)
        content = output_path.read_text()
        assert "  " in content  # Should have indentation

    def test_export_csv(self, output_dir):
        """Test CSV export with DataFrame."""
        exporter = MultiFormatReportExporter(output_dir)

        df = pd.DataFrame(
            {"name": ["Alice", "Bob", "Charlie"], "value": [10, 20, 30], "active": [True, False, True]}
        )

        output_path = exporter.export_csv(df, filename="data.csv")

        # Verify file was created
        assert output_path.exists()
        assert output_path.name == "data.csv"

        # Verify CSV content
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["name", "value", "active"]
        assert loaded_df["name"].tolist() == ["Alice", "Bob", "Charlie"]

    def test_export_csv_default_filename(self, output_dir):
        """Test CSV export with default filename."""
        exporter = MultiFormatReportExporter(output_dir)

        df = pd.DataFrame({"col1": [1, 2, 3]})

        output_path = exporter.export_csv(df)

        assert output_path.name == "data.csv"  # Default

    def test_export_csv_no_index(self, output_dir):
        """Test CSV export doesn't include index."""
        exporter = MultiFormatReportExporter(output_dir)

        df = pd.DataFrame({"values": [1, 2, 3]})

        output_path = exporter.export_csv(df)

        # Verify index is not in CSV
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "values"  # Only column name, no index column

    def test_multiple_exports(self, output_dir):
        """Test exporting multiple formats."""
        exporter = MultiFormatReportExporter(output_dir)

        # Export HTML
        html_path = exporter.export_html("<html><body>Test</body></html>", filename="report.html")

        # Export JSON
        report = SampleReport(title="Multi", total=123, items=["A", "B"])
        json_path = exporter.export_json(report, filename="report.json")

        # Export CSV
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        csv_path = exporter.export_csv(df, filename="report.csv")

        # Verify all files exist
        assert html_path.exists()
        assert json_path.exists()
        assert csv_path.exists()

        # Verify they're in the same directory
        assert html_path.parent == output_dir
        assert json_path.parent == output_dir
        assert csv_path.parent == output_dir

    def test_export_overwrites_existing(self, output_dir):
        """Test that export overwrites existing files."""
        exporter = MultiFormatReportExporter(output_dir)

        # First export
        first_html = "<html><body>First</body></html>"
        exporter.export_html(first_html, filename="test.html")

        # Second export (overwrite)
        second_html = "<html><body>Second</body></html>"
        output_path = exporter.export_html(second_html, filename="test.html")

        # Verify content was overwritten
        content = output_path.read_text()
        assert "Second" in content
        assert "First" not in content


class TestHTMLReportIntegration:
    """Integration tests for HTML report generation."""

    def test_end_to_end_report_generation(self, sample_template_dir, output_dir):
        """Test complete workflow: generate HTML and export to file."""
        # Create generator
        generator = HTMLReportGenerator(template_dir=sample_template_dir)

        # Prepare context
        context = {
            "title": "Integration Test Report",
            "total": 999.99,
            "items": ["First", "Second", "Third"],
        }

        # Generate HTML
        html = generator.generate(
            template_name="test_template.html",
            context=context,
            output_path=output_dir / "integration_report.html",
        )

        # Verify HTML content
        assert "Integration Test Report" in html
        assert "999.99" in html

        # Verify file exists
        report_file = output_dir / "integration_report.html"
        assert report_file.exists()

    def test_export_multiple_formats_workflow(self, output_dir):
        """Test exporting same report in multiple formats."""
        exporter = MultiFormatReportExporter(output_dir)

        # Create report data
        report = SampleReport(title="Multi-Format Report", total=777.77, items=["X", "Y", "Z"])

        # Export as JSON
        json_path = exporter.export_json(report, filename="multi_report.json")

        # Create HTML from report data
        html_content = f"""
        <html>
        <body>
            <h1>{report.title}</h1>
            <p>Total: {report.total}</p>
            <ul>
                {"".join(f"<li>{item}</li>" for item in report.items)}
            </ul>
        </body>
        </html>
        """
        html_path = exporter.export_html(html_content, filename="multi_report.html")

        # Create DataFrame from report
        df = pd.DataFrame({"item": report.items, "report_total": [report.total] * len(report.items)})
        csv_path = exporter.export_csv(df, filename="multi_report.csv")

        # Verify all exports
        assert json_path.exists()
        assert html_path.exists()
        assert csv_path.exists()

        # Verify content consistency
        import json

        with open(json_path) as f:
            json_data = json.load(f)
        assert json_data["title"] == "Multi-Format Report"

        html_content_file = html_path.read_text()
        assert "Multi-Format Report" in html_content_file

        loaded_df = pd.read_csv(csv_path)
        assert loaded_df["report_total"].iloc[0] == 777.77
