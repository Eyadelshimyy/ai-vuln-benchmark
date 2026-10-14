"""Smoke: project template get/parse/create roundtrips."""
from tools.project_types import (
    create_project_from_template,
    get_project_template,
    parse_project_template,
)

_VALID_TEMPLATE = """---
project_type: merchant-onboarding
name: "Acme Corp"
description: "BNPL integration test"
market: "DE"
products: "BNPL 30d"
phase: "Discovery"
go_live: "2026-12-31"
budget: ""
notes: ""
---
## Hints
Some hints here.
"""


# ---------------------------------------------------------------------------
# get_project_template
# ---------------------------------------------------------------------------

def test_get_template_known_type():
    result = get_project_template("merchant-onboarding")
    assert "error" not in result
    assert result["project_type"] == "merchant-onboarding"
    assert "template_content" in result
    assert "merchant-onboarding" in result["template_content"]
    assert "name:" in result["template_content"]


def test_get_template_generic():
    result = get_project_template("generic")
    assert "error" not in result
    assert result["project_type"] == "generic"
    assert "template_content" in result


def test_get_template_fallback_for_unknown_type():
    result = get_project_template("nonexistent-type-xyz")
    assert "error" not in result
    assert result.get("fallback") is True
    assert "template_content" in result


# ---------------------------------------------------------------------------
# parse_project_template
# ---------------------------------------------------------------------------

def test_parse_extracts_fields():
    fields = parse_project_template(_VALID_TEMPLATE)
    assert "error" not in fields
    assert fields["name"] == "Acme Corp"
    assert fields["project_type"] == "merchant-onboarding"
    assert fields["market"] == "DE"
    assert fields["phase"] == "Discovery"
    assert fields["go_live"] == "2026-12-31"


def test_parse_empty_string_fields():
    content = """---
project_type: generic
name: "Test"
description: ""
market: ""
---
"""
    fields = parse_project_template(content)
    assert "error" not in fields
    assert fields["name"] == "Test"
    assert fields["description"] == ""
    assert fields["market"] == ""


def test_parse_date_value_becomes_string():
    # YAML parses unquoted ISO dates as date objects — must be coerced to str
    content = """---
project_type: generic
name: "Date Test"
go_live: 2026-12-31
---
"""
    fields = parse_project_template(content)
    assert "error" not in fields
    assert fields["go_live"] == "2026-12-31"


def test_parse_complex_yaml_value_returns_error():
    content = """---
project_type: generic
name: "Test"
budget:
  - high
  - low
---
"""
    result = parse_project_template(content)
    assert "error" in result
    assert "budget" in result["error"]


def test_parse_no_frontmatter_returns_error():
    result = parse_project_template("Just plain text, no YAML")
    assert "error" in result


def test_parse_unclosed_frontmatter_returns_error():
    content = """---
project_type: generic
name: "No closing marker"
"""
    result = parse_project_template(content)
    assert "error" in result


# ---------------------------------------------------------------------------
# create_project_from_template
# ---------------------------------------------------------------------------

def test_create_from_template_content():
    result = create_project_from_template(template_content=_VALID_TEMPLATE)
    assert "error" not in result
    assert result["name"] == "Acme Corp"
    assert result["type"] == "merchant-onboarding"


def test_create_from_template_missing_name():
    content = """---
project_type: generic
name: ""
description: "No name"
---
"""
    result = create_project_from_template(template_content=content)
    assert "error" in result


def test_create_from_template_file_path(tmp_path):
    template_file = tmp_path / "project.md"
    template_file.write_text("""---
project_type: it-project
name: "File Import Test"
description: "Imported from file"
phase: "Planning"
go_live: ""
market: ""
products: ""
budget: ""
notes: ""
---
""")
    result = create_project_from_template(file_path=str(template_file))
    assert "error" not in result
    assert result["name"] == "File Import Test"
    assert result["type"] == "it-project"


def test_create_from_template_file_not_found():
    result = create_project_from_template(file_path="/nonexistent/path/file.md")
    assert "error" in result


def test_create_from_template_file_path_must_be_md():
    result = create_project_from_template(file_path="/tmp/project.txt")
    assert "error" in result
    assert ".md" in result["error"]


def test_create_from_template_no_args_returns_error():
    result = create_project_from_template()
    assert "error" in result
