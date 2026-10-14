from collections import OrderedDict

import pytest

from scripts.generate_features import (
    check_docs,
    collect_features,
    extract_test_names,
    feature_area_from_filename,
    humanize_name,
    render_features_markdown,
)


def test_given_snake_case_test_name_when_humanize_name_called_then_returns_capitalized_sentence():
    assert humanize_name("test_analyze_valid_ticker") == "Analyze valid ticker"


def test_given_file_with_mixed_functions_when_extract_test_names_called_then_returns_only_test_prefixed_names(tmp_path):
    # Includes a top-level test function, a top-level helper, and a method inside
    # a Test* class to verify all three cases are handled correctly.
    test_file = tmp_path / "sample_test.py"
    test_file.write_text(
        "def test_alpha():\n"
        "    pass\n"
        "\n"
        "def helper_setup():\n"
        "    pass\n"
        "\n"
        "class TestBeta:\n"
        "    def test_beta_method(self):\n"
        "        pass\n"
        "\n"
        "    def not_a_test(self):\n"
        "        pass\n"
    )

    names = extract_test_names(test_file)

    assert sorted(names) == ["test_alpha", "test_beta_method"]


@pytest.mark.parametrize("stem, expected", [
    ("test_analyze_endpoint", "Sentiment Analysis"),
    ("test_auth_endpoints", "Authentication"),
    ("test_news_comparison_endpoint", "News Comparison"),
    ("test_workflow", "Analysis Pipeline"),
    ("test_app_flow", "App Experience"),
    ("test_login_page", "Login"),
    ("test_some_new_thing", "Some New Thing"),
])
def test_given_filename_stem_when_feature_area_from_filename_called_then_returns_readable_area_name(stem, expected):
    assert feature_area_from_filename(stem) == expected


def test_given_integration_and_e2e_test_files_when_collect_features_called_then_returns_ordered_dict_with_humanized_behaviors(tmp_path):
    # Pre-implementation — expected to fail until code-builder adds collect_features to
    # scripts/generate_features.py.
    (tmp_path / "integration").mkdir()
    (tmp_path / "e2e").mkdir()

    (tmp_path / "integration" / "test_foo_endpoint.py").write_text(
        "def test_bar():\n"
        "    pass\n"
        "\n"
        "def test_baz():\n"
        "    pass\n"
    )
    (tmp_path / "e2e" / "test_login_page.py").write_text(
        "def test_google_button():\n"
        "    pass\n"
    )

    result = collect_features(tmp_path)

    assert len(result) == 2
    assert "Login" in result
    assert "Google button" in result["Login"]


def test_given_ordered_dict_with_two_feature_areas_when_render_features_markdown_called_then_produces_valid_markdown():
    features = OrderedDict([
        ("Authentication", ["Log in with Google", "Reject expired token"]),
        ("Analysis Pipeline", ["Fetch market data", "Score sentiment"]),
    ])

    output = render_features_markdown(features)

    assert output.startswith("# QSent Features")
    assert "## Authentication\n" in output
    assert "## Analysis Pipeline\n" in output
    assert "- Log in with Google\n" in output
    assert "- Reject expired token\n" in output
    assert "- Fetch market data\n" in output
    assert "- Score sentiment\n" in output
    # Determinism: calling twice with the same input must return identical output
    assert render_features_markdown(features) == output


def test_check_docs_should_return_correct_result_for_missing_matching_and_differing_content(tmp_path):
    # Pre-implementation — expected to fail with ImportError until code-builder
    # adds check_docs to scripts/generate_features.py.
    docs_file = tmp_path / "FEATURES.md"
    generated = "# QSent Features\n\n## Auth\n\n- Log in\n"

    # Case 1: file does not exist
    assert check_docs(docs_file, generated) is False

    # Case 2: file exists and contents match exactly
    docs_file.write_text(generated)
    assert check_docs(docs_file, generated) is True

    # Case 3: file exists but contents differ
    docs_file.write_text(generated + "\nextra line\n")
    assert check_docs(docs_file, generated) is False
