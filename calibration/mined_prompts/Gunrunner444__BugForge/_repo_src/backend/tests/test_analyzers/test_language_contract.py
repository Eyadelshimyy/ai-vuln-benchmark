"""Central language-analysis test contract.

Future languages must satisfy this module rather than shipping a single
smoke parse. Categories that a profile does not implement are asserted as
unsupported instead of being skipped silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.language import LanguageCapability
from app.parsing.profiles import profile_for
from app.plugins.errors import UnsupportedCapabilityError
from tests.language_support import (
    ANALYSIS_LANGUAGE_IDS,
    DETECTION_ONLY_LANGUAGE_IDS,
    EXTENSIONS,
    PARSE_CAPS,
    SPECIALIZED_LANGUAGE_IDS,
    adapter,
    graph_for,
    profile_supports,
)


@pytest.mark.parametrize("language_id", ANALYSIS_LANGUAGE_IDS)
def test_analysis_language_is_registered(language_id: str) -> None:
    lang = adapter(language_id)
    for cap in PARSE_CAPS:
        assert lang.supports(cap), f"{language_id} missing {cap}"
    assert lang.language_id == language_id


@pytest.mark.parametrize("language_id", ANALYSIS_LANGUAGE_IDS)
def test_analysis_language_extensions(language_id: str) -> None:
    lang = adapter(language_id)
    expected = set(EXTENSIONS[language_id])
    assert expected <= set(lang.file_extensions)
    for ext in expected:
        sample = Path(f"sample{ext}")
        assert lang.matches_path(sample)


@pytest.mark.parametrize("language_id", ANALYSIS_LANGUAGE_IDS)
def test_analysis_language_has_profile(language_id: str) -> None:
    profile = profile_for(language_id)
    assert profile is not None
    assert profile.language_id == language_id


@pytest.mark.parametrize("language_id", DETECTION_ONLY_LANGUAGE_IDS)
def test_detection_only_contract(language_id: str) -> None:
    lang = adapter(language_id)
    assert lang.supports(LanguageCapability.DETECTION)
    assert not lang.supports(LanguageCapability.PARSE)
    assert not lang.supports(LanguageCapability.SECURITY_ANALYSIS)
    assert not lang.supports(LanguageCapability.STATIC_ANALYSIS)
    with pytest.raises(UnsupportedCapabilityError):
        lang.parse_file(Path(f"x{next(iter(lang.file_extensions))}"))
    with pytest.raises(UnsupportedCapabilityError):
        lang.analyze_file(Path(f"x{next(iter(lang.file_extensions))}"), "x")


@pytest.mark.parametrize("language_id", ANALYSIS_LANGUAGE_IDS)
def test_empty_and_malformed_source(language_id: str) -> None:
    empty = graph_for(language_id, f"empty{EXTENSIONS[language_id][0]}", "")
    assert empty.language == language_id
    broken = graph_for(language_id, f"bad{EXTENSIONS[language_id][0]}", "((({{{")
    assert broken.language == language_id
    assert broken.diagnostics.has_errors or broken.errors


@pytest.mark.parametrize("language_id", SPECIALIZED_LANGUAGE_IDS)
def test_specialized_languages_parse(language_id: str) -> None:
    lang = adapter(language_id)
    assert lang.supports(LanguageCapability.PARSE)
    assert lang.supports(LanguageCapability.SECURITY_ANALYSIS)
    assert str(lang.parser_tier()) in {"specialized", "full_ast", "profile_fallback"}
    graph = graph_for(language_id, f"x{EXTENSIONS[language_id][0]}", "x { y }")
    assert graph.language == language_id
    assert graph.parser_backend in {"tree_sitter", "profile"}


@pytest.mark.parametrize("language_id", ANALYSIS_LANGUAGE_IDS)
def test_profile_category_support_is_explicit(language_id: str) -> None:
    profile = profile_for(language_id)
    assert profile is not None
    mapping = {
        "sql": profile.sql_sinks,
        "command": profile.command_sinks,
        "path": profile.path_sinks,
        "ssrf": profile.ssrf_sinks,
        "xss": profile.xss_sinks,
        "deser": profile.deser_sinks,
        "eval": profile.eval_sinks,
        "redirect": profile.redirect_sinks,
        "crypto": profile.crypto_patterns,
    }
    for category, attr in mapping.items():
        assert profile_supports(language_id, category) is bool(attr)
