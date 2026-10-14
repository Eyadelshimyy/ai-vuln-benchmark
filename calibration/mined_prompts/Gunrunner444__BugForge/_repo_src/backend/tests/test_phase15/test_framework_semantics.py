"""Phase 15 framework-aware sources, sinks, and routes.

Route facts come from syntax. Unknown frameworks and ordinary names stay
unknown. Static findings stay potential.
"""

from __future__ import annotations

from pathlib import Path

from app.domain.findings import FindingStatus
from app.domain.security import VulnerabilityClass
from app.parsing.engine import parse_source
from app.security.engine import SecurityAnalysisEngine

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "phase15_repo"
EVAL = VulnerabilityClass.DYNAMIC_EXECUTION
SQL = VulnerabilityClass.SQL_INJECTION
XSS = VulnerabilityClass.XSS
REDIR = VulnerabilityClass.UNSAFE_REDIRECT


def _scan(root: Path):
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    return SecurityAnalysisEngine().analyze_repository(root, paths)


def _lines(result, vuln: VulnerabilityClass, filename: str) -> list[int]:
    return sorted(
        obs.line
        for obs in result.observations
        if obs.vulnerability_class is vuln and str(obs.file_path).endswith(filename)
    )


def test_a_variable_named_request_is_not_a_source() -> None:
    result = _scan(FIXTURE / "named_request")
    assert result.observations == []


def test_real_request_attributes_are_sources() -> None:
    assert _lines(_scan(FIXTURE / "flask_source"), EVAL, "app.py") == [2]
    assert _lines(_scan(FIXTURE / "django_source"), EVAL, "app.py") == [2]


def test_explicit_route_parameters_are_sources_and_plain_functions_are_not() -> None:
    routed = _scan(FIXTURE / "route_param")
    assert _lines(routed, EVAL, "app.py") == [8]
    obs = routed.observations[0]
    assert obs.metadata["route"] == "/item/{id}"
    assert obs.metadata["route_method"] == "GET"
    assert obs.metadata["endpoint"] == "item"
    assert (
        obs.metadata["taint"].startswith("source:route:id")
        or "source:route:id" in obs.metadata["taint"]
    )

    flask = _scan(FIXTURE / "flask_route")
    assert _lines(flask, EVAL, "app.py") == [8]
    assert flask.observations[0].metadata["route"] == "/item/<int:id>"
    assert flask.observations[0].metadata["route_method"] == "GET"

    assert _scan(FIXTURE / "plain_function").observations == []
    assert _scan(FIXTURE / "cache_get").observations == []
    cache = parse_source(
        "python", FIXTURE / "cache_get" / "app.py", (FIXTURE / "cache_get" / "app.py").read_text()
    )
    assert cache.routes == ()


def test_sql_distinguishes_parameters_builders_and_dynamic_text() -> None:
    assert _lines(_scan(FIXTURE / "parameterized_sql"), SQL, "app.py") == []
    assert _lines(_scan(FIXTURE / "dynamic_sql"), SQL, "app.py") == [3]
    assert _lines(_scan(FIXTURE / "sqlalchemy_safe"), SQL, "app.py") == []
    assert _lines(_scan(FIXTURE / "sqlalchemy_unsafe"), SQL, "app.py") == [3]
    assert _lines(_scan(FIXTURE / "orm_filter"), SQL, "app.py") == []
    assert _lines(_scan(FIXTURE / "orm_raw"), SQL, "app.py") == [3, 4]


def test_templates_xss_and_redirects_stay_vulnerability_specific() -> None:
    assert _scan(FIXTURE / "template_safe").observations == []
    assert _lines(_scan(FIXTURE / "template_string"), EVAL, "app.py") == [2]
    xss = _scan(FIXTURE / "xss_escape")
    assert _lines(xss, XSS, "app.py") == [7]
    redirects = _scan(FIXTURE / "redirects")
    assert _lines(redirects, REDIR, "app.py") == [3]
    assert all(item.status is not FindingStatus.VERIFIED for item in redirects.findings)


def test_unknown_framework_and_non_route_calls_stay_unresolved() -> None:
    assert _scan(FIXTURE / "unknown_framework").observations == []
    bare = _scan(FIXTURE / "bare_get")
    assert bare.observations == []
    assert bare.graphs["app.js"].routes == ()
    express = _scan(FIXTURE / "express_route")
    assert _lines(express, EVAL, "app.js") == [4]
    assert express.graphs["app.js"].routes[0].path == "/search"
    assert express.graphs["app.js"].routes[0].method == "GET"
    callback = _scan(FIXTURE / "express_callback")
    assert callback.observations == []
    assert callback.graphs["app.js"].routes[0].path == "/item/:id"


def test_route_facts_are_deterministic() -> None:
    source = (FIXTURE / "route_param" / "app.py").read_text()
    first = parse_source("python", Path("app.py"), source)
    second = parse_source("python", Path("app.py"), source)
    assert first.routes == second.routes
    assert len(first.routes) == 1
