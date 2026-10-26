"""The coverage gate decides whether a green scan actually attacked anything."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dast_coverage  # noqa: E402
from dast_coverage import SWEEP, coverage, load_exemptions, matches  # noqa: E402
from route_inventory import Route  # noqa: E402

DOCUMENT = Route("GET", "/api/v1/documents/{}", "document-service", "app/api/documents.py")
LIST = Route("GET", "/api/v1/documents", "document-service", "app/api/documents.py")


def test_a_concrete_request_matches_its_declared_template() -> None:
    assert matches(DOCUMENT, "GET", "/api/v1/documents/9f1c2a04-0000-4000-8000-000000000000")
    assert matches(DOCUMENT, "GET", "/api/v1/documents/anything/")


def test_a_request_does_not_match_another_route() -> None:
    assert not matches(DOCUMENT, "POST", "/api/v1/documents/abc")  # method
    assert not matches(DOCUMENT, "GET", "/api/v1/documents")  # too few segments
    assert not matches(DOCUMENT, "GET", "/api/v1/documents/abc/comments")  # too many
    assert not matches(LIST, "GET", "/api/v1/files")


def test_coverage_separates_reached_from_attacked_as_a_caller() -> None:
    exercised = [
        {
            "method": "GET",
            "path": "/api/v1/documents/abc",
            "authenticated": False,
            "probe": dast_coverage.SWEEP,
        },
        {
            "method": "GET",
            "path": "/api/v1/documents",
            "authenticated": True,
            "probe": "DAST-BOLA-DOCUMENTS",
        },
    ]
    reached, attacked, authenticated, missed = coverage([DOCUMENT, LIST], exercised)
    assert reached == [DOCUMENT, LIST]
    # Swept anonymously is not the same as attacked: the sweep walks the same
    # inventory this gate reads, so counting its own requests as coverage would
    # measure nothing but itself.
    assert attacked == [LIST]
    assert authenticated == [LIST]
    assert missed == []


def test_a_report_without_probe_attribution_is_not_credited_as_attacked() -> None:
    """An older report cannot tell the sweep apart, so it claims the lower depth."""
    exercised = [{"method": "GET", "path": "/api/v1/documents", "authenticated": True}]
    reached, attacked, authenticated, _ = coverage([LIST], exercised)
    assert (reached, attacked, authenticated) == ([LIST], [], [LIST])


def test_a_route_nothing_requested_is_missed() -> None:
    *_, missed = coverage([DOCUMENT, LIST], [{"method": "GET", "path": "/api/v1/documents"}])
    assert missed == [DOCUMENT]


def test_exemptions_are_keyed_by_method_and_path(tmp_path: Path) -> None:
    spec = tmp_path / "attack-surface.yaml"
    spec.write_text(
        "coverage_exemptions:\n"
        "  - method: get\n"
        "    path: /api/v1/documents/{}\n"
        "    reason: covered by the contract suite\n"
    )
    assert load_exemptions(spec) == {"GET /api/v1/documents/{}": "covered by the contract suite"}


def test_no_report_is_a_setup_failure_not_a_pass(tmp_path: Path) -> None:
    assert dast_coverage.main(["--report", str(tmp_path / "absent.json")]) == 2


def test_a_report_without_recorded_requests_is_a_setup_failure(tmp_path: Path) -> None:
    report = tmp_path / "dast-report.json"
    report.write_text(json.dumps({"target": "http://localhost:8080", "results": []}))
    # A scan from before request recording has no coverage information at all;
    # reading that as full coverage is exactly the false pass this gate prevents.
    assert dast_coverage.main(["--report", str(report)]) == 2


def _report(tmp_path: Path, exercised: list[dict[str, object]], **extra: object) -> list[str]:
    report = tmp_path / "dast-report.json"
    report.write_text(
        json.dumps({"target": "http://localhost:8080", "exercised": exercised, **extra})
    )
    return ["--report", str(report)]


def test_an_unattacked_route_fails_the_gate(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(dast_coverage, "edge_routes", lambda: ([DOCUMENT, LIST], {}))
    monkeypatch.setattr(dast_coverage, "load_exemptions", dict)
    argv = _report(tmp_path, [{"method": "GET", "path": "/api/v1/documents"}])
    assert dast_coverage.main(argv) == 1
    capsys.readouterr()
    assert dast_coverage.main([*argv, "--warn-only"]) == 0
    # Warning is not passing: the summary line must not contradict the rows above it.
    assert "PASSED" not in capsys.readouterr().out


def test_a_single_probe_report_cannot_be_graded_for_coverage(tmp_path: Path, monkeypatch) -> None:
    """`make dast-verify` writes this same report having attacked one finding."""
    monkeypatch.setattr(dast_coverage, "edge_routes", lambda: ([DOCUMENT, LIST], {}))
    monkeypatch.setattr(dast_coverage, "load_exemptions", dict)
    report = tmp_path / "dast-report.json"
    report.write_text(
        json.dumps({"target": "http://localhost:8080", "partial": True, "exercised": []})
    )
    assert dast_coverage.main(["--report", str(report)]) == 2


def test_an_exempted_route_passes_the_gate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dast_coverage, "edge_routes", lambda: ([DOCUMENT, LIST], {}))
    monkeypatch.setattr(
        dast_coverage, "load_exemptions", lambda: {DOCUMENT.key: "needs a seeded document"}
    )
    argv = _report(tmp_path, [{"method": "GET", "path": "/api/v1/documents"}])
    assert dast_coverage.main(argv) == 0


def test_unknown_services_are_named_even_when_the_gate_passes(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(dast_coverage, "edge_routes", lambda: ([LIST], {"/api/v1/admin": "admin"}))
    monkeypatch.setattr(dast_coverage, "load_exemptions", dict)
    argv = _report(tmp_path, [{"method": "GET", "path": "/api/v1/documents"}])
    assert dast_coverage.main(argv) == 0
    assert "/api/v1/admin" in capsys.readouterr().out


SEARCH = Route("GET", "/api/v1/documents/search", "document-service", "app/api/documents.py")


def test_a_literal_route_does_not_cover_its_parameterized_sibling() -> None:
    """A request reaches one handler, so it can only be credited to one route."""
    exercised = [
        {
            "method": "GET",
            "path": "/api/v1/documents/search",
            "authenticated": True,
            "probe": "DAST-SEARCH-LEAK",
        }
    ]
    reached, _, _, missed = coverage([DOCUMENT, SEARCH], exercised)
    assert reached == [SEARCH]
    assert missed == [DOCUMENT]


def test_a_parameterized_request_still_lands_on_the_template() -> None:
    exercised = [
        {
            "method": "GET",
            "path": "/api/v1/documents/9f1c2a04",
            "authenticated": True,
            "probe": "DAST-BOLA-DOCUMENTS",
        }
    ]
    reached, _, _, missed = coverage([DOCUMENT, SEARCH], exercised)
    assert reached == [DOCUMENT]
    assert missed == [SEARCH]


def test_a_request_the_target_never_answered_is_not_coverage() -> None:
    """A 502 or a redirect was answered by something short of the handler."""
    for status in (502, 429, 307):
        exercised = [
            {
                "method": "GET",
                "path": "/api/v1/documents/9f1c2a04",
                "authenticated": True,
                "probe": "DAST-BOLA-DOCUMENTS",
                "status": status,
            }
        ]
        reached, _, _, missed = coverage([DOCUMENT], exercised)
        assert (reached, missed) == ([], [DOCUMENT]), status


def test_a_refusal_is_still_coverage() -> None:
    """401 is the handler's chain answering: the route was attacked and held."""
    exercised = [
        {
            "method": "GET",
            "path": "/api/v1/documents/9f1c2a04",
            "authenticated": False,
            "probe": "DAST-ANONYMOUS-ROUTE-SWEEP",
            "status": 401,
        }
    ]
    reached, _, _, missed = coverage([DOCUMENT], exercised)
    assert (reached, missed) == ([DOCUMENT], [])


WRITE = Route("DELETE", "/api/v1/documents/{}", "document-service", "app/api/documents.py")


def test_a_write_route_the_scan_did_sweep_is_still_gated(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The excuse belongs to the scan that withheld the route, not to the grader.

    Grading runs in its own step in CI, so reading DAST_SWEEP_UNSAFE_METHODS here
    would excuse every write route the scan had in fact swept — and with it any
    write route nothing covers, which is the case the gate exists for.
    """
    monkeypatch.setattr(dast_coverage, "edge_routes", lambda: ([WRITE], {}))
    monkeypatch.setattr(dast_coverage, "load_exemptions", dict)
    monkeypatch.delenv("DAST_SWEEP_UNSAFE_METHODS", raising=False)
    assert dast_coverage.main(_report(tmp_path, [], swept_unsafely=True)) == 1
    capsys.readouterr()
    assert dast_coverage.main(_report(tmp_path, [], swept_unsafely=False)) == 0
    assert "not declared" in capsys.readouterr().out


NOTIFY = Route("GET", "/api/v1/notifications", "notification-service", "Routing.kt")


def test_a_route_the_target_could_not_answer_is_named_but_not_gated(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """CI runs five services of eleven, so the gateway 502s the rest.

    That is the target's shape rather than a probe nobody wrote, so it must not
    fail the build — but it cannot be counted as covered either.
    """
    monkeypatch.setattr(dast_coverage, "edge_routes", lambda: ([NOTIFY, LIST], {}))
    monkeypatch.setattr(dast_coverage, "load_exemptions", dict)
    argv = _report(
        tmp_path,
        [
            {"method": "GET", "path": "/api/v1/notifications", "probe": SWEEP, "status": 502},
            {"method": "GET", "path": "/api/v1/documents", "probe": SWEEP, "status": 200},
        ],
    )
    assert dast_coverage.main(argv) == 0
    out = capsys.readouterr().out
    assert "UNANSWERED" in out
    assert "reached by a probe:              1/2" in out


def test_a_route_nothing_requested_at_all_still_fails(tmp_path: Path, monkeypatch) -> None:
    """The unanswered excuse needs a request behind it; silence is the gap."""
    monkeypatch.setattr(dast_coverage, "edge_routes", lambda: ([NOTIFY], {}))
    monkeypatch.setattr(dast_coverage, "load_exemptions", dict)
    assert dast_coverage.main(_report(tmp_path, [])) == 1


def test_a_route_declared_twice_is_covered_once() -> None:
    """Two files declaring one endpoint are two entries but one attack surface."""
    twin = Route("GET", "/api/v1/documents/{}", "document-service", "app/api/legacy.py")
    exercised = [
        {
            "method": "GET",
            "path": "/api/v1/documents/9f1c2a04",
            "authenticated": True,
            "probe": "DAST-BOLA-DOCUMENTS",
            "status": 200,
        }
    ]
    reached, _, _, missed = coverage([DOCUMENT, twin], exercised)
    assert reached == [DOCUMENT, twin]
    assert missed == []


def test_a_redirect_followed_to_an_answer_is_not_reported_unanswered(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The sweep records the 307 and then follows it, so both statuses are there."""
    monkeypatch.setattr(dast_coverage, "edge_routes", lambda: ([NOTIFY], {}))
    monkeypatch.setattr(dast_coverage, "load_exemptions", dict)
    argv = _report(
        tmp_path,
        [
            {"method": "GET", "path": "/api/v1/notifications", "probe": SWEEP, "status": 307},
            {"method": "GET", "path": "/api/v1/notifications", "probe": SWEEP, "status": 200},
        ],
    )
    assert dast_coverage.main(argv) == 0
    assert "UNANSWERED" not in capsys.readouterr().out


def test_a_documented_write_route_keeps_the_reason_its_owner_wrote(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The blanket "not declared disposable" note must not bury real debt."""
    reindex = Route("POST", "/api/v1/search/reindex", "search-service", "app/api/search.py")
    monkeypatch.setattr(dast_coverage, "edge_routes", lambda: ([reindex], {}))
    monkeypatch.setattr(
        dast_coverage,
        "load_exemptions",
        lambda: {reindex.key: "rebuilds the whole index; needs a probe that can undo it"},
    )
    argv = _report(tmp_path, [], swept_unsafely=False)
    assert dast_coverage.main(argv) == 0
    assert "needs a probe that can undo it" in capsys.readouterr().out


def test_a_target_that_answered_nothing_is_not_a_pass(tmp_path: Path, monkeypatch) -> None:
    """Everything 502/429 is the gate measuring nothing, not a covered surface."""
    monkeypatch.setattr(dast_coverage, "edge_routes", lambda: ([NOTIFY, LIST], {}))
    monkeypatch.setattr(dast_coverage, "load_exemptions", dict)
    argv = _report(
        tmp_path,
        [
            {"method": "GET", "path": "/api/v1/notifications", "probe": SWEEP, "status": 502},
            {"method": "GET", "path": "/api/v1/documents", "probe": SWEEP, "status": 429},
        ],
    )
    assert dast_coverage.main(argv) == 2
