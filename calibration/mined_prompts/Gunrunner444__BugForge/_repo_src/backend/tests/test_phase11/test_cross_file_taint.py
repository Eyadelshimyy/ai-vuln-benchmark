"""Cross-file taint and adversarial flow cases.

Static observations stay potential. Profile fallback is not dataflow.
Ambiguous and unresolved imports do not invent edges.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import settings
from app.domain.findings import FindingStatus
from app.domain.language import ParserTier
from app.domain.security import VulnerabilityClass
from app.parsing.engine import parse_source
from app.security.cross_file import build_project
from app.security.engine import SecurityAnalysisEngine


def _scan(tmp_path: Path, files: dict[str, str]):
    paths: list[Path] = []
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        paths.append(path)
    return SecurityAnalysisEngine().analyze_repository(tmp_path, paths)


def _of(result, vuln: VulnerabilityClass, filename: str | None = None) -> list[int]:
    return sorted(
        obs.line
        for obs in result.observations
        if obs.vulnerability_class is vuln
        and (filename is None or obs.file_path.endswith(filename))
    )


def test_cross_file_source_return_reaches_sink(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "request_data.py": 'def get_user_input(request):\n    return request.args.get("q")\n',
            "handler.py": (
                "from request_data import get_user_input\n\n"
                "def handle(request):\n"
                "    q = get_user_input(request)\n"
                "    eval(q)\n"
            ),
        },
    )
    assert _of(result, VulnerabilityClass.DYNAMIC_EXECUTION, "handler.py") == [5]
    obs = next(o for o in result.observations if o.file_path.endswith("handler.py"))
    assert obs.metadata["taint_scope"] == "cross_file"
    assert obs.metadata["taint_precision"] == "flow-sensitive"
    assert all(f.status is not FindingStatus.VERIFIED for f in result.findings)


def test_cross_file_forward_return_and_later_clear(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "helpers.py": "def forward(value):\n    return value\n",
            "app.py": (
                "from helpers import forward\n"
                "q = request.args.get('q')\n"
                "b = forward(q)\n"
                "eval(b)\n"
                "b = 'safe'\n"
                "eval(b)\n"
            ),
        },
    )
    assert _of(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [4]


def test_cross_file_wrapper_sink_and_constant_argument(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "helpers.py": "def run_code(value):\n    eval(value)\n",
            "app.py": (
                "from helpers import run_code\nrun_code(request.args.get('q'))\nrun_code('safe')\n"
            ),
        },
    )
    assert _of(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [2]
    obs = next(o for o in result.observations if o.line == 2 and o.file_path.endswith("app.py"))
    assert obs.metadata["taint_scope"] == "cross_file"
    assert obs.metadata["callee_file"].endswith("helpers.py")
    assert _of(result, VulnerabilityClass.DYNAMIC_EXECUTION, "helpers.py") == []


def test_cross_file_sanitizer_is_kind_specific(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "helpers.py": "def clean(value):\n    return html.escape(value)\n",
            "app.py": (
                "from helpers import clean\n"
                "raw = request.args.get('q')\n"
                "q = clean(raw)\n"
                "markupsafe.Markup(q)\n"
                "eval(q)\n"
                "q = request.args.get('q')\n"
                "markupsafe.Markup(q)\n"
            ),
        },
    )
    assert _of(result, VulnerabilityClass.XSS, "app.py") == [7]
    assert _of(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [5]


def test_unknown_call_does_not_return_taint(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "helpers.py": "def nope(value):\n    return 'safe'\n",
            "app.py": (
                "from helpers import nope\n"
                "q = request.args.get('q')\n"
                "b = nope(q)\n"
                "eval(b)\n"
                "eval(q)\n"
            ),
        },
    )
    assert _of(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [5]


def test_two_hop_wrapper_and_depth_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = {
        "c.py": "def run_code(value):\n    eval(value)\n",
        "b.py": "from c import run_code\n\ndef wrap(value):\n    run_code(value)\n",
        "a.py": "from b import wrap\n\nwrap(request.args.get('q'))\n",
    }
    result = _scan(tmp_path, files)
    assert _of(result, VulnerabilityClass.DYNAMIC_EXECUTION, "a.py") == [3]

    monkeypatch.setattr(settings, "taint_max_import_depth", 1)
    limited = _scan(tmp_path / "limited", files)
    assert _of(limited, VulnerabilityClass.DYNAMIC_EXECUTION, "a.py") == []
    assert any(item.kind == "cross_file_depth_limited" for item in limited.diagnostics)


def test_same_name_in_different_modules(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "pkg/a.py": "def run_code(value):\n    eval(value)\n",
            "pkg/b.py": "def run_code(value):\n    return value\n",
            "app.py": (
                "from pkg.a import run_code\n"
                "from pkg.b import run_code as other\n"
                "run_code(request.args.get('q'))\n"
                "value = other(request.args.get('q'))\n"
                "eval(value)\n"
            ),
        },
    )
    assert _of(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [3, 5]


def test_ambiguous_and_unresolved_imports_do_not_flow(tmp_path: Path) -> None:
    ambiguous = _scan(
        tmp_path / "amb",
        {
            "left/helpers.py": "def run_code(value):\n    eval(value)\n",
            "right/helpers.py": "def run_code(value):\n    eval(value)\n",
            "app.py": "from helpers import run_code\nrun_code(request.args.get('q'))\n",
        },
    )
    assert _of(ambiguous, VulnerabilityClass.DYNAMIC_EXECUTION) == []
    assert any(item.kind == "ambiguous_import" for item in ambiguous.diagnostics)

    missing = _scan(
        tmp_path / "miss",
        {"app.py": "from .missing import run_code\nrun_code(request.args.get('q'))\n"},
    )
    assert _of(missing, VulnerabilityClass.DYNAMIC_EXECUTION) == []
    assert any(item.kind == "unresolved_import" for item in missing.diagnostics)


def test_relative_import_and_js_named_import(tmp_path: Path) -> None:
    py = _scan(
        tmp_path / "py",
        {
            "pkg/helpers.py": "def run_code(value):\n    eval(value)\n",
            "pkg/app.py": "from .helpers import run_code\nrun_code(request.args.get('q'))\n",
        },
    )
    assert _of(py, VulnerabilityClass.DYNAMIC_EXECUTION, "pkg/app.py") == [2]

    js = _scan(
        tmp_path / "js",
        {
            "helpers.js": "function runCode(value) {\n  eval(value);\n}\n",
            "app.js": 'import { runCode } from "./helpers.js";\nrunCode(req.query.q);\n',
        },
    )
    assert _of(js, VulnerabilityClass.DYNAMIC_EXECUTION, "app.js") == [2]
    safe = _scan(
        tmp_path / "js-safe",
        {
            "helpers.js": "function runCode(value) {\n  eval(value);\n}\n",
            "app.js": 'import { runCode } from "./helpers.js";\nrunCode("safe");\n',
        },
    )
    assert _of(safe, VulnerabilityClass.DYNAMIC_EXECUTION, "app.js") == []


def test_wrong_argument_is_not_a_sink(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "helpers.py": "def run_code(label, value):\n    eval(value)\n",
            "app.py": (
                "from helpers import run_code\n"
                "run_code(request.args.get('q'), 'safe')\n"
                "run_code('ok', request.args.get('q'))\n"
            ),
        },
    )
    assert _of(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [3]


@pytest.mark.parametrize(
    ("helper", "caller", "vuln"),
    [
        (
            "def run_query(value):\n    cursor.execute(value)\n",
            "run_query",
            VulnerabilityClass.SQL_INJECTION,
        ),
        (
            "def run_cmd(value):\n    os.system(value)\n",
            "run_cmd",
            VulnerabilityClass.COMMAND_INJECTION,
        ),
        (
            "def read(value):\n    open(value)\n",
            "read",
            VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL,
        ),
        ("def fetch(value):\n    requests.get(value)\n", "fetch", VulnerabilityClass.SSRF),
        (
            "def render(value):\n    markupsafe.Markup(value)\n",
            "render",
            VulnerabilityClass.XSS,
        ),
        (
            "def load(value):\n    pickle.loads(value)\n",
            "load",
            VulnerabilityClass.UNSAFE_DESERIALIZATION,
        ),
        (
            "def run_code(value):\n    eval(value)\n",
            "run_code",
            VulnerabilityClass.DYNAMIC_EXECUTION,
        ),
        ("def go(value):\n    redirect(value)\n", "go", VulnerabilityClass.UNSAFE_REDIRECT),
    ],
)
def test_cross_file_categories(
    tmp_path: Path, helper: str, caller: str, vuln: VulnerabilityClass
) -> None:
    positive = _scan(
        tmp_path / "pos",
        {
            "helpers.py": helper,
            "app.py": f"from helpers import {caller}\n{caller}(request.args.get('q'))\n",
        },
    )
    assert _of(positive, vuln, "app.py"), (vuln, positive.observations)
    negative = _scan(
        tmp_path / "neg",
        {
            "helpers.py": helper,
            "app.py": f"from helpers import {caller}\n{caller}('safe')\n",
        },
    )
    assert _of(negative, vuln, "app.py") == []


def test_loop_and_reassignment_sequences(tmp_path: Path) -> None:
    tainted_loop = _scan(
        tmp_path / "a",
        {"a.py": "q = 'safe'\nfor i in range(2):\n    q = request.args.get('q')\neval(q)\n"},
    )
    assert _of(tainted_loop, VulnerabilityClass.DYNAMIC_EXECUTION) == [4]
    cleared_in_loop = _scan(
        tmp_path / "b",
        {"b.py": "q = request.args.get('q')\nfor i in range(2):\n    q = 'safe'\neval(q)\n"},
    )
    assert _of(cleared_in_loop, VulnerabilityClass.DYNAMIC_EXECUTION) == [4]
    sequence = _scan(
        tmp_path / "c",
        {
            "c.py": (
                "q = 'safe'\n"
                "eval(q)\n"
                "q = request.args.get('q')\n"
                "q = 'safe'\n"
                "q = request.args.get('q')\n"
                "eval(q)\n"
            )
        },
    )
    assert _of(sequence, VulnerabilityClass.DYNAMIC_EXECUTION) == [6]


def test_sibling_modules_do_not_share_locals(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "a.py": "def f():\n    q = request.args.get('q')\n    eval(q)\n",
            "b.py": "def f():\n    q = 'safe'\n    eval(q)\n",
        },
    )
    assert _of(result, VulnerabilityClass.DYNAMIC_EXECUTION, "a.py") == [3]
    assert _of(result, VulnerabilityClass.DYNAMIC_EXECUTION, "b.py") == []


def test_profile_fallback_is_not_a_cross_file_callee() -> None:
    helper = parse_source("python", Path("helpers.py"), "def run_code(value):\n    eval(value)\n")
    app = parse_source(
        "python",
        Path("app.py"),
        "from helpers import run_code\nrun_code(request.args.get('q'))\n",
    )
    helper.file_path = "helpers.py"
    app.file_path = "app.py"
    helper.parser_tier = ParserTier.PROFILE_FALLBACK
    project = build_project({"helpers.py": helper, "app.py": app})
    assert project.externals == {}


def test_cross_file_budget_keeps_local_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "taint_cross_file_budget_ms", 0)
    result = _scan(
        tmp_path,
        {
            "helpers.py": "def run_code(value):\n    eval(value)\n",
            "app.py": "from helpers import run_code\nrun_code(request.args.get('q'))\nq = request.args.get('q')\neval(q)\n",
        },
    )
    assert any(item.kind == "cross_file_incomplete" for item in result.diagnostics)
    assert _of(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [4]
    assert 2 not in _of(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py")


def test_analysis_is_deterministic(tmp_path: Path) -> None:
    files = {
        "helpers.py": "def forward(value):\n    return value\n",
        "app.py": "from helpers import forward\nq = forward(request.args.get('q'))\neval(q)\n",
    }
    first = _scan(tmp_path / "one", files)
    second = _scan(tmp_path / "two", files)

    def key(obs) -> tuple:
        return (
            obs.file_path,
            obs.line,
            obs.rule_id,
            obs.vulnerability_class.value,
            obs.taint_path,
            obs.node_id,
        )

    assert sorted(key(obs) for obs in first.observations) == sorted(
        key(obs) for obs in second.observations
    )


def test_malformed_python_is_not_a_clean_result(tmp_path: Path) -> None:
    result = _scan(tmp_path, {"broken.py": "def broken(:\n    eval(request.args.get('q'))\n"})
    assert result.diagnostics
    assert any(
        item.kind in {"parser_failure", "parser_error", "partial_analysis"}
        for item in result.diagnostics
    )


def test_js_named_import_bindings() -> None:
    graph = parse_source(
        "javascript",
        Path("app.js"),
        'import { forward, execute as run } from "./helpers.js";\n',
    )
    names = {(imp.name, imp.alias) for imp in graph.imports}
    assert ("forward", None) in names
    assert ("execute", "run") in names
    assert graph.parser_backend == "tree_sitter"
    assert graph.parser_tier is ParserTier.FULL_AST
