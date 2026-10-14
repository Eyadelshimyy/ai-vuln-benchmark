"""Phase 19 local interprocedural flow for Go, Java, and Kotlin.

Only a unique local file is a callee. Profile fallback, dynamic imports, and
languages without a reliable import identity stay unresolved.
"""

from __future__ import annotations

from pathlib import Path

from app.domain.findings import FindingStatus
from app.security.cross_file import build_project
from app.security.engine import SecurityAnalysisEngine


def _scan(root: Path, files: dict[str, str]):
    paths = []
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        paths.append(path)
    return SecurityAnalysisEngine().analyze_repository(root, paths)


def _classes(result, file_name: str) -> set[str]:
    return {
        obs.vulnerability_class.value
        for obs in result.observations
        if obs.file_path.replace("\\", "/") == file_name
    }


def test_go_package_import_reaches_command(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "helpers.go": "package helpers\nfunc Run(value string) {\n    exec.Command(value)\n}\n",
            "app.go": (
                'package main\nimport "helpers"\n'
                'func main() {\n    helpers.Run(r.FormValue("q"))\n}\n'
            ),
        },
    )
    assert "command_injection" in _classes(result, "app.go")
    assert all(item.status is not FindingStatus.VERIFIED for item in result.findings)
    assert all(graph.parser_backend == "tree_sitter" for graph in result.graphs.values())
    assert all(str(graph.parser_tier) == "full_ast" for graph in result.graphs.values())


def test_go_explicit_alias(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "proj/helpers.go": "package helpers\nfunc Run(value string) {\n    exec.Command(value)\n}\n",
            "app.go": (
                'package main\nimport h "proj/helpers"\n'
                'func main() {\n    h.Run(r.FormValue("q"))\n}\n'
            ),
        },
    )
    assert "command_injection" in _classes(result, "app.go")


def test_go_dot_blank_and_third_party_stay_unresolved(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "helpers.go": "package helpers\nfunc Run(value string) {\n    exec.Command(value)\n}\n",
            "app.go": (
                'package main\nimport (\n    . "helpers"\n    _ "helpers"\n    "os/exec"\n)\n'
                'func main() {\n    Run(r.FormValue("q"))\n}\n'
            ),
        },
    )
    assert "command_injection" not in _classes(result, "app.go")
    project = build_project(result.graphs, repo_root=tmp_path)
    assert not any("os/exec" in f"{edge.source_id} {edge.target_id}" for edge in project.edges)


def test_go_ambiguous_package_has_no_edge(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "a/helpers.go": "package helpers\nfunc Run(value string) {\n    exec.Command(value)\n}\n",
            "b/helpers.go": "package helpers\nfunc Run(value string) {\n    exec.Command(value)\n}\n",
            "app.go": (
                'package main\nimport "helpers"\n'
                'func main() {\n    helpers.Run(r.FormValue("q"))\n}\n'
            ),
        },
    )
    assert "command_injection" not in _classes(result, "app.go")
    project = build_project(result.graphs, repo_root=tmp_path)
    assert any(item.kind == "ambiguous_import" for item in project.diagnostics)


def test_go_partial_callee_is_not_a_summary(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "helpers.go": "package helpers\nfunc Run( {\n    exec.Command(value)\n}\n",
            "app.go": (
                'package main\nimport "helpers"\n'
                'func main() {\n    helpers.Run(r.FormValue("q"))\n}\n'
            ),
        },
    )
    assert "command_injection" not in _classes(result, "app.go")
    project = build_project(result.graphs, repo_root=tmp_path)
    assert any(item.kind == "cross_file_partial" for item in project.diagnostics)


def test_go_html_escape_does_not_clear_command(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "helpers.go": (
                "package helpers\nfunc Run(value string) {\n"
                "    safe := html.EscapeString(value)\n"
                "    template.HTML(safe)\n"
                "    exec.Command(value)\n}\n"
            ),
            "app.go": (
                'package main\nimport "helpers"\n'
                'func main() {\n    helpers.Run(r.FormValue("q"))\n}\n'
            ),
        },
    )
    assert "command_injection" in _classes(result, "app.go")
    assert "xss" not in _classes(result, "app.go")


def test_java_type_import_method(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "pkg/Helper.java": (
                "package pkg;\npublic class Helper {\n"
                "    public static void run(String value) {\n"
                "        Runtime.getRuntime().exec(value);\n"
                "    }\n}\n"
            ),
            "App.java": (
                "import pkg.Helper;\npublic class App {\n"
                "    void item() {\n"
                '        Helper.run(request.getParameter("q"));\n'
                "    }\n}\n"
            ),
        },
    )
    assert "command_injection" in _classes(result, "App.java")
    assert all(item.status is not FindingStatus.VERIFIED for item in result.findings)


def test_java_static_import_stays_unresolved(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "pkg/Helper.java": (
                "package pkg;\npublic class Helper {\n"
                "    public static void run(String value) {\n"
                "        Runtime.getRuntime().exec(value);\n"
                "    }\n}\n"
            ),
            "App.java": (
                "import static pkg.Helper.run;\npublic class App {\n"
                "    void item() {\n"
                '        run(request.getParameter("q"));\n'
                "    }\n}\n"
            ),
        },
    )
    assert "command_injection" not in _classes(result, "App.java")


def test_kotlin_function_import_and_alias(tmp_path: Path) -> None:
    direct = _scan(
        tmp_path / "direct",
        {
            "pkg/run.kt": (
                "package pkg\nfun run(value: String) {\n    Runtime.getRuntime().exec(value)\n}\n"
            ),
            "App.kt": ('import pkg.run\nfun item() {\n    run(call.parameters["q"])\n}\n'),
        },
    )
    assert "command_injection" in _classes(direct, "App.kt")
    aliased = _scan(
        tmp_path / "alias",
        {
            "pkg/run.kt": (
                "package pkg\nfun run(value: String) {\n    Runtime.getRuntime().exec(value)\n}\n"
            ),
            "App.kt": (
                'import pkg.run as runCode\nfun item() {\n    runCode(call.parameters["q"])\n}\n'
            ),
        },
    )
    assert "command_injection" in _classes(aliased, "App.kt")


def test_ruby_csharp_and_swift_are_not_file_imports(tmp_path: Path) -> None:
    ruby = _scan(
        tmp_path / "ruby",
        {
            "helpers.rb": "def run(value)\n  system(value)\nend\n",
            "app.rb": 'require_relative "helpers"\nhelpers.run(params[:q])\n',
        },
    )
    assert "command_injection" not in _classes(ruby, "app.rb")
    csharp = _scan(
        tmp_path / "cs",
        {
            "Helper.cs": "class Helper { void Run(string value) { Process.Start(value); } }\n",
            "App.cs": 'using Helper;\nclass App { void Item() { Helper.Run(Request.Query["q"]); } }\n',
        },
    )
    assert "command_injection" not in _classes(csharp, "App.cs")
    swift = _scan(
        tmp_path / "swift",
        {
            "Helper.swift": "func run(value: String) {}\n",
            "App.swift": "import Helper\nfunc item() { Helper.run(request) }\n",
        },
    )
    assert not _classes(swift, "App.swift")
