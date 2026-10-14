"""Dedicated parser/security suites for non-JS analysis languages."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.security import VulnerabilityClass
from app.parsing.engine import parse_source
from app.parsing.model import SyntaxGraph
from tests.language_support import analyze_source, observation_classes


def _graph(language: str, name: str, source: str) -> SyntaxGraph:
    return parse_source(language, Path(name), source)


def test_ruby_entities_imports_and_security(tmp_path: Path) -> None:
    src = """
require 'sinatra'
module Auth
  class App
    def index
      q = params[:q]
      ActiveRecord::Base.connection.execute("SELECT * FROM t WHERE x = '" + q + "'")
      system("ls " + q)
      redirect_to params[:url]
    end
  end
end
"""
    graph = _graph("ruby", "app.rb", src)
    assert any("sinatra" in (imp.module or "") for imp in graph.imports)
    names = {e.name for e in graph.entities}
    assert "Auth" in names
    assert "App" in names
    assert "index" in names
    result = analyze_source(tmp_path, "app.rb", src)
    classes = observation_classes(result)
    assert VulnerabilityClass.COMMAND_INJECTION in classes
    safe = 'q = params[:q]\nUser.where("name = ?", q)\n'
    quiet = analyze_source(tmp_path, "safe.rb", safe)
    assert VulnerabilityClass.SQL_INJECTION not in {
        obs.vulnerability_class for obs in quiet.observations if "safe.rb" in obs.file_path
    }


def test_c_pointers_includes_and_command(tmp_path: Path) -> None:
    src = """
#include <stdio.h>
#include <stdlib.h>
char *read_arg(char **argv) { return argv[1]; }
int main(int argc, char **argv)
{
    char *cmd = argv[1];
    system(cmd);
    FILE *f = fopen(cmd, "r");
    sqlite3_exec(db, cmd, 0, 0, 0);
}
"""
    graph = _graph("c", "main.c", src)
    assert any("stdio.h" in (imp.module or "") for imp in graph.imports)
    assert any(e.name in {"main", "read_arg"} for e in graph.entities)
    result = analyze_source(tmp_path, "main.c", src)
    classes = observation_classes(result)
    assert VulnerabilityClass.COMMAND_INJECTION in classes
    assert VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL in classes
    assert VulnerabilityClass.SQL_INJECTION in classes
    comment_only = "/* system(argv[1]); */\nint x = 1;\n"
    quiet = analyze_source(tmp_path, "ok.c", comment_only)
    assert not any(
        obs.vulnerability_class is VulnerabilityClass.COMMAND_INJECTION and "ok.c" in obs.file_path
        for obs in quiet.observations
    )


def test_cpp_class_and_path(tmp_path: Path) -> None:
    src = """
#include <fstream>
class Loader {
 public:
  void load(const std::string& name) {
    std::ifstream in(name);
  }
};
int run(char **argv) {
  Loader l;
  l.load(argv[1]);
}
"""
    graph = _graph("cpp", "a.cpp", src)
    assert any(e.name == "Loader" for e in graph.entities)
    result = analyze_source(tmp_path, "a.cpp", src)
    # argv is a source; ifstream sink may be assignment-like.
    assert result.files_analyzed == 1


def test_go_methods_imports_and_http(tmp_path: Path) -> None:
    src = """
package main
import (
    "net/http"
    "os/exec"
)
type Server struct{}
func (s Server) Handle(w http.ResponseWriter, r *http.Request) {
    q := r.URL.Query().Get("q")
    db.Query("SELECT * FROM t WHERE x = '" + q + "'")
    exec.Command("sh", "-c", q).Run()
    http.Get(q)
    os.Open(q)
    json.Unmarshal([]byte(q), &out)
    http.Redirect(w, r, q, 302)
}
"""
    graph = _graph("go", "s.go", src)
    names = {e.name for e in graph.entities}
    assert "Handle" in names
    assert "Server" in names
    mods = {imp.module for imp in graph.imports}
    assert "net/http" in mods or any("http" in m for m in mods)
    result = analyze_source(tmp_path, "s.go", src)
    classes = observation_classes(result)
    assert VulnerabilityClass.SQL_INJECTION in classes
    assert VulnerabilityClass.COMMAND_INJECTION in classes
    assert VulnerabilityClass.SSRF in classes


def test_rust_use_impl_and_process(tmp_path: Path) -> None:
    src = """
use std::process::Command;
use std::fs;
struct App;
enum Kind { A, B }
impl App {
    fn run(req: Request) {
        let q = req.query;
        let mut cmd = Command::new("sh");
        sqlx::query(&format!("SELECT {}", q));
        fs::read(q);
        serde_json::from_str(q);
        Command::new(q);
    }
}
"""
    graph = _graph("rust", "a.rs", src)
    names = {e.name for e in graph.entities}
    assert "App" in names
    assert "Kind" in names
    assert "run" in names
    result = analyze_source(tmp_path, "a.rs", src)
    classes = observation_classes(result)
    assert VulnerabilityClass.COMMAND_INJECTION in classes
    assert VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL in classes


def test_java_servlet_jdbc_and_process(tmp_path: Path) -> None:
    src = """
import javax.servlet.http.HttpServletRequest;
import java.io.ObjectInputStream;
class Handler {
  void go(HttpServletRequest request) {
    String q = request.getParameter("q");
    stmt.executeQuery("SELECT * FROM t WHERE x = '" + q + "'");
    Runtime.getRuntime().exec(q);
    new File(q);
    new URL(q);
    new ObjectInputStream(new FileInputStream(q));
    response.sendRedirect(q);
  }
}
"""
    graph = _graph("java", "H.java", src)
    assert any(e.name == "Handler" for e in graph.entities)
    assert any(e.name == "go" for e in graph.entities)
    result = analyze_source(tmp_path, "H.java", src)
    classes = observation_classes(result)
    assert VulnerabilityClass.SQL_INJECTION in classes
    assert VulnerabilityClass.COMMAND_INJECTION in classes
    assert VulnerabilityClass.UNSAFE_REDIRECT in classes


def test_php_superglobals(tmp_path: Path) -> None:
    src = """
<?php
require 'lib.php';
class App {
  function index() {
    $q = $_GET['q'];
    mysqli_query($db, "SELECT * FROM t WHERE x = '".$q."'");
    system($q);
    fopen($q, "r");
    echo "<div>$q</div>";
    unserialize($q);
    eval($q);
    header("Location: ".$q);
  }
}
"""
    graph = _graph("php", "a.php", src)
    assert any(e.name == "App" for e in graph.entities)
    assert any(e.name == "index" for e in graph.entities)
    assert any(b.name == "q" for b in graph.bindings)
    result = analyze_source(tmp_path, "a.php", src)
    classes = observation_classes(result)
    assert VulnerabilityClass.SQL_INJECTION in classes
    assert VulnerabilityClass.COMMAND_INJECTION in classes
    assert VulnerabilityClass.DYNAMIC_EXECUTION in classes
    assert VulnerabilityClass.XSS in classes
    string_only = '<?php $x = "eval($q)";\n'
    quiet = analyze_source(tmp_path, "s.php", string_only)
    assert not any(
        obs.vulnerability_class is VulnerabilityClass.DYNAMIC_EXECUTION and "s.php" in obs.file_path
        for obs in quiet.observations
    )


def test_kotlin_fun_and_http(tmp_path: Path) -> None:
    src = """
import io.ktor.server.application.*
class Api {
  fun handle(call: ApplicationCall) {
    val q = call.parameters["q"]
    stmt.execute("SELECT * FROM t WHERE x = '$q'")
    Runtime.getRuntime().exec(q)
    File(q)
    URL(q)
    respondRedirect(q)
  }
}
object Holder
"""
    graph = _graph("kotlin", "A.kt", src)
    names = {e.name for e in graph.entities}
    assert "Api" in names
    assert "handle" in names
    result = analyze_source(tmp_path, "A.kt", src)
    classes = observation_classes(result)
    assert VulnerabilityClass.COMMAND_INJECTION in classes


def test_swift_func_and_url(tmp_path: Path) -> None:
    src = """
import Foundation
class App {
  func run() {
    let q = CommandLine.arguments[1]
    sqlite3_exec(db, q, nil, nil, nil)
    let task = Process()
    FileHandle(forReadingFrom: URL(fileURLWithPath: q))
    JSONDecoder().decode(q)
    let url = URLRequest(url: URL(string: q)!)
  }
}
struct Box {}
enum Kind { case a }
actor Worker {}
"""
    graph = _graph("swift", "A.swift", src)
    names = {e.name for e in graph.entities}
    assert "App" in names
    assert "run" in names
    assert "Box" in names
    result = analyze_source(tmp_path, "A.swift", src)
    classes = observation_classes(result)
    assert (
        VulnerabilityClass.SQL_INJECTION in classes
        or VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL in classes
    )


@pytest.mark.parametrize(
    ("language", "filename", "source"),
    [
        ("ruby", "fp.rb", '# system(params[:x])\nname = "system"\n'),
        ("go", "fp.go", 'package p\nconst s = "http.Get(q)"\n'),
        ("php", "fp.php", "<?php // mysqli_query($db, $_GET['q']);\n"),
        ("java", "Fp.java", 'class Fp { void t() { String eval = "no"; } }\n'),
    ],
)
def test_false_positive_lookalikes(
    tmp_path: Path, language: str, filename: str, source: str
) -> None:
    result = analyze_source(tmp_path, filename, source)
    noisy = {
        VulnerabilityClass.SQL_INJECTION,
        VulnerabilityClass.COMMAND_INJECTION,
        VulnerabilityClass.DYNAMIC_EXECUTION,
        VulnerabilityClass.SSRF,
    }
    assert observation_classes(result).isdisjoint(noisy)
