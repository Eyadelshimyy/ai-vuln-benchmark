"""TypeScript parser and security-analysis depth tests."""

from __future__ import annotations

from pathlib import Path

from app.adapters.languages import TypeScriptAdapter
from app.domain.language import LanguageCapability
from app.domain.security import VulnerabilityClass
from app.parsing.engine import parse_source
from app.parsing.model import SyntaxGraph
from tests.language_support import analyze_source, observation_classes


def _graph(source: str, name: str = "app.ts") -> SyntaxGraph:
    return parse_source("typescript", Path(name), source)


class TestTypeScriptSurface:
    def test_extensions_and_capabilities(self) -> None:
        adapter = TypeScriptAdapter()
        assert adapter.matches_path(Path("a.ts"))
        assert adapter.matches_path(Path("a.tsx"))
        assert adapter.supports(LanguageCapability.PARSE)
        assert adapter.supports(LanguageCapability.STATIC_ANALYSIS)
        assert adapter.supports(LanguageCapability.SECURITY_ANALYSIS)


class TestTypeScriptSyntax:
    def test_typed_function_and_interface(self) -> None:
        src = """
import axios from 'axios';
export async function go(req: Request): Promise<void> {}
interface User { id: string }
type Id = string;
enum Kind { A, B }
namespace Models { export class Box {} }
class Service {
  constructor(private readonly db: DB) {}
  async run(q: string): Promise<void> { this.db.query(q); }
}
"""
        graph = _graph(src)
        names = {e.name for e in graph.entities}
        assert "go" in names
        assert "User" in names
        assert "Kind" in names
        assert "Service" in names
        assert any("axios" in (imp.module or "") for imp in graph.imports)

    def test_typed_destructuring_and_generics(self) -> None:
        src = """
function wrap<T>(value: T): T { return value; }
const { query }: Request = req;
const q: string = req.query.q as string;
"""
        graph = _graph(src)
        names = {b.name for b in graph.bindings}
        assert "query" in names
        assert "q" in names
        assert any(e.name == "wrap" for e in graph.entities)

    def test_tsx_react_component(self) -> None:
        src = """
export function Widget({ html }: { html: string }) {
  return <div dangerouslySetInnerHTML={{__html: html}} />;
}
"""
        graph = _graph(src, "w.tsx")
        assert any(e.name == "Widget" for e in graph.entities)
        assert any(c.name == "dangerouslySetInnerHTML" for c in graph.calls)

    def test_types_do_not_hide_calls(self) -> None:
        src = """
const q: string = req.query.q as string;
db.query("SELECT * FROM t WHERE x = '" + q + "'");
"""
        graph = _graph(src)
        assert any(c.name == "query" for c in graph.calls)
        assert any(b.name == "q" for b in graph.bindings)

    def test_optional_chaining_nullish_and_assertion(self) -> None:
        src = """
const q = req?.query?.q ?? "safe";
const n = user!.name as string;
enum Mode { On, Off }
class Box<T> { value!: T }
"""
        graph = _graph(src)
        names = {b.name for b in graph.bindings}
        assert "q" in names
        assert "n" in names
        assert any(e.name == "Mode" for e in graph.entities)
        assert any(e.name == "Box" for e in graph.entities)


class TestTypeScriptSecurity:
    def test_nestjs_decorators(self, tmp_path: Path) -> None:
        src = """
import { Query, Param, Body } from '@nestjs/common';
export class C {
  search(@Query('q') q: string) {
    db.query("SELECT * FROM t WHERE x = '" + q + "'");
  }
  byId(@Param('id') id: string) {
    fetch(id);
  }
  save(@Body() body: Record<string, string>) {
    eval(body.code);
  }
}
"""
        result = analyze_source(tmp_path, "c.ts", src)
        classes = observation_classes(result)
        assert VulnerabilityClass.SQL_INJECTION in classes
        assert VulnerabilityClass.SSRF in classes
        assert VulnerabilityClass.DYNAMIC_EXECUTION in classes

    def test_typed_sql_and_ssrf(self, tmp_path: Path) -> None:
        src = """
export async function proxy(req: Request) {
  const target: string = req.query.url as string;
  await axios.get(target);
  const q: string = req.query.q as string;
  db.query("SELECT * FROM t WHERE x = '" + q + "'");
}
"""
        result = analyze_source(tmp_path, "p.ts", src)
        classes = observation_classes(result)
        assert VulnerabilityClass.SSRF in classes
        assert VulnerabilityClass.SQL_INJECTION in classes

    def test_parameterized_query_quiet(self, tmp_path: Path) -> None:
        src = """
export function search(req: Request) {
  const q: string = req.query.q as string;
  db.query("SELECT * FROM t WHERE x = $1", [q]);
}
"""
        result = analyze_source(tmp_path, "ok.ts", src)
        assert VulnerabilityClass.SQL_INJECTION not in observation_classes(result)
