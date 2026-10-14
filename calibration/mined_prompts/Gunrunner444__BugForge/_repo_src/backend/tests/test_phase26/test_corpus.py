"""Phase 26 golden security corpus for the priority languages."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.security import VulnerabilityClass
from app.security.coverage import COVERAGE
from app.security.engine import SecurityAnalysisEngine

SQL = VulnerabilityClass.SQL_INJECTION
CMD = VulnerabilityClass.COMMAND_INJECTION
PATH = VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL
SSRF = VulnerabilityClass.SSRF
XSS = VulnerabilityClass.XSS
DESER = VulnerabilityClass.UNSAFE_DESERIALIZATION
PDESER = VulnerabilityClass.POTENTIAL_UNSAFE_DESERIALIZATION
EVAL = VulnerabilityClass.DYNAMIC_EXECUTION
REDIR = VulnerabilityClass.UNSAFE_REDIRECT
SECRET = VulnerabilityClass.HARDCODED_SECRET


def _scan(tmp_path: Path, name: str, source: str):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return SecurityAnalysisEngine().analyze_repository(tmp_path, [path])


def _kinds(result) -> set[VulnerabilityClass]:
    return {obs.vulnerability_class for obs in result.observations}


def _secret_hidden(result, secret: str) -> None:
    blob = " ".join(obs.evidence_text for obs in result.observations)
    assert secret not in blob
    assert SECRET in _kinds(result)


def test_javascript_corpus(tmp_path: Path) -> None:
    base = tmp_path / "js"
    assert SQL in _kinds(
        _scan(
            base, "sql.js", "function h(req){ const q = req.query.q; db.query('SELECT ' + q); }\n"
        )
    )
    assert SQL not in _kinds(
        _scan(
            base,
            "param.js",
            "function h(req){ const q = req.query.q; db.query('SELECT ?', [q]); }\n",
        )
    )
    assert CMD in _kinds(
        _scan(base, "exec.js", "function h(req){ const q = req.query.q; exec(q); }\n")
    )
    assert CMD not in _kinds(
        _scan(
            base,
            "argv.js",
            "function h(req){ const q = req.query.q; execFile('git', ['show', q]); }\n",
        )
    )
    assert CMD in _kinds(
        _scan(base, "bin.js", "function h(req){ const q = req.query.q; spawn(q, ['--help']); }\n")
    )
    assert PATH in _kinds(
        _scan(base, "path.js", "function h(req){ const q = req.query.q; fs.readFile(q, cb); }\n")
    )
    assert SSRF in _kinds(
        _scan(base, "ssrf.js", "function h(req){ const q = req.query.q; fetch(q); }\n")
    )
    assert SSRF not in _kinds(
        _scan(base, "url.js", "function h(){ fetch('https://example.com'); }\n")
    )
    assert XSS in _kinds(
        _scan(base, "xss.js", "function h(req){ const q = req.query.q; el.innerHTML = q; }\n")
    )
    assert XSS not in _kinds(
        _scan(base, "text.js", "function h(req){ const q = req.query.q; el.textContent = q; }\n")
    )
    assert EVAL in _kinds(
        _scan(base, "eval.js", "function h(req){ const q = req.query.q; eval(q); }\n")
    )
    assert REDIR in _kinds(
        _scan(base, "redir.js", "function h(req, res){ res.redirect(req.query.next); }\n")
    )
    shadowed = _scan(
        base,
        "shadow.js",
        "function h(req){\n  function query(x){ return x; }\n  const q = req.query.q;\n  query(q);\n  db.query(q);\n}\n",
    )
    assert SQL in _kinds(shadowed)
    assert all(
        "function query" not in obs.evidence_text
        for obs in shadowed.observations
        if obs.vulnerability_class is SQL
    )
    assert PDESER in _kinds(_scan(base, "json.js", "function h(req){ JSON.parse(req.query.q); }\n"))
    assert DESER not in _kinds(
        _scan(base, "json.js", "function h(req){ JSON.parse(req.query.q); }\n")
    )
    secret = "sk_live_js_same_line"
    _secret_hidden(_scan(base, "sec.js", f'const api_key = "{secret}";\n'), secret)
    assert _kinds(_scan(base, "mention.js", "const message = 'this text mentions md5';\n")) == set()
    broken = _scan(base, "bad.js", "function h(req){ const q = req.query.q;\n  db.query(q\n")
    assert SQL not in _kinds(broken)


def test_typescript_corpus(tmp_path: Path) -> None:
    base = tmp_path / "ts"
    assert SQL in _kinds(
        _scan(base, "sql.ts", "function h(req: Request){ const q = req.query.q; db.query(q); }\n")
    )
    assert CMD in _kinds(
        _scan(
            base,
            "cmd.ts",
            "function h(req: Request){ const q = req.query.q; child_process.exec(q); }\n",
        )
    )
    assert PATH in _kinds(
        _scan(base, "path.ts", "function h(req: Request){ fs.readFileSync(req.query.q); }\n")
    )
    assert SSRF in _kinds(
        _scan(base, "ssrf.ts", "function h(req: Request){ axios.get(req.query.q); }\n")
    )
    assert XSS in _kinds(
        _scan(base, "xss.ts", "function h(req: Request){ el.innerHTML = req.query.q; }\n")
    )
    assert EVAL in _kinds(
        _scan(base, "eval.ts", "function h(req: Request){ eval(req.query.q); }\n")
    )
    assert REDIR in _kinds(
        _scan(
            base,
            "redir.ts",
            "function h(req: Request, res: Response){ res.redirect(req.query.next); }\n",
        )
    )
    assert SQL not in _kinds(
        _scan(base, "safe.ts", "function h(req: Request){ db.query('SELECT ?', [req.query.q]); }\n")
    )
    secret = "sk_live_ts_same_line"
    _secret_hidden(_scan(base, "sec.ts", f'const password = "{secret}";\n'), secret)


def test_go_corpus(tmp_path: Path) -> None:
    base = tmp_path / "go"
    handler = 'package main\nfunc h(r *http.Request) {\n  q := r.FormValue("q")\n  %s\n}\n'
    assert SQL in _kinds(_scan(base, "sql.go", handler % "db.Query(q)"))
    assert SQL not in _kinds(_scan(base, "const.go", handler % 'db.Query("SELECT 1")'))
    assert CMD not in _kinds(_scan(base, "git.go", handler % 'exec.Command("git", q)'))
    assert CMD in _kinds(_scan(base, "bin.go", handler % "exec.Command(q)"))
    assert PATH in _kinds(_scan(base, "path.go", handler % "os.Open(q)"))
    assert SSRF in _kinds(_scan(base, "ssrf.go", handler % "http.Get(q)"))
    assert REDIR in _kinds(_scan(base, "redir.go", handler % "http.Redirect(w, r, q, 302)"))
    assert XSS in _kinds(_scan(base, "xss.go", handler % "template.HTML(q)"))
    assert XSS not in _kinds(_scan(base, "write.go", handler % "w.Write([]byte(q))"))
    assert PDESER in _kinds(_scan(base, "json.go", handler % "json.Unmarshal([]byte(q), &out)"))
    assert DESER not in _kinds(_scan(base, "json.go", handler % "json.Unmarshal([]byte(q), &out)"))
    assert EVAL not in _kinds(_scan(base, "write.go", handler % "w.Write([]byte(q))"))
    secret = "sk_live_go_same_line"
    _secret_hidden(_scan(base, "sec.go", f'package main\nvar api_key = "{secret}"\n'), secret)
    assert COVERAGE["go"]["dynamic_exec"] == "unsupported"


def test_java_and_kotlin_corpus(tmp_path: Path) -> None:
    base = tmp_path / "jvm"
    java = 'class A { void h(HttpServletRequest req){ String q = req.getParameter("q"); %s } }\n'
    assert SQL in _kinds(_scan(base, "sql.java", java % "stmt.executeQuery(q);"))
    assert SQL not in _kinds(
        _scan(base, "prep.java", java % "ps.setString(1, q); ps.executeQuery();")
    )
    assert CMD in _kinds(_scan(base, "cmd.java", java % "Runtime.getRuntime().exec(q);"))
    assert CMD not in _kinds(_scan(base, "pb.java", java % 'new ProcessBuilder("git", q).start();'))
    assert PATH in _kinds(_scan(base, "path.java", java % "new File(q);"))
    assert SSRF in _kinds(_scan(base, "url.java", java % "new URL(q);"))
    assert REDIR in _kinds(_scan(base, "redir.java", java % "resp.sendRedirect(q);"))
    assert EVAL in _kinds(_scan(base, "eval.java", java % "engine.eval(q);"))
    assert XSS not in _kinds(_scan(base, "print.java", java % "System.out.print(q);"))
    assert DESER in _kinds(_scan(base, "deser.java", java % "new ObjectInputStream(q);"))
    assert DESER not in _kinds(
        _scan(
            base,
            "read.java",
            "class A { void h(InputStream in){ new ObjectInputStream(in).readObject(); } }\n",
        )
    )
    secret = "sk_live_java_same_line"
    _secret_hidden(
        _scan(base, "sec.java", f'class A {{ String password = "{secret}"; }}\n'), secret
    )

    kotlin = 'fun h(call: ApplicationCall){ val q = call.parameters["q"]; %s }\n'
    assert SQL in _kinds(_scan(base, "sql.kt", kotlin % "stmt.executeQuery(q)"))
    assert CMD in _kinds(_scan(base, "cmd.kt", kotlin % "Runtime.getRuntime().exec(q)"))
    assert PATH in _kinds(_scan(base, "path.kt", kotlin % "File(q)"))
    assert SSRF in _kinds(_scan(base, "url.kt", kotlin % "URL(q)"))
    assert REDIR in _kinds(_scan(base, "redir.kt", kotlin % "call.respondRedirect(q)"))
    assert XSS in _kinds(_scan(base, "xss.kt", kotlin % "call.respondText(q)"))
    assert EVAL not in _kinds(_scan(base, "xss.kt", kotlin % "call.respondText(q)"))
    assert COVERAGE["kotlin"]["dynamic_exec"] == "unsupported"
    assert COVERAGE["java"]["xss"] == "limited"


def test_php_corpus(tmp_path: Path) -> None:
    base = tmp_path / "php"
    body = "<?php function h(){ $q = $_GET['q']; %s }\n"
    assert SQL in _kinds(_scan(base, "sql.php", body % "mysqli_query($c, $q);"))
    assert SQL not in _kinds(
        _scan(base, "prep.php", body % "$pdo->prepare('SELECT * FROM t WHERE n = ?');")
    )
    assert CMD in _kinds(_scan(base, "cmd.php", body % "system($q);"))
    assert PATH in _kinds(_scan(base, "inc.php", body % "include $q;"))
    assert SSRF in _kinds(_scan(base, "file.php", body % "file_get_contents($q);"))
    assert SSRF not in _kinds(
        _scan(base, "curl.php", body % "curl_setopt($ch, CURLOPT_URL, $q); curl_exec($ch);")
    )
    assert DESER in _kinds(_scan(base, "unser.php", body % "unserialize($q);"))
    assert EVAL in _kinds(_scan(base, "eval.php", body % "eval($q);"))
    assert REDIR in _kinds(_scan(base, "redir.php", body % "header('Location: ' . $q);"))
    assert XSS not in _kinds(_scan(base, "echo.php", body % "echo $q;"))
    assert XSS in _kinds(_scan(base, "page.php", "<div><?php $q = $_GET['q']; echo $q; ?></div>\n"))
    assert XSS not in _kinds(
        _scan(base, "esc.php", "<div><?php $q = htmlspecialchars($_GET['q']); echo $q; ?></div>\n")
    )
    secret = "sk_live_php_same_line"
    _secret_hidden(_scan(base, "sec.php", f"<?php $password = '{secret}';\n"), secret)


def test_csharp_corpus(tmp_path: Path) -> None:
    base = tmp_path / "cs"
    body = 'void H(){ var q = Request.Query["q"]; %s }\n'
    assert SQL in _kinds(_scan(base, "sql.cs", body % "cmd.ExecuteReader(q);"))
    assert SQL in _kinds(_scan(base, "raw.cs", body % "db.FromSqlRaw(q);"))
    assert SQL not in _kinds(
        _scan(base, "param.cs", body % 'cmd.Parameters.AddWithValue("@q", q);')
    )
    assert CMD in _kinds(_scan(base, "proc.cs", body % "Process.Start(q);"))
    assert CMD not in _kinds(_scan(base, "git.cs", body % 'Process.Start("git", q);'))
    assert PATH in _kinds(_scan(base, "file.cs", body % "File.ReadAllText(q);"))
    assert SSRF in _kinds(_scan(base, "http.cs", body % "client.GetAsync(q);"))
    assert XSS in _kinds(_scan(base, "xss.cs", body % "Html.Raw(q);"))
    assert XSS not in _kinds(_scan(base, "enc.cs", body % "Html.Raw(WebUtility.HtmlEncode(q));"))
    assert REDIR in _kinds(_scan(base, "redir.cs", body % "Redirect(q);"))
    assert DESER not in _kinds(
        _scan(base, "fmt.cs", "void H(){ var f = new BinaryFormatter(); f.Deserialize(stream); }\n")
    )
    assert EVAL not in _kinds(_scan(base, "redir.cs", body % "Redirect(q);"))
    secret = "sk_live_cs_same_line"
    _secret_hidden(_scan(base, "sec.cs", f'var password = "{secret}";\n'), secret)
    assert COVERAGE["csharp"]["deserialization"] == "limited"
    assert COVERAGE["csharp"]["dynamic_exec"] == "unsupported"


def test_ruby_corpus(tmp_path: Path) -> None:
    base = tmp_path / "rb"

    def method(body: str) -> str:
        return f"def h\n  q = params[:q]\n  {body}\nend\n"

    assert SQL in _kinds(_scan(base, "sql.rb", method("connection.execute(q)")))
    assert SQL in _kinds(_scan(base, "raw.rb", method('User.where("name = " + q)')))
    assert SQL not in _kinds(_scan(base, "hash.rb", method("User.where(name: q)")))
    assert CMD in _kinds(_scan(base, "cmd.rb", method("system(q)")))
    assert CMD not in _kinds(_scan(base, "git.rb", method("system('git', q)")))
    assert PATH in _kinds(_scan(base, "path.rb", method("File.read(q)")))
    assert SSRF in _kinds(_scan(base, "http.rb", method("Net::HTTP.get(q)")))
    assert XSS in _kinds(_scan(base, "xss.rb", method("q.html_safe")))
    assert XSS not in _kinds(
        _scan(base, "esc.rb", "def h\n  q = h(params[:q])\n  q.html_safe\nend\n")
    )
    assert DESER in _kinds(_scan(base, "mar.rb", method("Marshal.load(q)")))
    assert EVAL in _kinds(_scan(base, "eval.rb", method("eval(q)")))
    assert EVAL not in _kinds(_scan(base, "send.rb", method("obj.send(q)")))
    assert REDIR in _kinds(_scan(base, "redir.rb", method("redirect_to q")))
    assert REDIR not in _kinds(_scan(base, "home.rb", method("redirect_to '/home'")))
    secret = "sk_live_rb_same_line"
    _secret_hidden(_scan(base, "sec.rb", f"password = '{secret}'\n"), secret)


def test_rust_corpus(tmp_path: Path) -> None:
    base = tmp_path / "rs"

    def fn(body: str) -> str:
        return f"fn h(req: Request) {{\n    let q = req.query();\n    {body}\n}}\n"

    assert SQL in _kinds(_scan(base, "sql.rs", fn("sqlx::query(&q);")))
    assert CMD in _kinds(_scan(base, "cmd.rs", fn("Command::new(q);")))
    assert CMD not in _kinds(_scan(base, "git.rs", fn('Command::new("git").arg(q);')))
    assert PATH in _kinds(_scan(base, "path.rs", fn("File::open(q);")))
    assert SSRF in _kinds(_scan(base, "ssrf.rs", fn("reqwest::get(q);")))
    assert SSRF not in _kinds(_scan(base, "client.rs", fn("client.get(q);")))
    assert XSS in _kinds(_scan(base, "xss.rs", fn("Html::from_string_unchecked(q);")))
    assert XSS not in _kinds(_scan(base, "html.rs", fn("Html::new(q);")))
    assert PDESER in _kinds(_scan(base, "json.rs", fn("serde_json::from_str(&q);")))
    assert DESER in _kinds(_scan(base, "bin.rs", fn("bincode::deserialize(&q);")))
    assert DESER not in _kinds(_scan(base, "json.rs", fn("serde_json::from_str(&q);")))
    assert REDIR in _kinds(_scan(base, "redir.rs", fn("Redirect::to(&q);")))
    assert EVAL not in _kinds(_scan(base, "html.rs", fn("Html::new(q);")))
    secret = "sk_live_rs_same_line"
    _secret_hidden(_scan(base, "sec.rs", f'let password = "{secret}";\n'), secret)
    assert COVERAGE["rust"]["dynamic_exec"] == "unsupported"


def test_nestjs_parameter_source(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        "nest.js",
        "import { Query } from '@nestjs/common';\nfunction h(@Query() q){ db.query(q); }\n",
    )
    assert SQL in _kinds(result)
    unknown = _scan(
        tmp_path / "local",
        "local.js",
        "import { Query } from './local';\nfunction h(@Query() q){ db.query(q); }\n",
    )
    assert SQL not in _kinds(unknown)


@pytest.mark.parametrize(
    "language",
    ["javascript", "typescript", "go", "java", "kotlin", "php", "csharp", "ruby", "rust"],
)
def test_priority_languages_have_dedicated_security_fixtures(language: str) -> None:
    tested = [family for family, label in COVERAGE[language].items() if label == "tested"]
    assert tested
    assert all(
        label in {"tested", "existing-suite", "limited", "unsupported"}
        for label in COVERAGE[language].values()
    )
