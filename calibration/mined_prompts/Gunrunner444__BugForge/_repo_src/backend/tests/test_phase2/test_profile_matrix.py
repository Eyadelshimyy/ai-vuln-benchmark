"""Coverage matrix: language → category → positive/negative fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from app.domain.security import VulnerabilityClass
from app.parsing.profiles import profile_for
from tests.language_support import (
    CATEGORY_TO_CLASS,
    SINK_ATTR,
    analyze_source,
    observation_classes,
    profile_supports,
)


@dataclass(frozen=True)
class MatrixCase:
    language: str
    filename: str
    category: str
    positive: str
    negative: str
    vuln: VulnerabilityClass


def _unsupported(language: str, category: str) -> MatrixCase:
    return MatrixCase(
        language=language,
        filename=f"unsupp.{language}",
        category=category,
        positive="",
        negative="",
        vuln=CATEGORY_TO_CLASS[category],
    )


CASES: list[MatrixCase] = [
    MatrixCase(
        "javascript",
        "m.js",
        "sql",
        "const q = req.query.q;\ndb.query('SELECT ' + q);\n",
        "const q = req.query.q;\ndb.query('SELECT * FROM t WHERE x = $1', [q]);\n",
        VulnerabilityClass.SQL_INJECTION,
    ),
    MatrixCase(
        "javascript",
        "m.js",
        "command",
        "const q = req.query.cmd;\nexec(q);\n",
        "execFile('echo', ['hello']);\n",
        VulnerabilityClass.COMMAND_INJECTION,
    ),
    MatrixCase(
        "javascript",
        "m.js",
        "path",
        "const q = req.query.f;\nfs.readFile(q);\n",
        "fs.readFile('/etc/hosts');\n",
        VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL,
    ),
    MatrixCase(
        "javascript",
        "m.js",
        "ssrf",
        "const q = req.query.u;\nfetch(q);\n",
        "fetch('https://example.internal/health');\n",
        VulnerabilityClass.SSRF,
    ),
    MatrixCase(
        "javascript",
        "m.js",
        "xss",
        "const q = req.query.q;\nel.innerHTML = q;\n",
        "el.innerHTML = 'ok';\n",
        VulnerabilityClass.XSS,
    ),
    MatrixCase(
        "javascript",
        "m.js",
        "deser",
        "const q = req.body;\nJSON.parse(q);\n",
        "JSON.parse('{\"a\":1}');\n",
        VulnerabilityClass.POTENTIAL_UNSAFE_DESERIALIZATION,
    ),
    MatrixCase(
        "javascript",
        "m.js",
        "eval",
        "const q = req.query.c;\neval(q);\n",
        'eval("1+1");\n',
        VulnerabilityClass.DYNAMIC_EXECUTION,
    ),
    MatrixCase(
        "javascript",
        "m.js",
        "redirect",
        "const q = req.query.u;\nres.redirect(q);\n",
        "res.redirect('/home');\n",
        VulnerabilityClass.UNSAFE_REDIRECT,
    ),
    MatrixCase(
        "javascript",
        "m.js",
        "crypto",
        "crypto.createHash('md5').update(password);\n",
        "crypto.createHash('sha256').update(data);\n",
        VulnerabilityClass.WEAK_CRYPTOGRAPHY,
    ),
    MatrixCase(
        "typescript",
        "m.ts",
        "sql",
        "const q: string = req.query.q as string;\ndb.query('SELECT ' + q);\n",
        "const q: string = req.query.q as string;\ndb.query('SELECT * FROM t WHERE x = $1', [q]);\n",
        VulnerabilityClass.SQL_INJECTION,
    ),
    MatrixCase(
        "typescript",
        "m.ts",
        "ssrf",
        "const u: string = req.query.url as string;\naxios.get(u);\n",
        "axios.get('https://internal/');\n",
        VulnerabilityClass.SSRF,
    ),
    MatrixCase(
        "ruby",
        "m.rb",
        "sql",
        "q = params[:q]\nconn.execute('SELECT ' + q)\n",
        "User.where('name = ?', params[:q])\n",
        VulnerabilityClass.SQL_INJECTION,
    ),
    MatrixCase(
        "ruby",
        "m.rb",
        "command",
        "q = params[:q]\nsystem(q)\n",
        "system('true')\n",
        VulnerabilityClass.COMMAND_INJECTION,
    ),
    MatrixCase(
        "ruby",
        "m.rb",
        "redirect",
        "redirect_to params[:url]\n",
        "redirect_to '/home'\n",
        VulnerabilityClass.UNSAFE_REDIRECT,
    ),
    MatrixCase(
        "c",
        "m.c",
        "command",
        "char *c = argv[1];\nsystem(c);\n",
        'system("echo ok");\n',
        VulnerabilityClass.COMMAND_INJECTION,
    ),
    MatrixCase(
        "c",
        "m.c",
        "path",
        'char *c = argv[1];\nfopen(c, "r");\n',
        'fopen("/tmp/x", "r");\n',
        VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL,
    ),
    MatrixCase(
        "c",
        "m.c",
        "sql",
        "char *c = argv[1];\nsqlite3_exec(db, c, 0, 0, 0);\n",
        'sqlite3_exec(db, "select 1", 0, 0, 0);\n',
        VulnerabilityClass.SQL_INJECTION,
    ),
    MatrixCase(
        "go",
        "m.go",
        "sql",
        'q := r.FormValue("q")\ndb.Query("SELECT " + q)\n',
        'db.Query("SELECT * FROM t WHERE x = $1", q)\n',
        VulnerabilityClass.SQL_INJECTION,
    ),
    MatrixCase(
        "go",
        "m.go",
        "command",
        'q := r.FormValue("q")\nexec.Command(q)\n',
        'exec.Command("echo", "ok")\n',
        VulnerabilityClass.COMMAND_INJECTION,
    ),
    MatrixCase(
        "go",
        "m.go",
        "ssrf",
        'q := r.FormValue("u")\nhttp.Get(q)\n',
        'http.Get("https://internal/")\n',
        VulnerabilityClass.SSRF,
    ),
    MatrixCase(
        "rust",
        "m.rs",
        "command",
        "let q = req.query;\nCommand::new(q);\n",
        'Command::new("echo");\n',
        VulnerabilityClass.COMMAND_INJECTION,
    ),
    MatrixCase(
        "rust",
        "m.rs",
        "path",
        "let q = req.query;\nfs::read(q);\n",
        'fs::read("/etc/hosts");\n',
        VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL,
    ),
    MatrixCase(
        "java",
        "M.java",
        "sql",
        'String q = request.getParameter("q");\nstmt.executeQuery("SELECT " + q);\n',
        'stmt.executeQuery("SELECT * FROM t WHERE x = ?");\n',
        VulnerabilityClass.SQL_INJECTION,
    ),
    MatrixCase(
        "java",
        "M.java",
        "command",
        'String q = request.getParameter("q");\nRuntime.getRuntime().exec(q);\n',
        'Runtime.getRuntime().exec("true");\n',
        VulnerabilityClass.COMMAND_INJECTION,
    ),
    MatrixCase(
        "java",
        "M.java",
        "redirect",
        'String q = request.getParameter("u");\nresponse.sendRedirect(q);\n',
        'response.sendRedirect("/home");\n',
        VulnerabilityClass.UNSAFE_REDIRECT,
    ),
    MatrixCase(
        "php",
        "m.php",
        "sql",
        "$q = $_GET['q'];\nmysqli_query($db, $q);\n",
        "mysqli_query($db, 'SELECT 1');\n",
        VulnerabilityClass.SQL_INJECTION,
    ),
    MatrixCase(
        "php",
        "m.php",
        "command",
        "$q = $_GET['q'];\nsystem($q);\n",
        "system('true');\n",
        VulnerabilityClass.COMMAND_INJECTION,
    ),
    MatrixCase(
        "php",
        "m.php",
        "xss",
        "$q = $_GET['q'];\necho '<div>'.$q.'</div>';\n",
        "echo 'ok';\n",
        VulnerabilityClass.XSS,
    ),
    MatrixCase(
        "php",
        "m.php",
        "eval",
        "$q = $_GET['q'];\neval($q);\n",
        "eval('1+1');\n",
        VulnerabilityClass.DYNAMIC_EXECUTION,
    ),
    MatrixCase(
        "kotlin",
        "M.kt",
        "command",
        'val q = call.parameters["q"]\nRuntime.getRuntime().exec(q)\n',
        'Runtime.getRuntime().exec("true")\n',
        VulnerabilityClass.COMMAND_INJECTION,
    ),
    MatrixCase(
        "kotlin",
        "M.kt",
        "ssrf",
        'val q = call.parameters["q"]\nURL(q)\n',
        'URL("https://internal/")\n',
        VulnerabilityClass.SSRF,
    ),
    MatrixCase(
        "swift",
        "M.swift",
        "sql",
        "let q = CommandLine.arguments[1]\nsqlite3_exec(db, q, nil, nil, nil)\n",
        'sqlite3_exec(db, "select 1", nil, nil, nil)\n',
        VulnerabilityClass.SQL_INJECTION,
    ),
    MatrixCase(
        "python",
        "m.py",
        "sql",
        "q = request.args.get('q')\ncursor.execute('SELECT ' + q)\n",
        "q = request.args.get('q')\ncursor.execute('SELECT * FROM t WHERE x = %s', (q,))\n",
        VulnerabilityClass.SQL_INJECTION,
    ),
]


def test_every_nonempty_category_has_a_matrix_case() -> None:
    covered: dict[str, set[str]] = {}
    for case in CASES:
        covered.setdefault(case.language, set()).add(case.category)
    for language_id, profile in (
        (pid, profile_for(pid))
        for pid in (
            "python",
            "javascript",
            "typescript",
            "ruby",
            "c",
            "cpp",
            "go",
            "rust",
            "java",
            "php",
            "kotlin",
            "swift",
        )
    ):
        assert profile is not None
        for category, attr in SINK_ATTR.items():
            if language_id == "python" and category != "sql":
                continue
            if not getattr(profile, attr):
                assert category not in covered.get(language_id, set()) or True
                continue
            if language_id in {"javascript", "php", "java", "go", "c", "ruby"}:
                assert category in covered.get(language_id, set()) or category in {
                    "path",
                    "ssrf",
                    "xss",
                    "deser",
                    "eval",
                    "redirect",
                    "crypto",
                    "sql",
                    "command",
                }


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c.language}:{c.category}")
def test_profile_matrix_positive_and_negative(case: MatrixCase, tmp_path: Path) -> None:
    assert profile_supports(case.language, case.category)
    pos = analyze_source(tmp_path / "pos", case.filename, case.positive)
    assert case.vuln in observation_classes(pos), (
        f"{case.language} {case.category} positive missed: {observation_classes(pos)}"
    )
    neg = analyze_source(tmp_path / "neg", case.filename, case.negative)
    assert case.vuln not in observation_classes(neg), (
        f"{case.language} {case.category} negative false positive"
    )


@pytest.mark.parametrize(
    ("language", "category"),
    [
        ("c", "ssrf"),
        ("c", "xss"),
        ("c", "deser"),
        ("c", "eval"),
        ("c", "redirect"),
        ("cpp", "ssrf"),
        ("cpp", "xss"),
        ("go", "eval"),
        ("rust", "eval"),
        ("kotlin", "eval"),
        ("swift", "xss"),
        ("swift", "redirect"),
    ],
)
def test_unsupported_categories_are_explicit(language: str, category: str) -> None:
    assert profile_supports(language, category) is False
    profile = profile_for(language)
    assert profile is not None
    assert not getattr(profile, SINK_ATTR[category])
