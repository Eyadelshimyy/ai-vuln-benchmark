"""Semantic edge cases for Phase 25 detection.

``gap`` means the engine emitted nothing and that absence is a known
limitation. It is not a claim that the snippet is safe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.security import VulnerabilityClass
from app.security.engine import SecurityAnalysisEngine

SQL = VulnerabilityClass.SQL_INJECTION
CMD = VulnerabilityClass.COMMAND_INJECTION
PATH = VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL
SSRF = VulnerabilityClass.SSRF
XSS = VulnerabilityClass.XSS
DESER = VulnerabilityClass.UNSAFE_DESERIALIZATION
EVAL = VulnerabilityClass.DYNAMIC_EXECUTION
REDIR = VulnerabilityClass.UNSAFE_REDIRECT
SECRET = VulnerabilityClass.HARDCODED_SECRET

Q = "request.args.get('q')"

# Absence here is a miss or an unmodeled pattern, not a negative proof.
GAPS = {
    "sql-wrap": "Unknown wrappers do not propagate taint.",
    "sql-ambig": "An ambiguous callee is not resolved.",
    "sql-partial": "A syntax error does not produce a partial finding.",
    "cmd-wrap": "Unknown wrappers do not propagate taint.",
    "cmd-ambig": "An ambiguous callee is not resolved.",
    "cmd-partial": "A syntax error does not produce a partial finding.",
    "path-wrap": "Unknown wrappers do not propagate taint.",
    "path-ambig": "An ambiguous callee is not resolved.",
    "path-partial": "A syntax error does not produce a partial finding.",
    "ssrf-wrap": "Unknown wrappers do not propagate taint.",
    "ssrf-san": "A conditional allowlist around the URL is not modeled.",
    "ssrf-ambig": "An ambiguous callee is not resolved.",
    "ssrf-partial": "A syntax error does not produce a partial finding.",
    "xss-wrap": "Unknown wrappers do not propagate taint.",
    "xss-ambig": "An ambiguous callee is not resolved.",
    "xss-partial": "A syntax error does not produce a partial finding.",
    "deser-wrap": "Unknown wrappers do not propagate taint.",
    "deser-ambig": "An ambiguous callee is not resolved.",
    "deser-partial": "A syntax error does not produce a partial finding.",
    "eval-wrap": "Unknown wrappers do not propagate taint.",
    "eval-ambig": "An ambiguous callee is not resolved.",
    "eval-partial": "A syntax error does not produce a partial finding.",
    "redir-wrap": "Unknown wrappers do not propagate taint.",
    "redir-san": "A conditional allowlist around the redirect is not modeled.",
    "redir-ambig": "An ambiguous callee is not resolved.",
    "redir-partial": "A syntax error does not produce a partial finding.",
}


def _files(body: str) -> tuple[tuple[str, str], ...]:
    return (("a.py", body),)


CASES: list[tuple[str, VulnerabilityClass, tuple[tuple[str, str], ...], str]] = [
    ("sql-direct", SQL, _files(f"def f():\n    cursor.execute({Q})\n"), "finding"),
    ("sql-alias", SQL, _files(f"def f():\n    q = {Q}\n    cursor.execute(q)\n"), "finding"),
    (
        "sql-wrap",
        SQL,
        _files(f"def wrap(v):\n    return v\ndef f():\n    cursor.execute(wrap({Q}))\n"),
        "gap",
    ),
    ("sql-field", SQL, _files("def f():\n    cursor.execute(request.args['q'])\n"), "finding"),
    ("sql-const", SQL, _files("def f():\n    cursor.execute('SELECT 1')\n"), "clean"),
    (
        "sql-san",
        SQL,
        _files(f"def f():\n    q = {Q}\n    cursor.execute('SELECT ?', (q,))\n"),
        "clean",
    ),
    (
        "sql-shadow",
        SQL,
        _files(f"def execute(sql):\n    return sql\ndef f():\n    execute({Q})\n"),
        "finding",
    ),
    ("sql-arg", SQL, _files(f"def f():\n    cursor.execute('SELECT 1', {Q})\n"), "clean"),
    (
        "sql-cross",
        SQL,
        (
            ("lib.py", "def run_query(value):\n    cursor.execute(value)\n"),
            ("app.py", f"from lib import run_query\ndef f():\n    run_query({Q})\n"),
        ),
        "finding",
    ),
    (
        "sql-ambig",
        SQL,
        _files(
            f"def helper(v):\n    cursor.execute(v)\ndef helper(v):\n    return v\ndef f():\n    helper({Q})\n"
        ),
        "gap",
    ),
    ("sql-partial", SQL, _files("def f(\n    cursor.execute(request.args.get('q'))\n"), "gap"),
    ("cmd-direct", CMD, _files(f"import os\ndef f():\n    os.system({Q})\n"), "finding"),
    ("cmd-alias", CMD, _files(f"import os\ndef f():\n    c = {Q}\n    os.system(c)\n"), "finding"),
    (
        "cmd-wrap",
        CMD,
        _files(f"import os\ndef wrap(v):\n    return v\ndef f():\n    os.system(wrap({Q}))\n"),
        "gap",
    ),
    (
        "cmd-field",
        CMD,
        _files("import os\ndef f():\n    os.system(request.args['cmd'])\n"),
        "finding",
    ),
    ("cmd-const", CMD, _files("import os\ndef f():\n    os.system('ls')\n"), "clean"),
    (
        "cmd-san",
        CMD,
        _files(f"import os, shlex\ndef f():\n    os.system(shlex.quote({Q}))\n"),
        "clean",
    ),
    (
        "cmd-shadow",
        CMD,
        _files(f"import os\ndef system(c):\n    return c\ndef f():\n    system({Q})\n"),
        "clean",
    ),
    (
        "cmd-arg",
        CMD,
        _files(f"import subprocess\ndef f():\n    subprocess.run(['git', {Q}])\n"),
        "clean",
    ),
    (
        "cmd-cross",
        CMD,
        (
            ("lib.py", "import os\ndef run(value):\n    os.system(value)\n"),
            ("app.py", f"from lib import run\ndef f():\n    run({Q})\n"),
        ),
        "finding",
    ),
    (
        "cmd-ambig",
        CMD,
        _files(
            f"import os\ndef helper(v):\n    os.system(v)\ndef helper(v):\n    return v\ndef f():\n    helper({Q})\n"
        ),
        "gap",
    ),
    (
        "cmd-partial",
        CMD,
        _files("import os\ndef f(\n    os.system(request.args.get('q'))\n"),
        "gap",
    ),
    ("path-direct", PATH, _files(f"def f():\n    open({Q})\n"), "finding"),
    ("path-alias", PATH, _files(f"def f():\n    n = {Q}\n    open(n)\n"), "finding"),
    (
        "path-wrap",
        PATH,
        _files(f"def wrap(v):\n    return v\ndef f():\n    open(wrap({Q}))\n"),
        "gap",
    ),
    ("path-field", PATH, _files("def f():\n    open(request.args['name'])\n"), "finding"),
    ("path-const", PATH, _files("def f():\n    open('README.md')\n"), "clean"),
    (
        "path-san",
        PATH,
        _files(f"import os\ndef f():\n    open(os.path.realpath({Q}))\n"),
        "finding",
    ),
    (
        "path-shadow",
        PATH,
        _files(f"def open(p):\n    return p\ndef f():\n    open({Q})\n"),
        "finding",
    ),
    ("path-arg", PATH, _files(f"def f():\n    open('README.md', {Q})\n"), "finding"),
    (
        "path-cross",
        PATH,
        (
            ("lib.py", "def read(value):\n    open(value)\n"),
            ("app.py", f"from lib import read\ndef f():\n    read({Q})\n"),
        ),
        "finding",
    ),
    (
        "path-ambig",
        PATH,
        _files(
            f"def helper(v):\n    open(v)\ndef helper(v):\n    return v\ndef f():\n    helper({Q})\n"
        ),
        "gap",
    ),
    ("path-partial", PATH, _files("def f(\n    open(request.args.get('q'))\n"), "gap"),
    ("ssrf-direct", SSRF, _files(f"import requests\ndef f():\n    requests.get({Q})\n"), "finding"),
    (
        "ssrf-alias",
        SSRF,
        _files(f"import requests\ndef f():\n    u = {Q}\n    requests.get(u)\n"),
        "finding",
    ),
    (
        "ssrf-wrap",
        SSRF,
        _files(
            f"import requests\ndef wrap(v):\n    return v\ndef f():\n    requests.get(wrap({Q}))\n"
        ),
        "gap",
    ),
    (
        "ssrf-field",
        SSRF,
        _files("import requests\ndef f():\n    requests.get(request.args['url'])\n"),
        "finding",
    ),
    (
        "ssrf-const",
        SSRF,
        _files("import requests\ndef f():\n    requests.get('https://example.com')\n"),
        "clean",
    ),
    (
        "ssrf-san",
        SSRF,
        _files(
            f"import requests\ndef f():\n    u = {Q}\n"
            "    requests.get(u if url_has_allowed_host_and_scheme(u, None) else 'https://example.com')\n"
        ),
        "gap",
    ),
    ("ssrf-shadow", SSRF, _files(f"def get(u):\n    return u\ndef f():\n    get({Q})\n"), "clean"),
    (
        "ssrf-arg",
        SSRF,
        _files(f"import requests\ndef f():\n    requests.get('https://example.com', params={Q})\n"),
        "clean",
    ),
    (
        "ssrf-cross",
        SSRF,
        (
            ("lib.py", "import requests\ndef fetch(value):\n    requests.get(value)\n"),
            ("app.py", f"from lib import fetch\ndef f():\n    fetch({Q})\n"),
        ),
        "finding",
    ),
    (
        "ssrf-ambig",
        SSRF,
        _files(
            "import requests\n"
            f"def helper(v):\n    requests.get(v)\ndef helper(v):\n    return v\ndef f():\n    helper({Q})\n"
        ),
        "gap",
    ),
    (
        "ssrf-partial",
        SSRF,
        _files("import requests\ndef f(\n    requests.get(request.args.get('q'))\n"),
        "gap",
    ),
    (
        "xss-direct",
        XSS,
        _files(f"from flask import Markup\ndef f():\n    return Markup({Q})\n"),
        "finding",
    ),
    (
        "xss-alias",
        XSS,
        _files(f"from flask import Markup\ndef f():\n    n = {Q}\n    return Markup(n)\n"),
        "finding",
    ),
    (
        "xss-wrap",
        XSS,
        _files(
            f"from flask import Markup\ndef wrap(v):\n    return v\ndef f():\n    return Markup(wrap({Q}))\n"
        ),
        "gap",
    ),
    (
        "xss-field",
        XSS,
        _files("from flask import Markup\ndef f():\n    return Markup(request.args['name'])\n"),
        "finding",
    ),
    (
        "xss-const",
        XSS,
        _files("from flask import Markup\ndef f():\n    return Markup('ok')\n"),
        "clean",
    ),
    (
        "xss-san",
        XSS,
        _files(
            f"import html\nfrom flask import Markup\ndef f():\n    return Markup(html.escape({Q}))\n"
        ),
        "clean",
    ),
    (
        "xss-shadow",
        XSS,
        _files(f"def Markup(v):\n    return v\ndef f():\n    return Markup({Q})\n"),
        "finding",
    ),
    (
        "xss-arg",
        XSS,
        _files(f"from flask import Markup\ndef f():\n    return Markup('ok', {Q})\n"),
        "finding",
    ),
    (
        "xss-cross",
        XSS,
        (
            ("lib.py", "from flask import Markup\ndef render(value):\n    return Markup(value)\n"),
            ("app.py", f"from lib import render\ndef f():\n    render({Q})\n"),
        ),
        "finding",
    ),
    (
        "xss-ambig",
        XSS,
        _files(
            "from flask import Markup\n"
            f"def helper(v):\n    return Markup(v)\ndef helper(v):\n    return v\ndef f():\n    helper({Q})\n"
        ),
        "gap",
    ),
    (
        "xss-partial",
        XSS,
        _files("from flask import Markup\ndef f(\n    return Markup(request.args.get('q'))\n"),
        "gap",
    ),
    ("deser-direct", DESER, _files(f"import pickle\ndef f():\n    pickle.loads({Q})\n"), "finding"),
    (
        "deser-alias",
        DESER,
        _files(f"import pickle\ndef f():\n    b = {Q}\n    pickle.loads(b)\n"),
        "finding",
    ),
    (
        "deser-wrap",
        DESER,
        _files(
            f"import pickle\ndef wrap(v):\n    return v\ndef f():\n    pickle.loads(wrap({Q}))\n"
        ),
        "gap",
    ),
    (
        "deser-field",
        DESER,
        _files("import pickle\ndef f():\n    pickle.loads(request.args['blob'])\n"),
        "finding",
    ),
    (
        "deser-const",
        DESER,
        _files("import pickle\ndef f():\n    pickle.loads(b'constant')\n"),
        "clean",
    ),
    ("deser-san", DESER, _files(f"import json\ndef f():\n    json.loads({Q})\n"), "clean"),
    (
        "deser-shadow",
        DESER,
        _files(f"def loads(b):\n    return b\ndef f():\n    loads({Q})\n"),
        "clean",
    ),
    (
        "deser-arg",
        DESER,
        _files(f"import pickle\ndef f():\n    pickle.loads(b'abc', {Q})\n"),
        "finding",
    ),
    (
        "deser-cross",
        DESER,
        (
            ("lib.py", "import pickle\ndef load(value):\n    pickle.loads(value)\n"),
            ("app.py", f"from lib import load\ndef f():\n    load({Q})\n"),
        ),
        "finding",
    ),
    (
        "deser-ambig",
        DESER,
        _files(
            "import pickle\n"
            f"def helper(v):\n    pickle.loads(v)\ndef helper(v):\n    return v\ndef f():\n    helper({Q})\n"
        ),
        "gap",
    ),
    (
        "deser-partial",
        DESER,
        _files("import pickle\ndef f(\n    pickle.loads(request.args.get('q'))\n"),
        "gap",
    ),
    ("eval-direct", EVAL, _files(f"def f():\n    eval({Q})\n"), "finding"),
    ("eval-alias", EVAL, _files(f"def f():\n    c = {Q}\n    eval(c)\n"), "finding"),
    (
        "eval-wrap",
        EVAL,
        _files(f"def wrap(v):\n    return v\ndef f():\n    eval(wrap({Q}))\n"),
        "gap",
    ),
    ("eval-field", EVAL, _files("def f():\n    eval(request.args['code'])\n"), "finding"),
    ("eval-const", EVAL, _files("def f():\n    eval('1')\n"), "clean"),
    ("eval-san", EVAL, _files(f"def f():\n    eval(str({Q}))\n"), "finding"),
    (
        "eval-shadow",
        EVAL,
        _files(f"def eval(c):\n    return c\ndef f():\n    eval({Q})\n"),
        "clean",
    ),
    ("eval-arg", EVAL, _files(f"def f():\n    eval('1', {{'x': {Q}}})\n"), "finding"),
    (
        "eval-cross",
        EVAL,
        (
            ("lib.py", "def run(value):\n    eval(value)\n"),
            ("app.py", f"from lib import run\ndef f():\n    run({Q})\n"),
        ),
        "finding",
    ),
    (
        "eval-ambig",
        EVAL,
        _files(
            f"def helper(v):\n    eval(v)\ndef helper(v):\n    return v\ndef f():\n    helper({Q})\n"
        ),
        "gap",
    ),
    ("eval-partial", EVAL, _files("def f(\n    eval(request.args.get('q'))\n"), "gap"),
    ("redir-direct", REDIR, _files(f"def f():\n    redirect({Q})\n"), "finding"),
    ("redir-alias", REDIR, _files(f"def f():\n    u = {Q}\n    redirect(u)\n"), "finding"),
    (
        "redir-wrap",
        REDIR,
        _files(f"def wrap(v):\n    return v\ndef f():\n    redirect(wrap({Q}))\n"),
        "gap",
    ),
    ("redir-field", REDIR, _files("def f():\n    redirect(request.args['next'])\n"), "finding"),
    ("redir-const", REDIR, _files("def f():\n    redirect('/home')\n"), "clean"),
    (
        "redir-san",
        REDIR,
        _files(
            f"def f():\n    u = {Q}\n"
            "    redirect(u if url_has_allowed_host_and_scheme(u, None) else '/')\n"
        ),
        "gap",
    ),
    (
        "redir-shadow",
        REDIR,
        _files(f"def redirect(u):\n    return u\ndef f():\n    redirect({Q})\n"),
        "finding",
    ),
    ("redir-arg", REDIR, _files(f"def f():\n    redirect('/home', {Q})\n"), "finding"),
    (
        "redir-cross",
        REDIR,
        (
            ("lib.py", "def go(value):\n    redirect(value)\n"),
            ("app.py", f"from lib import go\ndef f():\n    go({Q})\n"),
        ),
        "finding",
    ),
    (
        "redir-ambig",
        REDIR,
        _files(
            f"def helper(v):\n    redirect(v)\ndef helper(v):\n    return v\ndef f():\n    helper({Q})\n"
        ),
        "gap",
    ),
    ("redir-partial", REDIR, _files("def f(\n    redirect(request.args.get('q'))\n"), "gap"),
]


@pytest.mark.parametrize(
    ("case_id", "kind", "files", "expect"),
    CASES,
    ids=[case[0] for case in CASES],
)
def test_taint_edge(
    case_id: str,
    kind: VulnerabilityClass,
    files: tuple[tuple[str, str], ...],
    expect: str,
    tmp_path: Path,
) -> None:
    paths: list[Path] = []
    for name, source in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        paths.append(path)
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, paths)
    found = any(obs.vulnerability_class is kind for obs in result.observations)
    if expect == "finding":
        assert found
        return
    if expect == "clean":
        assert case_id not in GAPS
        assert not found
        return
    assert expect == "gap"
    assert case_id in GAPS
    assert GAPS[case_id]
    assert not found


def test_secret_indicator_edges(tmp_path: Path) -> None:
    def scan(name: str, source: str):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return SecurityAnalysisEngine().analyze_repository(path.parent, [path])

    nested = scan("cfg.py", 'config = {"password": "s3cret-production-value"}\n')
    assert any(obs.vulnerability_class is SECRET for obs in nested.observations)
    assert "s3cret-production-value" not in " ".join(
        obs.evidence_text for obs in nested.observations
    )
    pem = scan(
        "key.py",
        'KEY = """-----BEGIN PRIVATE KEY-----\nMIIEsupersecretmaterial\n-----END PRIVATE KEY-----"""\n',
    )
    assert any(obs.vulnerability_class is SECRET for obs in pem.observations)
    assert "MIIEsupersecretmaterial" not in " ".join(obs.evidence_text for obs in pem.observations)
    comment = scan("doc.py", '# API_KEY = "sk_live_91ab88cdef012345"\n')
    assert not any(obs.vulnerability_class is SECRET for obs in comment.observations)
    checksum = scan("sum.py", 'md5_checksum = "abcdef1234567890abcd"\n')
    assert not any(obs.vulnerability_class is SECRET for obs in checksum.observations)
